class StrategyV7_AdaptiveOptim(BaseStrategy):
    """
    V7 (WFO): Walk-Forward Optimized Strategy.
    
    Instead of static rules, this strategy runs a 'Tournament' every quarter.
    It tests 4 distinct parameter sets (Profiles) on the past 252 days:
    
    1. Trend_Aggro: Kalman Slope > 0 (No Macro Filter)
    2. Trend_Defense: Kalman Slope > 0 AND Macro_Bull (Like V3)
    3. Reversion_Deep: RSI < 30 (Buying Crashes)
    4. Reversion_Active: RSI < 45 (Buying Dips)
    
    It selects the Profile with the highest Sharpe Ratio in the lookback window
    and uses it for the next execution window.
    """
    def __init__(self, ticker, start_date, end_date):
        super().__init__(ticker, start_date, end_date)
        self.spy_data = None

    def fetch_data(self, warmup_years=2):
        super().fetch_data(warmup_years)
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=warmup_years*365)
        try:
            spy = yf.download("SPY", start=start_dt.strftime("%Y-%m-%d"), end=self.end_date, progress=False, auto_adjust=False)
            if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
            spy['Macro_Trend'] = (spy['Adj Close'] > spy['Adj Close'].rolling(200).mean()).astype(int)
            self.spy_data = spy[['Macro_Trend']]
        except: pass

    def _apply_kalman_filter(self, prices):
        x = prices.values
        n = len(x)
        state = np.zeros(n)
        slope = np.zeros(n)
        state[0] = x[0]
        P, Q, R = 1.0, 0.01, 0.1 # Q=0.01 makes it slightly more responsive than V6
        
        for t in range(1, n):
            pred_state = state[t-1] + slope[t-1]
            pred_P = P + Q
            measurement = x[t]
            residual = measurement - pred_state
            
            K = pred_P / (pred_P + R)
            state[t] = pred_state + K * residual
            slope[t] = 0.9 * slope[t-1] + 0.1 * (state[t] - state[t-1])
            P = (1 - K) * pred_P
            
        return pd.Series(slope, index=prices.index)

    def generate_signals(self):
        if self.data is None or self.data.empty: return
        df = self.data.copy()
        
        # --- 1. Global Feature Engineering ---
        if self.spy_data is not None:
            df = df.join(self.spy_data, how='left').fillna(method='ffill')
        else: df['Macro_Trend'] = 1
            
        df['Volatility'] = FeatureLab.yang_zhang_volatility(df)
        df['Kalman_Slope'] = self._apply_kalman_filter(np.log(df['Adj Close']))
        df['RSI'] = FeatureLab.compute_rsi(df['Adj Close'])
        df.dropna(inplace=True)
        
        # --- 2. Pre-Calculate Strategy Candidates (Vectorized) ---
        # We calculate the raw signals for all profiles upfront
        
        # Profile 1: Aggressive Trend (Chase the move)
        sig_trend_aggro = (df['Kalman_Slope'] > 0).astype(int)
        
        # Profile 2: Defensive Trend (V3 Style - Only if Macro agrees)
        sig_trend_def = ((df['Kalman_Slope'] > 0) & (df['Macro_Trend'] == 1)).astype(int)
        
        # Profile 3: Deep Reversion (Catch Falling Knife)
        sig_rev_deep = (df['RSI'] < 30).astype(int)
        
        # Profile 4: Active Reversion (Buy Shallow Dips)
        sig_rev_active = (df['RSI'] < 45).astype(int)
        
        # Store in a dict for easy access
        candidates = {
            'Trend_Aggro': sig_trend_aggro,
            'Trend_Defense': sig_trend_def,
            'Rev_Deep': sig_rev_deep,
            'Rev_Active': sig_rev_active
        }
        
        # --- 3. Walk-Forward Optimization Loop ---
        df['Signal'] = 0.0
        df['Selected_Profile'] = 'None' # For debugging/analysis
        
        lookback = 252       # 1 Year Lookback for Optimization
        rebalance_freq = 63  # Quarterly Re-optimization
        
        indices = df.index
        daily_returns = df['Returns']
        
        if len(df) > lookback:
            for t in range(lookback, len(df), rebalance_freq):
                train_start = indices[t - lookback]
                train_end = indices[t]
                test_end_idx = min(t + rebalance_freq, len(df))
                test_end = indices[test_end_idx - 1]
                
                # The Tournament: Check Sharpe of each candidate in lookback period
                best_score = -999
                best_profile = 'Trend_Defense' # Default safety
                
                lb_returns = daily_returns.loc[train_start:train_end]
                
                for name, sig_series in candidates.items():
                    # Simulate Strategy Return in Lookback
                    # Lag signal by 1 to avoid lookahead in backtest
                    sigs = sig_series.loc[train_start:train_end].shift(1).fillna(0)
                    strat_ret = lb_returns * sigs
                    
                    # Calculate Metric (Sharpe)
                    mean_ret = strat_ret.mean()
                    std_ret = strat_ret.std()
                    
                    if std_ret > 1e-6:
                        score = mean_ret / std_ret # Simple Sharpe
                    else:
                        score = -999 # Flat line is bad
                        
                    if score > best_score:
                        best_score = score
                        best_profile = name
                
                # Apply Best Profile to Next Window (Test Set)
                # We use the signal series for the *future* window based on the *past* winner
                winner_signals = candidates[best_profile].loc[train_end:test_end]
                df.loc[train_end:test_end, 'Signal'] = winner_signals
                df.loc[train_end:test_end, 'Selected_Profile'] = best_profile

        # --- 4. Volatility Targeting (Risk Management) ---
        target_vol = 0.15 / np.sqrt(252)
        vol_scaler = (target_vol / df['Volatility']).clip(upper=1.5)
        
        df['Signal'] = df['Signal'] * vol_scaler
        self.data = df