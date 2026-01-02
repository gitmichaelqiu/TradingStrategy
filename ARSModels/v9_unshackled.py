class StrategyV9_RegimeUnshackled(BaseStrategy):
    """
    V9: The 'Unshackled' Regime Model.
    
    Improvements over V8:
    1. TRUST THE BULL (Fixes JPM):
       - If GMM says 'Stable Bull', we go Long immediately.
       - Kalman Slope is demoted from a 'Gatekeeper' to a 'Sizing Booster'.
       
    2. THE ALPHA CLAUSE (Fixes NVDA):
       - If Stock is Bullish but SPY is Bearish (Macro Divergence), we do NOT exit.
       - We trade at 50% size. This captures relative strength leaders early.
       
    3. SURVIVAL MODE (Keeps BABA safe):
       - If GMM says 'Bear', we hard-exit to Cash (unless Deep Value RSI < 25).
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
        P, Q, R = 1.0, 0.001, 0.1
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
        
        # --- 1. Features ---
        if self.spy_data is not None:
            df = df.join(self.spy_data, how='left').fillna(method='ffill')
        else: df['Macro_Trend'] = 1
            
        df['Volatility'] = FeatureLab.yang_zhang_volatility(df)
        df['Kalman_Slope'] = self._apply_kalman_filter(np.log(df['Adj Close']))
        df['RSI'] = FeatureLab.compute_rsi(df['Adj Close'])
        df['Returns_Smoothed'] = df['Returns'].rolling(5).mean()
        df['Vol_Smoothed'] = df['Volatility'].rolling(5).mean()
        df.dropna(inplace=True)
        
        # --- 2. GMM Regime Detection ---
        X = df[['Returns_Smoothed', 'Vol_Smoothed']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # We use warm_start=True logic simulation by consistent random_state
        gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
        df['Cluster'] = gmm.fit_predict(X_scaled)
        
        # Dynamic Mapping
        stats = df.groupby('Cluster')['Returns_Smoothed'].mean().sort_values()
        bear_cluster = stats.index[0]
        chop_cluster = stats.index[1]
        bull_cluster = stats.index[2]
        
        conditions = [
            (df['Cluster'] == bull_cluster),
            (df['Cluster'] == bear_cluster),
            (df['Cluster'] == chop_cluster)
        ]
        choices = ['BULL', 'BEAR', 'CHOP']
        df['Regime_Type'] = np.select(conditions, choices, default='CHOP')
        
        # --- 3. Unshackled Signal Logic ---
        df['Signal'] = 0.0
        
        # A. BULL REGIME: "Trust The Trend"
        # If GMM says Bull, we are Long. Period.
        # This captures JPM's "slow grind" that Kalman missed.
        bull_signal = (df['Regime_Type'] == 'BULL')
        df.loc[bull_signal, 'Signal'] = 1.0
        
        # Boost: If Kalman agrees (Strong Trend), we go 1.3x leverage
        strong_trend = bull_signal & (df['Kalman_Slope'] > 0)
        df.loc[strong_trend, 'Signal'] = 1.3
        
        # B. CHOP REGIME: "Active Trading"
        # Buy Dips.
        chop_buy = (df['Regime_Type'] == 'CHOP') & (df['RSI'] < 45)
        df.loc[chop_buy, 'Signal'] = 1.0
        
        # C. BEAR REGIME: "Survival"
        # Cash is King. Only buy extreme panic.
        panic_buy = (df['Regime_Type'] == 'BEAR') & (df['RSI'] < 25)
        df.loc[panic_buy, 'Signal'] = 1.0
        
        # --- 4. The Alpha Clause (Macro Handling) ---
        
        # Standard Volatility Sizing (Risk Parity)
        target_vol = 0.15 / np.sqrt(252)
        vol_scaler = (target_vol / df['Volatility']).clip(upper=1.5)
        
        # Macro Logic:
        # If SPY is Bearish (0), we don't kill the trade. We just HALVE it.
        # This allows NVDA to run while still being defensive.
        macro_scaler = df['Macro_Trend'].map({1: 1.0, 0: 0.5})
        
        df['Signal'] = df['Signal'] * vol_scaler * macro_scaler
        self.data = df