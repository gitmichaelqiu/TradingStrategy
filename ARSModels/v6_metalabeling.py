class StrategyV6_MetaLabeling(BaseStrategy):
    """
    V6.1 (Hybrid): The 'Regime-Adaptive' Institutional Model.
    
    Architecture:
    1. Primary Signal (Hybrid): 
       - TREND: Kalman Slope > 0 (Catch the run)
       - VALUE: RSI < 30 (Catch the dip)
       This ensures we have candidates in both trending and chopping markets.
       
    2. Meta-Labeling (Random Forest): 
       - Learns WHICH of the above signals works for the current asset/regime.
       
    3. Soft-Sizing: Scales leverage based on ML confidence.
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
        # Kalman Params
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
        df['Spread'] = df['Adj Close'] - df['Adj Close'].rolling(20).mean()
        df.dropna(inplace=True)
        
        # --- 2. Primary Signal (The Hybrid Generator) ---
        df['Primary_Signal'] = 0
        
        # A. MOMENTUM LEG (For NVDA/SPY)
        # Catch the trend when slope is positive
        trend_signal = (df['Kalman_Slope'] > 0)
        
        # B. MEAN REVERSION LEG (For JPM/Chop)
        # Catch the knife when oversold (Value)
        value_signal = (df['RSI'] < 30)
        
        # Combine: We are interested if EITHER is true
        df.loc[trend_signal | value_signal, 'Primary_Signal'] = 1
        
        # --- 3. Meta-Labeling (The Validator) ---
        # Label: Did buying here result in profit?
        labels = FeatureLab.triple_barrier_labels(df['Adj Close'], df['Volatility'], pt=1.0, sl=1.0, barrier_window=10)
        
        df['Meta_Prob'] = 0.5
        train_window = 252 * 2
        update_freq = 63 
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        feature_cols = ['Volatility', 'RSI', 'Spread', 'Kalman_Slope']
        
        indices = df.index
        if len(df) > train_window:
            for t in range(train_window, len(df), update_freq):
                train_start = indices[t - train_window]
                train_end = indices[t]
                test_end_idx = min(t + update_freq, len(df))
                test_end = indices[test_end_idx - 1]
                
                X_train = df.loc[train_start:train_end, feature_cols]
                y_train = labels.loc[train_start:train_end]
                
                # Training on all data allows the model to learn "High RSI = Good" for NVDA
                # and "Low RSI = Good" for JPM automatically based on recent history.
                clf.fit(X_train, y_train)
                
                X_test = df.loc[train_end:test_end, feature_cols]
                probs = clf.predict_proba(X_test)
                
                if probs.shape[1] == 2:
                    pos_probs = probs[:, 1]
                else:
                    pos_probs = probs[:, 0] if clf.classes_[0] == 1 else 0.0
                    
                df.loc[train_end:test_end, 'Meta_Prob'] = pos_probs
        
        # --- 4. Signal Construction ---
        df['Signal'] = 0.0
        
        # Confidence Floor: 
        # If the ML confirms the hybrid signal (Prob > 0.45), we execute.
        # This allows RSI Dips to pass IF the ML thinks they are profitable.
        active_trade = (df['Primary_Signal'] == 1) & (df['Meta_Prob'] > 0.45)
        df.loc[active_trade, 'Signal'] = 1
        
        # Sizing (Volatility + Confidence)
        target_vol = 0.15 / np.sqrt(252)
        vol_scaler = (target_vol / df['Volatility']).clip(upper=2.0)
        ml_scaler = (df['Meta_Prob'] / 0.5).clip(0.5, 2.0)
        
        # Macro Override
        # If Bear Market, we are defensive, BUT we allow Deep Value (RSI < 30) 
        # to have slightly more room if the ML loves it.
        macro_scaler = df['Macro_Trend'].map({1: 1.0, 0: 0.5})
        
        df['Signal'] = df['Signal'] * vol_scaler * ml_scaler * macro_scaler
        
        self.data = df