class StrategyV8_GrandUnification(BaseStrategy):
    """
    V8: The Regime-Adaptive Ensemble.
    
    It combines the strengths of previous iterations:
    1. UNSUPERVISED STATE DETECTION (from V1): 
       Uses Gaussian Mixture Models (GMM) to classify the market into 3 regimes:
       - Low Vol / High Return -> "Stable Bull"
       - High Vol / Negative Return -> "Crisis/Bear"
       - Medium Vol / Flat Return -> "Chop"
       
    2. CONDITIONAL LOGIC (from V6):
       - If State == Stable Bull: Deploy AGGRESSIVE TREND (Kalman Slope).
       - If State == Chop: Deploy ACTIVE REVERSION (RSI < 45).
       - If State == Crisis: Go CASH (or Deep Value RSI < 25 only).
       
    3. GLOBAL SAFETY (from V3):
       - Overrides everything if SPY is below 200 SMA (Systemic Risk).
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
        
        # --- 1. Data & Features ---
        if self.spy_data is not None:
            df = df.join(self.spy_data, how='left').fillna(method='ffill')
        else: df['Macro_Trend'] = 1
            
        df['Volatility'] = FeatureLab.yang_zhang_volatility(df)
        df['Kalman_Slope'] = self._apply_kalman_filter(np.log(df['Adj Close']))
        df['RSI'] = FeatureLab.compute_rsi(df['Adj Close'])
        df['Returns_Smoothed'] = df['Returns'].rolling(5).mean()
        df['Vol_Smoothed'] = df['Volatility'].rolling(5).mean()
        df.dropna(inplace=True)
        
        # --- 2. GMM Regime Detection (The "Brain") ---
        # We cluster the market into 3 states based on Return & Risk
        X = df[['Returns_Smoothed', 'Vol_Smoothed']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train GMM
        gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
        df['Cluster'] = gmm.fit_predict(X_scaled)
        
        # --- 3. Dynamic Profile Mapping ---
        # We must figure out which cluster is which (Bull vs Bear vs Chop)
        # We assume:
        # - Highest Return = Bull
        # - Lowest Return = Bear
        # - Middle = Chop
        
        stats = df.groupby('Cluster')['Returns_Smoothed'].mean().sort_values()
        bear_cluster = stats.index[0]
        chop_cluster = stats.index[1]
        bull_cluster = stats.index[2]
        
        # Map clusters to readable names
        conditions = [
            (df['Cluster'] == bull_cluster),
            (df['Cluster'] == bear_cluster),
            (df['Cluster'] == chop_cluster)
        ]
        choices = ['BULL', 'BEAR', 'CHOP']
        df['Regime_Type'] = np.select(conditions, choices, default='CHOP')
        
        # --- 4. Regime-Conditional Signal Logic ---
        df['Signal'] = 0.0
        
        # A. BULL REGIME: Trend Following
        # Use Kalman Slope. If Slope > 0, we ride.
        # We ignore RSI overbought because strong trends stay overbought.
        bull_signal = (df['Regime_Type'] == 'BULL') & (df['Kalman_Slope'] > 0)
        df.loc[bull_signal, 'Signal'] = 1
        
        # B. CHOP REGIME: Mean Reversion
        # Market is going nowhere. Buy dips.
        # RSI < 45 is a good entry in chop.
        chop_signal = (df['Regime_Type'] == 'CHOP') & (df['RSI'] < 45)
        df.loc[chop_signal, 'Signal'] = 1
        
        # C. BEAR REGIME: Cash / Deep Value
        # Mostly Cash. Only buy EXTREME panic (RSI < 25).
        # This saved V1 on BABA.
        bear_signal = (df['Regime_Type'] == 'BEAR') & (df['RSI'] < 25)
        df.loc[bear_signal, 'Signal'] = 1
        
        # --- 5. Global Safety Filters ---
        
        # Filter 1: Macro Override (V3)
        # If the broad market is crashing (SPY < 200MA), reduce all long exposure by 50%
        # or cut entirely if it's a Bear stock in a Bear market.
        if 'Macro_Trend' in df.columns:
            # If Macro is bad, we kill the 'Chop' and 'Bull' signals for beta stocks
            # but we might keep 'Deep Value' signals.
            # For simplicity: Scale down everything.
            df.loc[df['Macro_Trend'] == 0, 'Signal'] *= 0.0 
            # STRICT RULE: No longs in Bear Market. This mimics V3's success.
        
        # Filter 2: Volatility Targeting
        target_vol = 0.15 / np.sqrt(252)
        vol_scaler = (target_vol / df['Volatility']).clip(upper=1.5)
        
        df['Signal'] = df['Signal'] * vol_scaler
        self.data = df