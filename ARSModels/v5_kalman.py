class StrategyV5_KalmanState(BaseStrategy):
    """
    V5 (Formerly V10): Kalman Filter + Macro Filter + Volatility Burst Control.
    Uses Kalman Filter for noise-free slope estimation[cite: 151].
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
        
        if self.spy_data is not None:
            df = df.join(self.spy_data, how='left').fillna(method='ffill')
        else:
            df['Macro_Trend'] = 1 
            
        log_prices = np.log(df['Adj Close'])
        df['Kalman_Slope'] = self._apply_kalman_filter(log_prices)
        df['Volatility'] = FeatureLab.yang_zhang_volatility(df)
        df['Vol_Change'] = df['Volatility'].diff()
        
        df.dropna(inplace=True)
        
        # Primary Logic
        df['Signal'] = 0.0
        long_condition = (df['Kalman_Slope'] > 0) & (df['Macro_Trend'] == 1)
        df.loc[long_condition, 'Signal'] = 1
        
        # Vol Targeting & Burst Protection
        target_vol = 0.15 / np.sqrt(252)
        df['Vol_Scaler'] = (target_vol / df['Volatility']).clip(upper=1.5)
        
        vol_spike = df['Vol_Change'] > df['Vol_Change'].rolling(20).std() * 2
        df.loc[vol_spike, 'Vol_Scaler'] *= 0.5
        df.loc[df['Macro_Trend'] == 0, 'Vol_Scaler'] *= 0.5
        
        df['Signal'] = df['Signal'] * df['Vol_Scaler']
        self.data = df