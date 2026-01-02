class StrategyV4_Meta(BaseStrategy):
    """V4: Dynamic Profiling with OBV."""
    def generate_signals(self):
        if self.data is None or self.data.empty: return
        df = self.data.copy()
        
        df['Volatility'] = FeatureLab.yang_zhang_volatility(df)
        df['FracDiff'] = FeatureLab.frac_diff_fixed(df['Adj Close'].apply(np.log), d=0.4, window=50)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_Trend'] = df['OBV'].rolling(50).mean()
        df['RSI'] = FeatureLab.compute_rsi(df['Adj Close'])
        df.dropna(inplace=True)
        
        df['Signal'] = 0
        # Trend
        df.loc[(df['FracDiff'] > 0) & (df['OBV'] > df['OBV_Trend']), 'Signal'] = 1
        # Reversion
        df.loc[(df['RSI'] < 30), 'Signal'] = 1
        
        target_vol = 0.15 / np.sqrt(252)
        df['Signal'] = df['Signal'] * (target_vol / df['Volatility']).clip(upper=1.5)
        self.data = df