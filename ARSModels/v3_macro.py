class StrategyV3_Macro(BaseStrategy):
    """V3: Macro (SPY) Filter."""
    def __init__(self, ticker, start_date, end_date):
        super().__init__(ticker, start_date, end_date)
        self.spy_data = None

    def fetch_data(self, warmup_years=2):
        super().fetch_data(warmup_years)
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=warmup_years*365)
        try:
            spy = yf.download("SPY", start=start_dt.strftime("%Y-%m-%d"), end=self.end_date, progress=False, auto_adjust=False)
            if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
            self.spy_data = spy[['Adj Close']].rename(columns={'Adj Close': 'SPY_Price'})
        except: pass

    def generate_signals(self):
        if self.data is None or self.data.empty: return
        df = self.data.copy()
        
        if self.spy_data is not None:
            df = df.join(self.spy_data, how='left')
            df['SPY_MA200'] = df['SPY_Price'].rolling(window=200).mean()
            df['Macro_Bull'] = df['SPY_Price'] > df['SPY_MA200']
        else:
            df['Macro_Bull'] = True
            
        df['Volatility'] = FeatureLab.yang_zhang_volatility(df)
        df['FracDiff'] = FeatureLab.frac_diff_fixed(df['Adj Close'].apply(np.log), d=0.4, window=50)
        df['RSI'] = FeatureLab.compute_rsi(df['Adj Close'])
        df.dropna(inplace=True)
        
        df['Signal'] = 0
        df.loc[(df['FracDiff'] > 0), 'Signal'] = 1
        df.loc[df['Macro_Bull'] == False, 'Signal'] = 0
        
        target_vol = 0.15 / np.sqrt(252)
        df['Signal'] = df['Signal'] * (target_vol / df['Volatility']).clip(upper=1.5)
        self.data = df