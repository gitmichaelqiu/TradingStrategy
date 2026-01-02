class StrategyV2_Advanced(BaseStrategy):
    """V2: Rolling GMM."""
    def generate_signals(self):
        if self.data is None or self.data.empty: return
        df = self.data.copy()
        
        df['Volatility'] = FeatureLab.yang_zhang_volatility(df)
        df['FracDiff'] = FeatureLab.frac_diff_fixed(df['Adj Close'].apply(np.log), d=0.4, window=50)
        df['RSI'] = FeatureLab.compute_rsi(df['Adj Close'])
        df['Returns_Smoothed'] = df['Returns'].rolling(5).mean()
        df['Vol_Smoothed'] = df['Volatility'].rolling(5).mean()
        df.dropna(inplace=True)
        
        df['Regime'] = 0
        window_size, step_size = 504, 126
        preds, indices = [], []
        
        if len(df) > window_size:
            for t in range(window_size, len(df), step_size):
                train = df.iloc[t-window_size:t]
                test = df.iloc[t:t+step_size]
                if test.empty: break
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(train[['Returns_Smoothed', 'Vol_Smoothed']].values)
                X_test_s = scaler.transform(test[['Returns_Smoothed', 'Vol_Smoothed']].values)
                
                gmm = GaussianMixture(n_components=3, random_state=42).fit(X_train_s)
                train['Clust'] = gmm.predict(X_train_s)
                stats = train.groupby('Clust')['Returns_Smoothed'].mean().sort_values().index
                mapping = {stats[0]: -1, stats[1]: 0, stats[2]: 1}
                
                preds.extend([mapping[x] for x in gmm.predict(X_test_s)])
                indices.extend(test.index)
            
            df.loc[indices, 'Regime'] = pd.Series(preds, index=indices)
        
        df['Signal'] = 0
        df.loc[(df['Regime'] == 1) & (df['FracDiff'] > 0), 'Signal'] = 1
        df.loc[(df['Regime'] == 0) & (df['RSI'] < 45), 'Signal'] = 1
        
        target_vol = 0.15 / np.sqrt(252)
        df['Vol_Scaler'] = (target_vol / df['Volatility']).clip(upper=1.0)
        df['Signal'] = df['Signal'] * df['Vol_Scaler']
        self.data = df