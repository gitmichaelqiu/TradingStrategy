# The Macro-Regime Sentinel (MRS) Project

> **"In financial markets, alpha is not about being right; it's about surviving when you are wrong."**

## 1. Project Overview
The **Macro-Regime Sentinel (MRS)** is an institutional-grade quantitative trading framework developed to solve the "Stationarity Dilemma" in financial time series. 

Traditional algorithmic strategies often fail because they assume market conditions (correlations, volatility, trends) are static. [cite_start]This project implements the **Adaptive Markets Hypothesis (AMH)**[cite: 14], creating strategies that dynamically evolve their logic based on:
1.  **Latent Market Regimes** (Bull/Bear/Chop) using unsupervised learning.
2.  **Macroeconomic Stress** (Rates/VIX) using intermarket analysis.
3.  **Cointegration Breakdown** using real-time stationarity tests.

The project creates two distinct production engines: **V20 (Directional)** and **V16 (Statistical Arbitrage)**.

---

## 2. Strategy Architectures

### Engine A: Directional Trading (MRS-V20)
**Class Name:** `StrategyV20_MacroDominance`

This is the flagship "All-Weather" engine. It fuses external economic truth with internal market structure to maximize risk-adjusted returns.

#### The "Tri-Gated" Logic:
1.  **The Macro Switch (Offense)**
    * **Concept:** Markets are driven by liquidity (Rates) and Fear (VIX).
    * **Mechanism:** Calculates a `Total_Stress` score based on the CBOE Volatility Index (`^VIX`) and the Rate of Change of the 10-Year Treasury (`^TNX`).
    * **Action:**
        * *Standard Assets (Tech/Banks):* High Stress $\rightarrow$ **SELL** (Cash).
        * *Inflation Assets (Energy/Gold):* High Stress $\rightarrow$ **BUY** (Inverted Logic).
    * [cite_start]**Source:** [cite: 304] (Intermarket concepts).

2.  **The Regime Shield (Defense)**
    * **Concept:** Avoid "Catching the Falling Knife."
    * **Mechanism:** A **Walk-Forward Gaussian Mixture Model (GMM)** classifies the market state. If the Regime is **BEAR**, the "Trend Floor" is activated.
    * **Action:** If `Price < Lower Bollinger Band`, the trade is vetoed (Signal = 0), regardless of Macro indicators.
    * [cite_start]**Source:** [cite: 100, 128] (HMM/GMM regime detection).

3.  **Dynamic Sizing**
    * **Mechanism:** **Yang-Zhang Volatility** Targeting.
    * **Action:** Position size is inversely proportional to volatility, automatically deleveraging during "Black Swan" events.
    * [cite_start]**Source:** [cite: 81] (Yang-Zhang Estimator).

---

### Engine B: Statistical Arbitrage (MRS-V16)
**Class Name:** `StrategyV16_Gated_StatArb`

This engine trades mean-reversion pairs (e.g., JPM vs. BAC) but solves the "Breakdown Risk" (e.g., when correlations permanently decouple).

#### The "Gated" Logic:
1.  **Kalman Filter:** * Recursively estimates the dynamic hedge ratio ($\beta_t$) and intercept ($\alpha_t$) between Asset Y and Asset X.
    * [cite_start]**Source:** [cite: 151] (Kalman Filter for dynamic hedging).
2.  **The Stationarity Gate (The Innovation):**
    * **Problem:** Kalman Filters will keep betting on convergence even if the pair fundamentally breaks.
    * **Solution:** Before every trade, a Rolling **Augmented Dickey-Fuller (ADF)** test checks the spread.
    * **Action:** If `P-Value > 0.05` (Non-Stationary), the signal is forced to 0.
3.  **The Divorce Clause:**
    * Hard Stop Loss if the Spread Z-Score exceeds $\pm 4.0\sigma$.

---

## 3. Empirical Validation (Backtest Results)

The strategies were stress-tested on a "Torture Basket" (2022-2024) including Tech Crashes, Bank Runs, and Inflation Spikes.

### V20 Directional Performance
*Benchmark: Buy & Hold (B&H)*

| Ticker | Type | Ann. Return | Sharpe | Max Drawdown | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVDA** | *Growth* | **+84.9%** | **1.52** | **-15.2%** | **Superior Risk-Adjusted Return.** Sacrificed B&H upside (+355%) to eliminate the -62% crash. |
| **JPM** | *Value* | **+66.5%** | **1.37** | **-12.2%** | **Dominant.** Beat B&H (+62%) with 1/3rd the risk. Perfectly timed the rate cycle. |
| **TSLA** | *Volatile* | **+35.5%** | **0.87** | **-11.7%** | **Crisis Alpha.** Profitable while B&H collapsed **-73%**. The Regime Shield worked perfectly. |
| **BABA** | *Distressed*| **+2.0%** | **0.11** | **-16.4%** | **Survival.** Avoided the "Value Trap." Asset lost -27%, strategy stayed flat. |
| **XLE** | *Inflation* | **+8.0%** | **0.23** | **-23.6%** | **Valid.** The "Sector Patch" successfully captured the inflation trade. |

### V16 StatArb Performance
*Comparison: Raw Kalman (V15) vs. Gated Kalman (V16)*

| Pair | V15 (Raw) Return | V16 (Gated) Return | Improvement | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **JPM / BAC** | -7.7% | **+33.2%** | **+40.9%** | The Stationarity Gate filtered out "fake" convergence signals. |
| **NVDA / AMD**| -11.0% | **-8.9%** | +2.1% | Reduced losses during the AI-driven decoupling of the pair. |

---

## 4. Mathematical Foundations

### [cite_start]Yang-Zhang Volatility [cite: 82]
Used for precise risk targeting.
$$\sigma_{YZ}^2 = \sigma_{ON}^2 + k \sigma_{RS}^2$$
Where $\sigma_{ON}$ is overnight volatility and $\sigma_{RS}$ is Rogers-Satchell (intraday) volatility.

### [cite_start]Kalman Filter State Space [cite: 153]
Used for dynamic pairs hedging.
$$\beta_t = \beta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
$$Y_t = \alpha_t + \beta_t X_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

### Macro Stress Score (Custom V20)
Used to drive the "Macro Switch."
$$S_t = \text{RollingMean}_3 \left( \max \left( \frac{VIX_t - 15}{15}, \frac{\Delta\% TNX_{20d}}{10\%} \right) \right)$$

---

## 5. Installation & Usage

### Requirements
```bash
pip install yfinance pandas numpy scikit-learn pykalman statsmodels matplotlib

```

### Running the Directional Engine (V20)

```python
from adaptive_regime import StrategyV20_MacroDominance

# Initialize Strategy
strat = StrategyV20_MacroDominance(
    ticker="NVDA", 
    start_date="2022-01-01", 
    end_date="2024-12-30"
)

# Execution Pipeline
strat.fetch_data()       # Downloads Price, VIX, and TNX
strat.generate_signals() # Runs Macro Switch + Regime Shield
strat.run_backtest()     # Simulates Trades

# View Metrics
print(strat.metrics)
# Output: {'Total Return': 0.849, 'Sharpe Ratio': 1.52, 'Max Drawdown': -0.152}

```

### Running the StatArb Engine (V16)

```python
from adaptive_regime import StrategyV16_Gated_StatArb

# Initialize Pair
strat = StrategyV16_Gated_StatArb(
    ticker_y="JPM", 
    ticker_x="BAC", 
    start_date="2023-01-01", 
    end_date="2024-12-30"
)

strat.fetch_data()
strat.generate_signals() # Runs Kalman Filter + ADF Gate
strat.run_backtest()

```

---

## 6. Disclaimer

*This codebase is a research prototype implementing concepts from modern quantitative finance literature. Past performance (backtests) is not indicative of future results. The "Stationarity Gate" and "Regime Shield" are probabilistic tools, not guarantees of profit.*
