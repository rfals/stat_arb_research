# stat_arb_research: Exploring Statistical Arbitrage Strategy Potential for 10 Cryptocurrencies
 
**stat_arb_research** is a repository dedicated to testing and analyzing two statistical arbitrage strategies on the top 10 cryptocurrencies by market cap, as listed on [coinmarketcap.com](https://coinmarketcap.com/) .

## Data:
The data for this project is obtained from [Kaggle](https://www.kaggle.com/datasets/jorijnsmit/binance-full-history) and contains 1 minute OHLCV data for the top 10 cryptocurrencies by market cap, as listed on [coinmarketcap.com](https://coinmarketcap.com/) . The data is available from 2017-08-17 04:00:00 to 2021-08-17 04:00:00.
<br>
<br>
The data necessary for this project can also be obtained from [Binance](https://www.binance.com/en/binance-api) API, using the *binanceDataGather.py* script, but it should be noted that the process is very time-consuming due to the rate limits imposed by the API. The script is provided for reference purposes only.

## Strategies:

 1. **Cointegrated Pairs:** This project focuses on developing a straightforward application to apply a simple mean-reversion strategy tailored to cryptocurrencies, specifically 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'SOLUSDT', 'DOTUSDT', and 'LTCUSDT'. Firstly, a VAR model is built using cryptocurrency price returns as the initial step before conducting IRF (Impulse Response Function) analysis and Granger Causality tests to obtain preliminary clues about the most interesting pair trading combinations. Secondly, the 2-step Engle-Granger method is used to provide a formal framework for shortlisting the best pair combinations, with the first stage based on testing each pair for cointegration, and the second step allowing for the testing of the existence of an ECM (Error Correction Mechanism) that brings the cointegrated pair ultimately towards a long-term equilibrium. The results obtained significantly downsized the pair combinations more likely to be successful, with one of the best choices emerging from the analysis. The last section delves into the backtesting of the selected pair, starting by identifying key strategy parameters using an Ornstein-Uhlenbeck (OU) process to model the spread.

 2. **Top3 Flop3:** This project is inspired by [Fischer, Krauss & Deinert (2019)](file:///C:/Users/ReinisFals/Downloads/jrfm-12-00031-v2.pdf) article exploring mature financial market machine-learning based statistical arbitrage strategies [Takeuchi and Lee (2013)](https://cs229.stanford.edu/proj2013/TakeuchiLee-ApplyingDeepLearningToEnhanceMomentumTradingStrategiesInStocks.pdf) in the cryptocurrency space. Specifically, I train a random forest on lagged returns of the 10 cryptocurrency coins, with the objective to predict whether a coin outperforms the cross-sectional median of all 10 coins over the subsequent 120 min. I buy the coins with the top-3 predictions and short-sell the coins with the flop-3 predictions, only to reverse the positions after 120 min.

## Repository Structure:
```plaintext
stat_arb_research/
│
├── 1 Cointegrated Pairs/
│   ├── EDA & Data Cleaning/
│   ├── Research/
│   └── __pycache__/
│
├── 2 HFT Strategies/
│   ├── top3_flop3/
│   └── __pycache__/
│
└── data/
    ├── Binance API Data/
    │   └── functions/
    └── Kaggle Binance Data/ 
```
<br>

## Requirements:

**Using pip**
    
    ```bash
    pip install -r requirements.txt
    ```

**Using conda**
    
    ```bash
    conda env create -f stat_arb_env.yml
    conda activate stat_arb_env
    ```
