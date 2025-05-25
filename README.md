# Description

This project simulates a rolling window strategy optimization and deployment for crypto futures trading. It uses historical Binance Futures data to:

- Collect and prepare 15-minute candlestick data for specified symbols
- Perform a rolling 7-day window genetic algorithm (GA) optimization to find the best Stop Loss (SL) and Take Profit (TP) parameters daily
- Backtest the optimized parameters on the following day’s data to evaluate performance
- Repeat the process day-by-day to simulate a live trading deployment workflow

The strategy optimization focuses on maximizing a custom weighted fitness score incorporating return, Sharpe ratio, Sortino ratio, Calmar ratio, win rate, profit factor, and drawdown penalties.

This approach helps model how a real-world trader or automated system might continuously adapt strategy parameters over time, accounting for market changes and avoiding overfitting by always testing on unseen future data.

---

## Features

- Rolling 7-day window GA optimization of SL and TP parameters
- Backtesting using `backtesting.py` library on Binance Futures historical data
- Uses DEAP genetic algorithm framework for strategy parameter search
- Multiprocessing support to speed up GA evaluations
- Detailed daily performance reporting with average returns and statistics

---

## Use Cases

- Understand rolling window strategy optimization concepts in quantitative trading
- Experiment with genetic algorithm parameter tuning for trading strategies
- Simulate strategy deployment workflows with live-like evaluation

---

## Technologies

- Python 3
- Binance API
- DEAP (Genetic Algorithms)
- backtesting.py (Backtesting Framework)
- pandas, numpy for data manipulation
- tqdm for progress bars


abhinav00345@gmail.com
abhinav00345@gmail.com
