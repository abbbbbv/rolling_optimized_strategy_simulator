#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random
import multiprocessing
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from deap import base, creator, tools, algorithms
from backtesting import Backtest, Strategy
from binance.client import Client
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
client = Client('key', 'secret')

futures_list = ["ETHUSDT"]

def collect_futures_data(start_time, end_time):
    start_datetime = datetime.strptime(start_time, "%d %b %Y %H:%M:%S")
    extended_start_time = (start_datetime - timedelta(days=7)).strftime("%d %b %Y %H:%M:%S")

    if not os.path.exists('futures_data'):
        os.makedirs('futures_data')
    results = []
    for symbol in tqdm(futures_list, desc="Downloading Futures Data"):
        try:
            klines = client.futures_historical_klines_generator(
                symbol,
                Client.KLINE_INTERVAL_15MINUTE,
                extended_start_time,  # Use extended start time
                end_time
            )
            filename = f"futures_data/{symbol.lower()}_15m_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(filename, 'w') as file:
                for kline in klines:
                    data_point = f"{kline[0]},{kline[1]},{kline[2]},{kline[3]},{kline[4]},{kline[5]},{kline[6]}"
                    file.write(data_point + "\n")
            results.append(filename)
            time.sleep(0.5)
        except Exception as e:
            print(f"\nError downloading {symbol}: {str(e)}")
            continue
    return results

def prepare_data(filename):
    df = pd.read_csv(filename, header=None,
                     names=['timestamp','open','high','low','close','volume','next_timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.drop(columns=['next_timestamp'], inplace=True)
    df.columns = ['Open','High','Low','Close','Volume']
    return df

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def initialize_ga():
    toolbox = base.Toolbox()
    def gen_sl(): 
        return round(random.uniform(0.6, 3.5), 2)
    def gen_tp(): 
        return round(random.uniform(0.6, 3.5), 2)
    def create_ind():
        return [gen_sl(), gen_tp()]
    toolbox.register("individual", tools.initIterate, creator.Individual, create_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox

def mutate_ind(ind):
    mutation_rate = 0.2
    for i in range(len(ind)):
        if random.random() < mutation_rate:
            change = random.uniform(-0.02, 0.02)
            val = ind[i] + change
            ind[i] = round(max(0.6, min(3.5, val)), 2)
    return ind,

def evaluate_strat(ind, df, init_cash=10000, comm=0.00075):
    try:
        class CandlePatternStrategy(Strategy):
            def init(self):
                self.sl = ind[0]
                self.tp = ind[1]
            def next(self):
                if self.data.Open[-2] == self.data.High[-2]:
                    if self.position.is_long:
                        self.position.close()
                    self.sell(sl=self.data.Close[-1]*(1+self.sl/100),
                              tp=self.data.Close[-1]*(1-self.tp/100))
                elif self.data.Open[-2] == self.data.Low[-2]:
                    if self.position.is_short:
                        self.position.close()
                    self.buy(sl=self.data.Close[-1]*(1-self.sl/100),
                             tp=self.data.Close[-1]*(1+self.tp/100))
        bt = Backtest(df, CandlePatternStrategy, cash=init_cash, commission=comm, exclusive_orders=True)
        stats = bt.run()
        if stats['# Trades'] < 2:
            return (-999999,)
        
        def safe_get(val):
            v = 0 if pd.isna(val) or val is None else float(val)
            if np.isinf(v):
                return 0
            return v

        ret = safe_get(stats['Return [%]'])
        sharpe = safe_get(stats['Sharpe Ratio'])
        sortino = safe_get(stats['Sortino Ratio'])
        calmar = safe_get(stats['Calmar Ratio'])
        max_dd = safe_get(stats['Max. Drawdown [%]'])
        win_rate = safe_get(stats['Win Rate [%]'])
        profit_factor = safe_get(stats['Profit Factor'])
        sqn = safe_get(stats['SQN'])
        expectancy = safe_get(stats['Expectancy [%]'])
        kelly = safe_get(stats['Kelly Criterion'])
        
        # Weighted scoring
        weights = {
            'return': 0.25,
            'sharpe': 0.15,
            'sortino': 0.15,
            'calmar': 0.15,
            'profit_factor': 0.10,
            'win_rate': 0.10,
            'drawdown_penalty': 0.15
        }

        def normalize(value, scale_factor, min_val=0, max_val=10):
            return max(min(value * scale_factor, max_val), min_val)

        score_components = {
            'return': normalize(ret, 0.5) * weights['return'],
            'sharpe': normalize(sharpe, 2) * weights['sharpe'],
            'sortino': normalize(sortino, 1.5) * weights['sortino'],
            'calmar': normalize(calmar, 1) * weights['calmar'],
            'profit_factor': normalize(profit_factor, 3) * weights['profit_factor'],
            'win_rate': normalize(win_rate, 0.1) * weights['win_rate'],
            'drawdown_penalty': normalize(-max_dd, 1) * weights['drawdown_penalty']
        }

        total_score = sum(score_components.values()) / sum(weights.values())

        if np.isinf(total_score):
            return (-999999,)
        return (total_score,)
    except Exception:
        return (-999999,)

def optimize_symbol_ga(df):
    toolbox = initialize_ga()
    toolbox.register("evaluate", evaluate_strat, df=df)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_ind)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    pop_size = 50
    n_gen = 40
    population = toolbox.population(n=pop_size)
    stats_ga = tools.Statistics(lambda x: x.fitness.values)
    stats_ga.register("avg", np.mean)
    stats_ga.register("max", np.max)
    for _ in range(n_gen):
        algorithms.eaSimple(
            population, toolbox,
            cxpb=0.7, mutpb=0.2,
            ngen=1, stats=stats_ga,
            verbose=False
        )
    pool.close()
    best = tools.selBest(population, k=1)[0]
    return best, best.fitness.values[0]

def symbol_optimize_multiple_runs(df, runs=3):
    best_ind = None
    best_val = -999999
    for _ in range(runs):
        candidate, val = optimize_symbol_ga(df)
        if val > best_val:
            best_val = val
            best_ind = candidate
    return best_ind, best_val

def backtest_sl_tp(df, sl, tp):
    class FinalStrategy(Strategy):
        def init(self):
            self.sl = sl
            self.tp = tp
        def next(self):
            if self.data.Open[-2] == self.data.High[-2]:
                if self.position.is_long:
                    self.position.close()
                self.sell(sl=self.data.Close[-1]*(1+self.sl/100),
                          tp=self.data.Close[-1]*(1-self.tp/100))
            elif self.data.Open[-2] == self.data.Low[-2]:
                if self.position.is_short:
                    self.position.close()
                self.buy(sl=self.data.Close[-1]*(1-self.sl/100),
                         tp=self.data.Close[-1]*(1+self.tp/100))
    bt = Backtest(df, FinalStrategy, cash=10000, commission=0.00075, exclusive_orders=True)
    stats = bt.run()
    return stats
def safe_float(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def main():
    start_time = "08 Mar 2025 00:00:00"
    end_time = "22 Mar 2025 23:59:59"
    data_files = collect_futures_data(start_time, end_time)
    all_data = {}
    for fn in data_files:
        sym = os.path.basename(fn).split('_')[0].upper()
        df = prepare_data(fn)
        
        start_datetime = datetime.strptime(start_time, "%d %b %Y %H:%M:%S")
        all_data[sym] = df.loc[start_datetime:]
    
    results = []
    day_range = pd.date_range(start_time, end_time, freq='D')
    
    for i in tqdm(range(7, len(day_range)), desc="Simulating trading days"):
        opt_start = day_range[i-7]
        opt_end = day_range[i-1] + timedelta(hours=23, minutes=59, seconds=59)
        trade_day_start = day_range[i]
        trade_day_end = trade_day_start + timedelta(hours=23, minutes=59, seconds=59)
        
        day_results = {
            'date': trade_day_start.strftime("%Y-%m-%d"),
            'symbols': [],
            'avg_return': 0
        }
        
        df_opt_start_str = opt_start.strftime("%Y-%m-%d %H:%M:%S")
        df_opt_end_str = opt_end.strftime("%Y-%m-%d %H:%M:%S")
        
        qualifying_symbols = []
        
        for sym, df_full in all_data.items():
            df_opt = df_full.loc[df_opt_start_str:df_opt_end_str]
            
            if len(df_opt) < 10:
                continue
            
            df_daily = df_opt.resample('D').agg({'Close':'last','High':'max','Low':'min'})
            daily_change = df_daily['Close'].pct_change().abs()
            daily_amplitude = (df_daily['High'] - df_daily['Low']) / df_daily['Low']
            
            if daily_change.max() > 0.955 or daily_amplitude.max() > 0.955:
                continue 
            
            best_ind, ga_score = symbol_optimize_multiple_runs(df_opt, runs=3)
            
            if not best_ind:
                continue
            
            if ga_score > 5:
                df_trade_start_str = trade_day_start.strftime("%Y-%m-%d %H:%M:%S")
                df_trade_end_str = trade_day_end.strftime("%Y-%m-%d %H:%M:%S")
                trade_df = df_full.loc[df_trade_start_str:df_trade_end_str]
                
                if len(trade_df) < 2:
                    continue
                
                trade_stats = backtest_sl_tp(trade_df, best_ind[0], best_ind[1])
                
                qualifying_symbols.append({
                    'symbol': sym,
                    'ga_score': ga_score,
                    'sl': best_ind[0],
                    'tp': best_ind[1],
                    'stats': trade_stats
                })
        
        if not qualifying_symbols:
            results.append({
                'date': trade_day_start.strftime("%Y-%m-%d"),
                'symbols': [],
                'avg_return': 0
            })
            continue
        
        returns = []
        for symbol_data in qualifying_symbols:
            ret = symbol_data['stats'].get('Return [%]', 0)
            returns.append(ret)
            day_results['symbols'].append(symbol_data)
        
        day_results['avg_return'] = np.mean(returns)
        results.append(day_results)

    print("\n=== FINAL DAILY RESULTS ===")
    print("{:<12} | {:<20} | {:>10} | {:>10} | {:>15}".format(
        "Date", "Symbols", "GA Scores", "Return [%]", "Avg Return [%]"
    ))
    
    for day_result in results:
        if not day_result['symbols']:
            print("{:<12} | {:<20} | {:>10} | {:>10}".format(
                day_result['date'], 'NO SYMBOL SELECTED', 'N/A', 'N/A'
            ))
        else:
            first_symbol = True
            for symbol_data in day_result['symbols']:
                print("{:<12} | {:<20} | {:>10.2f} | {:>10.2f} | {:>15.2f}".format(
                    day_result['date'] if first_symbol else '',
                    symbol_data['symbol'],
                    safe_float(symbol_data['ga_score']),
                    safe_float(symbol_data['stats'].get('Return [%]', 0)),
                    safe_float(day_result['avg_return']) if first_symbol else 0.0
                ))
                first_symbol = False
    
    print("\n=== DETAILED STATISTICS ===")
    for day_result in results:
        if day_result['symbols']:
            print(f"\nDate: {day_result['date']}")
            print(f"Average Return: {safe_float(day_result['avg_return']):.2f}%")
            for symbol_data in day_result['symbols']:
                print(f"\nSymbol: {symbol_data['symbol']}")
                print(f"GA Score: {safe_float(symbol_data['ga_score']):.2f}")
                print(f"SL: {symbol_data['sl']}")
                print(f"TP: {symbol_data['tp']}")
                for key, value in symbol_data['stats'].items():
                    print(f"{key}: {value}")

    return results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("\nSimulation completed.")