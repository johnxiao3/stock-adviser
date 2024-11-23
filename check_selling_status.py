import os,sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
import pytz
import time
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import json

def delete_stock(ticker, filename="./static/investment_data.json"):
    # Load the current JSON data from the file
    try:
        with open(filename, 'r') as file:
            stock_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: {filename} contains invalid JSON.")
        return

    # Find and delete the stock with the given ticker
    updated_data = [stock for stock in stock_data if stock["ticker"] != ticker]
    
    if len(updated_data) == len(stock_data):
        print(f"No stock with ticker {ticker} found.")
        return
    
    # Write the updated data back to the JSON file
    with open(filename, 'w') as file:
        json.dump(updated_data, file, indent=4)

    print(f"Stock with ticker {ticker} deleted successfully.")



# Function to log the information to a file
def log_to_file(log_message, log_file='./static/log.txt'):
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Get the current date and time in EDT
    edt = pytz.timezone('US/Eastern')
    current_time = datetime.now(edt).strftime("%Y-%m-%d %H:%M:%S %Z")

    # Create the final message with date and log
    log_entry = f"{current_time} - {log_message}\n"

    # Open the log file and append the log message
    with open(log_file, 'a') as f:
        f.write(log_entry)

def get_tickers_from_json(filename="./static/investment_data.json"):
    try:
        # Read the investment data from the JSON file
        with open(filename, 'r') as file:
            investment_data = json.load(file)
        
        # Check if the data is a list of stocks
        if isinstance(investment_data, list):
            # Extract tickers from the list of stock dictionaries
            tickers = [stock["ticker"] for stock in investment_data]
            return tickers
        else:
            print(f"Error: Expected a list of stocks, but got {type(investment_data)}.")
            return []
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: {filename} contains invalid JSON.")
        return []
    except KeyError:
        print(f"Error: Missing 'ticker' field in one of the stock entries.")
        return []

# Function to calculate EMA
def ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Function to calculate MACD
def calculate_macd(data, slow=26, fast=12, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = 2*(macd - macd_signal)
    return macd, macd_signal, macd_hist

# Function to calculate the nearest MACD histogram crossover
def calculate_crossover_days(macd_hist):
    for i in range(len(macd_hist)-1, 0, -1):
        if macd_hist[i] < 0 and macd_hist[i-1] >= 0:  # Positive to negative crossover
            return i, '-'
        elif macd_hist[i] > 0 and macd_hist[i-1] <= 0:
            return i,'+'
    return None, ''

# Function to fit a line and estimate days to positive
def fit_line_and_predict(macd_hist_values):
    x = np.array([1, 2, 3])
    y = macd_hist_values[-3:]
    p = np.polyfit(x, y, 1)
    days_to_positive = -p[1] / p[0]
    return p, days_to_positive

# Modified plot_candlestick function to use continuous trading day indices
def plot_candlestick(ax, data):
    for idx in range(len(data)):
        open_price = data['Open'].iloc[idx]
        close_price = data['Close'].iloc[idx]
        high_price = data['High'].iloc[idx]
        low_price = data['Low'].iloc[idx]
        
        # Use integer index instead of date
        ax.plot([idx, idx], [low_price, high_price], color='black',linewidth=1,zorder=1)
        ax.add_patch(plt.Rectangle((idx - 0.2, min(open_price, close_price)), 
                                 0.4, 
                                 abs(close_price - open_price),
                                 color='green' if close_price >= open_price else 'red', 
                                 alpha=1.0,zorder=2))

# Function to fit line for EMA and calculate slope and error
def fit_ema_line(data, start_idx, end_idx):
    y = data['EMA_3'].iloc[start_idx:end_idx + 1].values
    x = np.arange(len(y))
    p = np.polyfit(x, y, 1)
    slope, intercept = p
    fit_values = np.polyval(p, x)
    mse = np.mean((y - fit_values) ** 2)
    return slope, mse, fit_values


def calculate_rsi(data, window):
    """Calculate the Relative Strength Index (RSI) for a given window."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_wr(data, window):
    """Calculate the Williams %R for a given window."""
    highest_high = data['High'].rolling(window=window).max()
    lowest_low = data['Low'].rolling(window=window).min()
    wr = (highest_high - data['Close']) / (highest_high - lowest_low) * 100
    return wr

def add_technical_indicators(data):
    """Calculate and add RSI and WR to the DataFrame."""
    # Calculate RSI for 6, 12, and 24 periods
    data['RSI_6'] = calculate_rsi(data, 6)
    data['RSI_12'] = calculate_rsi(data, 12)
    data['RSI_24'] = calculate_rsi(data, 24)

    # Calculate Williams %R for 6 and 10 periods
    data['WR_6'] = calculate_wr(data, 6)
    data['WR_10'] = calculate_wr(data, 10)

    # Ensure you have the volume data in your DataFrame
    if 'Volume' not in data.columns:
        raise ValueError("The DataFrame must contain a 'Volume' column.")
    return data

def find_buy_sell_points(x_valid,y_valid,hist_valid):
    # Initialize lists to hold the buy and sell points
    buy_points = []
    sell_points = []
    # Loop through y_valid to find buy and sell points
    i = 1
    sellbuy = 0
    while i < len(y_valid) - 2:
        # Check for buy point
        if sellbuy ==0 :
            if (y_valid[i + 2] < 50 and
                y_valid[i+1]>y_valid[i+2] and
                y_valid[i]>y_valid[i+2] and
                (y_valid[i]>50 or y_valid[i-1]>50) and
                hist_valid[i+2]>hist_valid[i+1] and
                (hist_valid[i]<0 or hist_valid[i]<hist_valid[i+1])):
                buy_points.append(x_valid[i + 2])   # The second point is the buy point
                #i += 2  # Move to the point after the buy point
                sellbuy = 1
                continue
        
        # Check for sell point
        if sellbuy == 1:
            if (y_valid[i+2] > 50 and
                y_valid[i+1]<y_valid[i+2] and
                y_valid[i]<y_valid[i+2] and
                hist_valid[i+2]<hist_valid[i+1]):
                sell_points.append(x_valid[i + 2])  # The second point is the sell point
                #i += 2  # Move to the point after the sell point
                sellbuy = 0
                continue
        
        i += 1  # Move to the next point
    return buy_points,sell_points

def find_sell_stocks(today, future_days=0):
    realtoday = datetime.today()
    today_date = datetime.strptime(today, '%Y%m%d')
    time_delta = (realtoday-today_date).days
    # Example usage
    tickers = get_tickers_from_json()
    print("Stock Tickers:", tickers)

    total_stocks = len(tickers)
    
    selected_stock = {
        "stock_prices": [],
        "stock_tickers": []
    }
    for idx, stockticker in enumerate(tickers, start=1):
        stock = yf.Ticker(stockticker)
        data = stock.history(period="6mo")

        # Get the info dictionary, which sometimes contains the 'country' key
        market_cap = 0
        try:
            info = stock.info
            market_cap = info.get('marketCap')
        except:
            pass


        try:
            data.index = data.index.tz_localize(None)
        except:
            continue

        today_close_price = data['Close'].iloc[-1]
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = calculate_macd(data['Close'])
        add_technical_indicators(data)

        x_range = np.arange(len(data))
        # Mask NaN values in 'WR_6' to get valid data points
        meanWR = (data['WR_6']+data['WR_10'])/2
        valid_mask = ~np.isnan(meanWR)
        x_valid = x_range[valid_mask]     # Filtered x-values without NaNs
        y_valid = meanWR[valid_mask]      # Filtered WR_6 values without NaNs
        hist_valid = data['MACD_hist'][valid_mask]
        buy_points,sell_points = find_buy_sell_points(x_valid,y_valid,hist_valid)
        nearest_buy = x_valid[-1]-buy_points[-1]
        nearest_sell = x_valid[-1]-sell_points[-1]
        if nearest_sell.item() == 0:
            selected_stock['stock_prices'].append(today_close_price.item())
            selected_stock['stock_tickers'].append(stockticker)
        disp_str = f'|{idx:>4}/{total_stocks}|{stockticker:<5}|${today_close_price:<5.0f}|{market_cap/1000000000:<5.1f}B|BP:{nearest_buy.item():<2}|SP:{nearest_sell.item():<2}|'
        print(disp_str)
        log_to_file(disp_str)

    return selected_stock

def check_holding_stocks(deploy_mode):
    time.sleep(15)
    if deploy_mode:
        today = datetime.today().strftime('%Y%m%d')
    else:
        today = '20241114'
    selected_stock = find_sell_stocks(today, future_days=0)
    # Example logic to print and log the message
    if len(selected_stock['stock_tickers']) == 0:
        log_to_file('No stocks for sell')
        print('No stocks for sell')
    else:
        log_message = 'Stocks below need to be selled:\n' + '\n'.join(selected_stock['stock_tickers'])
        log_to_file(log_message)
        print('Stocks below need to be selled')
        print(selected_stock['stock_tickers'])
        for stock_ticker in selected_stock['stock_tickers']:
            delete_stock(stock_ticker)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_holding_stocks(1)
    else:
        check_holding_stocks(0)
