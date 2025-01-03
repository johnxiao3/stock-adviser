import os,sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import json

def save_investment_data(investment_data, filename="investment_data.json"):
    try:
        # Write the investment data to a JSON file
        with open(filename, 'w') as file:
            json.dump(investment_data, file, indent=4)
        print(f"Investment data saved to {filename}")
    except Exception as e:
        print(f"Error saving investment data: {e}")





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

# Function to create custom date formatter for x-axis
def format_date(x, p, trading_dates):
    if x >= 0 and x < len(trading_dates):
        return trading_dates[int(x)].strftime('%Y-%m-%d')
    return ''

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
                i += 2  # Move to the point after the buy point
                sellbuy = 1
                continue
        
        # Check for sell point
        if sellbuy == 1:
            if (y_valid[i+2] > 50 and
                y_valid[i+1]<y_valid[i+2] and
                y_valid[i]<y_valid[i+2] and
                hist_valid[i+2]<hist_valid[i+1]):
                sell_points.append(x_valid[i + 2])  # The second point is the sell point
                i += 2  # Move to the point after the sell point
                sellbuy = 0
                continue
        
        i += 1  # Move to the next point
    return buy_points,sell_points

def find_selected_stocks(today, future_days=0):
    # Define the number of future days to plot after today
    #future_days = 0  # Adjust as needed
    realtoday = datetime.today()
    #today = '20241101'
    filtered_file_path = f'./static/images/{today}/{today}_selected.txt'
    filtered_file_path1 = f'./static/images/{today}/{today}_selected1.txt'

    today_date = datetime.strptime(today, '%Y%m%d')

    time_delta = (realtoday-today_date).days
    print(time_delta)
    os.makedirs(f"./static/images/{today}", exist_ok=True)


    filtered = 0
    # Load the NASDAQ screener CSV file # Check if the filtered file exists
    if os.path.exists(filtered_file_path):
        filtered = 1
        stock_capitalizations = []
        with open(filtered_file_path, 'r') as f:
            for line in f:
                stock_id, cap_str = line.strip().split(',')
                try:
                    market_cap = float(cap_str) / 1e9  # Convert to billions
                except:
                    market_cap = 0
                stock_capitalizations.append((stock_id, market_cap))
        # Step 2: Sort by market capitalization in descending order
        stock_capitalizations.sort(key=lambda x: x[1], reverse=True)
        tickers = [a[0] for a in stock_capitalizations]
        print(f"Loaded {len(tickers)} tickers from {filtered_file_path}")
    else:
        # Load the NASDAQ screener CSV file if the filtered file doesn't exist
        screener = pd.read_csv("./nasdaq_screener.csv")
        tickers = screener['Symbol']
        print(f"Loaded {len(tickers)} tickers from CSV file")

    total_stocks = len(tickers)
    tot_filtered = 0
    # Fetch, process, and plot for each ticker
    #with open(filtered_file_path, 'w') as filtered_file:
    if filtered == 0:
        filtered_file = open(filtered_file_path, 'w')
    else:
        filtered_file = open(filtered_file_path1, 'w')
    selected_stock = {
        "stock_prices": [],
        "stock_tickers": []
    }
    for idx, stockticker in enumerate(tickers, start=1):
        #if idx<415:
        #    continue
        note = ''
        stock = yf.Ticker(stockticker)
        data = stock.history(period="6mo")

        # Get the info dictionary, which sometimes contains the 'country' key
        market_cap = 0
        try:
            info = stock.info
            market_cap = info.get('marketCap')
            country = info.get("country", "Country information not available")
            if country=='China':continue
        except:
            pass


        try:
            data.index = data.index.tz_localize(None)
        except:
            continue
        # Filter data to only include up to `today`
        if future_days ==0 :
            data_for_check = data.copy()
        else:
            data_for_check = data[data.index < today_date].copy()
        try:
            today_close_price = data_for_check['Close'].iloc[-1]
        except:
            continue

        # Store trading dates for x-axis formatting
        trading_dates = data.index.tolist()

        # Append future dates for plotting
        future_dates = pd.date_range(start=trading_dates[-1], periods=2 + 1)[1:]
        extended_dates = trading_dates + list(future_dates)

        # Calculate EMAs and MACD
        for window in range(3, 26, 2):
            data_for_check[f'EMA_{window}'] = ema(data_for_check['Close'], window)
        for window in range(27, 52, 2):
            data_for_check[f'EMA_{window}'] = ema(data_for_check['Close'], window)
        data_for_check['MACD'], data_for_check['MACD_signal'], data_for_check['MACD_hist'] = calculate_macd(data_for_check['Close'])

        # Calculate EMAs and MACD
        for window in range(3, 26, 2):
            data[f'EMA_{window}'] = ema(data['Close'], window)
        for window in range(27, 52, 2):
            data[f'EMA_{window}'] = ema(data['Close'], window)
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = calculate_macd(data['Close'])
        
        future_close_price = data['Close'].iloc[-1]
        percentage_change = ((future_close_price - today_close_price) / today_close_price) * 100
        # Calculate the daily percentage change
        data['Pct_Change'] = data['Close'].pct_change() * 100  # Convert to percentage

        # Count the days with a decrease in the close price
        decrease_days = (data['Pct_Change'] <= 0).sum()

        # Calculate the total number of trading days
        total_days = data['Pct_Change'].count()  # Ignore NaN from pct_change()

        # Calculate the percentage of decrease days
        decrease_percentage = (decrease_days / total_days) * 100
        if decrease_percentage > 60: continue

        # Proceed with your conditions and analysis logic here as before
        # Calculate crossover days
        last_crossover_idx, crossover_sign = calculate_crossover_days(data_for_check['MACD_hist'].values)
        if last_crossover_idx is not None:
            crossover_days = len(data_for_check) - last_crossover_idx
        else:
            crossover_days = 'N/A'

        try:
            MACD_hist_slope = (data_for_check['MACD_hist'].values[-1] - data_for_check['MACD_hist'].values[-3])/2
        except:
            continue
        # Skip if not meeting criteria
        if (crossover_sign != '-' or crossover_days == 'N/A' or
            data_for_check['MACD_hist'].values[-3:][0] > 0 or 
            data_for_check['MACD_hist'].values[-3:][-1] > 0 or
            not all(np.diff(data_for_check['MACD_hist'].values[-3:]) > 0)):
            continue

        # Check EMA conditions
        current_day_idx = -1
        green_ema = data_for_check['EMA_3'].values[current_day_idx]
        all_other_ema_values = [data_for_check[f'EMA_{window}'].values[current_day_idx] for window in range(5, 51, 2)]

        if all(green_ema < ema_value for ema_value in all_other_ema_values):
            note = 'GreenLow'
            #continue

        ema_3_last_3 = data_for_check['EMA_3'].values[-3:]
        if ema_3_last_3[-1] < ema_3_last_3[-2]:
            continue
        
        #if MACD_hist_slope <0.02:continue
        if MACD_hist_slope <0.15:continue


        if crossover_days>22:continue
        # Fit EMA 3 line from crossover point to today
        slope, mse, fit_values = fit_ema_line(data_for_check, last_crossover_idx, len(data_for_check) - 1)
        if mse<0.02 and slope <-0.04: continue

        tot_filtered += 1
        filtered_file.write(f"{stockticker},{market_cap}\n")
        if filtered==0:continue

        # Fit line on MACD histogram
        p, days_to_positive = fit_line_and_predict(data['MACD_hist'].values[-3:])

        add_technical_indicators(data)

        x_range = np.arange(len(data))
        # Mask NaN values in 'WR_6' to get valid data points
        meanWR = (data['WR_6']+data['WR_10'])/2
        valid_mask = ~np.isnan(meanWR)
        x_valid = x_range[valid_mask]           # Filtered x-values without NaNs
        y_valid = meanWR[valid_mask]      # Filtered WR_6 values without NaNs
        hist_valid = data['MACD_hist'][valid_mask]
        buy_points,sell_points = find_buy_sell_points(x_valid,y_valid,hist_valid)
        nearest_buy = x_valid[-1]-buy_points[-1]
        cap = stock_capitalizations[idx-1][1]
        print(f'|{idx:>4}/{total_stocks}|{stockticker:<5}|${today_close_price:<5.1f}|{stock_capitalizations[idx-1][1]:<5.1f}B|BP:{nearest_buy.item():<2}|')
        #if cap < 10:continue
        print(nearest_buy.item())
        if nearest_buy.item() == 0:
            selected_stock['stock_prices'].append(today_close_price.item())
            selected_stock['stock_tickers'].append(stockticker)
            print(f'|{idx:>4}/{total_stocks}|{stockticker:<5}|${today_close_price:<5.0f}|{stock_capitalizations[idx-1][1]:<5.1f}B|BP:{nearest_buy.item():<2}|')

    filtered_file.close()
    return selected_stock

def generate_investment_plan(selected_stock, budget):
    """
    Generates an investment plan based on available stock prices and a budget.
    
    Parameters:
        selected_stock (dict): Dictionary with 'stock_prices' and 'stock_tickers' lists.
        budget (float): The amount of money available for investment.
    
    Returns:
        list: A list of dictionaries, each containing 'ticker', 'price', 'quantity', and 'total_cost'.
        float: Remaining budget after purchases.
    """
    investment_plan = []  # List to store the stocks bought and quantity

    # Loop through stocks in order
    for price, ticker in zip(selected_stock['stock_prices'], selected_stock['stock_tickers']):
        if budget >= price:  # Check if we have enough budget to buy at least one share
            quantity = int(budget // price)  # Max shares that can be bought for this stock
            cost = quantity * price          # Total cost for these shares
            budget -= cost                   # Deduct the cost from the budget

            # Add to investment plan
            investment_plan.append({
                'ticker': ticker,
                'price': price,
                'quantity': quantity,
                'total_cost': cost
            })

    return investment_plan, budget

def get_buy_stocks(deploy_mode: int, datestr: str = 'aaa', budget: float = 1000.0):
    """
    Generate stock investment plans based on different deployment modes.
    
    Args:
        deploy_mode (int): 
            0: Local run mode
            1: Server auto-run mode
            2: Server manual mode
        datestr (str): Date string in format 'YYYYMMDD'
        budget (float): Investment budget amount
    """
    # Determine date and budget based on mode
    if deploy_mode == 2:
        # Server manual mode
        today = datestr
        current_budget = float(budget)
    elif deploy_mode == 1:
        # Server auto-run mode
        today = datetime.today().strftime('%Y%m%d')
        current_budget = 983.0
    else:
        # Local run mode
        today = '20241210'
        current_budget = 978.0

    try:
        # Get selected stocks for the specified date
        selected_stocks = find_selected_stocks(today, future_days=0)
        print(selected_stocks)
        # Generate investment plan
        investment_plan, remaining_budget = generate_investment_plan(
            selected_stocks, 
            current_budget
        )

        # Display investment plan
        print("\nInvestment Plan:")
        for stock in investment_plan:
            print(f"Buy {stock['quantity']} shares of {stock['ticker']} "
                  f"at ${stock['price']:.2f} each")
            print(f"Total Cost: ${stock['total_cost']:.2f}")
        print(f"Remaining Budget: ${remaining_budget:.2f}")

        # Save investment data
        save_investment_data(investment_plan, './static/investment_data.json')
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

def main():
    """
    Main entry point for the script.
    Handles command line arguments and calls get_buy_stocks with appropriate parameters.
    """
    arg_count = len(sys.argv)
    print('Number of arguments:', arg_count)
    
    try:
        if arg_count == 4:
            # Manual mode with date and budget specified
            get_buy_stocks(2, datestr=sys.argv[-2], budget=sys.argv[-1])
            # useage: python get_sellbuy_point 20241210 999
        elif arg_count == 2:
            # Auto-run mode
            get_buy_stocks(1)
        else:
            # Local run mode
            get_buy_stocks(0)
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()