import os,sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
import pytz
from post_filtered import run_post_process
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
# add edt
edt = pytz.timezone("America/New_York")

# Define the cosine fitting function
def cosine_fit(x, amplitude, frequency, phase):
    return amplitude * np.cos(frequency * x + phase) + 0 #offset

# Plot MACD histogram data in ax_floating with cosine fitting and zero-crossing points
def plot_macd_histogram_with_cosine_fit(ax, macd_hist_data):
    x_data = np.arange(len(macd_hist_data))
    y_data = macd_hist_data

    if (len(macd_hist_data)) == 3:
        return 1

    # Fit a cosine function to the MACD histogram data
    try:
        # Initial parameters: amplitude, frequency, phase, offset
        initial_guess = [np.max(y_data) - np.min(y_data), 2 * np.pi / len(y_data), 0]
        params, _ = curve_fit(cosine_fit, x_data, y_data, p0=initial_guess)

        # Generate fitted y values using the cosine function
        extended_x_data = np.arange(0, len(x_data) + int(np.pi / params[1]))  # Extend by π in frequency
        fitted_y_data = cosine_fit(extended_x_data, *params)
        
        # Plot the MACD histogram and the fitted cosine curve
        ax.plot(x_data, y_data, 'o-', label='MACD Histogram', markersize=4, color='purple', alpha=0.7)
        ax.plot(extended_x_data, fitted_y_data, 'b--', label='Cosine Fit')

        # Find zero-crossings and mark them
        zero_crossings = np.where(np.diff(np.sign(y_data)))[0]
        for crossing in zero_crossings:
            ax.plot(crossing, y_data[crossing], 'ro', label='Zero Crossing' if crossing == zero_crossings[0] else "")
        
        fitted_y_original_range = cosine_fit(x_data, *params)
        mse = np.mean((y_data - fitted_y_original_range) ** 2)
        
        # Annotate with fitting parameters if needed
        amplitude, frequency, phase = params
        ax.text(0.05, 0.95, f'Amp: {amplitude:.2f}, Freq: {frequency:.2f} Phase: {phase:.2f}\nMSE:{mse:.4f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top')
        
    except RuntimeError:
        print("Cosine fit failed; parameters may be unsuitable for fitting.")


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

def calculate_kdj(data, period=9):
    """Calculate kdj_k, kdj_d, and kdj_j for the KDJ indicator."""
    # Calculate the lowest low and highest high over the specified period
    low_min = data['Low'].rolling(window=period, min_periods=1).min()
    high_max = data['High'].rolling(window=period, min_periods=1).max()

    # Calculate K% (fast stochastic) and rename it to kdj_k
    data['kdj_k'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))

    # Smooth kdj_k to get kdj_d by applying a moving average
    data['kdj_d'] = data['kdj_k'].rolling(window=3, min_periods=1).mean()

    # Calculate kdj_j as 3 * kdj_k - 2 * kdj_d
    data['kdj_j'] = 3 * data['kdj_k'] - 2 * data['kdj_d']

    return data[['kdj_k', 'kdj_d', 'kdj_j']]

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
    # Ensure the DataFrame contains volume data
    # Calculate KDJ and merge with the main data
    kdj_data = calculate_kdj(data)
    data = pd.concat([data, kdj_data], axis=1)
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

def find_buy_sell_points7(x_valid,y_valid,hist_valid):
    # Initialize lists to hold the buy and sell points
    buy_points = []
    sell_points = []

    y_valid=100-y_valid
    # Loop through y_valid to find buy and sell points
    i = 1
    sellbuy = 0
    while i < len(y_valid) - 2:
        # Check for buy point
        if sellbuy ==0 :
            if (y_valid[i + 2] < 50 and
                y_valid[i+1]>y_valid[i+2] and
                (y_valid[i]>50 or y_valid[i-1]>50) and
                hist_valid[i+2]>hist_valid[i+1] and
                (hist_valid[i]<0 or hist_valid[i]<hist_valid[i+1])):
                buy_points.append(x_valid[i + 2])   # The second point is the buy point
                #i += 2  # Move to the point after the buy point
                sellbuy = 1
                continue
        
        # Check for sell point
        if sellbuy == 1:
            #print(y_valid[i+2],y_valid[i+1],y_valid[i])
            if (y_valid[i+2] > 50 and
                y_valid[i+1]<y_valid[i+2] and
                (y_valid[i]<50 or y_valid[i-1]<50) and
                hist_valid[i+2]<hist_valid[i+1]):
                sell_points.append(x_valid[i + 2])  # The second point is the sell point
                #i += 2  # Move to the point after the sell point
                sellbuy = 0
                continue
        
        i += 1  # Move to the next point
    return buy_points,sell_points

def analyze_and_plot_stocks(today, future_days=0):
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
                market_cap = float(cap_str) / 1e9  # Convert to billions
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
    sel_idx=0
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
        if filtered==0:continue

        # Fit line on MACD histogram
        p, days_to_positive = fit_line_and_predict(data['MACD_hist'].values[-3:])

        add_technical_indicators(data)
        meantrend = (data['RSI_6']+100-data['WR_6']+data['kdj_k'])/3
        x_range = np.arange(len(data))
        # Mask NaN values in 'WR_6' to get valid data points
        meanWR = (data['WR_6']+data['WR_10'])/2
        valid_mask = ~np.isnan(meanWR)
        x_valid = x_range[valid_mask]           # Filtered x-values without NaNs
        y_valid = meanWR[valid_mask]      # Filtered WR_6 values without NaNs
        hist_valid = data['MACD_hist'][valid_mask]
        buy_points,sell_points = find_buy_sell_points(x_valid,y_valid,hist_valid)
        buy_points7,sell_points7 = find_buy_sell_points7(x_valid,meantrend[valid_mask],hist_valid)
        nearest_buy = x_valid[-1]-buy_points[-1]
        nearest_buy7 = x_valid[-1]-buy_points7[-1]

        if nearest_buy>1 and nearest_buy7>1:
            continue
        min_buy = min(nearest_buy,nearest_buy7)
        sel_idx+=1
        print(f'|{sel_idx:>4}/{total_stocks}|{stockticker:<5}|f:{tot_filtered:<2}|{stock_capitalizations[idx-1][1]:<3.1f}B|BP:{min_buy}')
        filtered_file.write(f"{stockticker},{market_cap}\n")
        continue
        # Plotting logic
        fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(5, 1, figsize=(12, 8), 
                                    gridspec_kw={'height_ratios': [2, 1,1,1,1], 'hspace': 0}, 
                                    sharex=True)

        # Plot candlestick data
        plot_candlestick(ax1, data)

        # Plot EMAs
        for window in range(3, 26, 2):
            ax1.plot(range(len(data)), data[f'EMA_{window}'], 
                    label=f'EMA_{window}', 
                    alpha=0.5 if window != 3 else 1,
                    linewidth=2 if window == 3 else 1, color='green')
        for window in range(27, 52, 2):
            ax1.plot(range(len(data)), data[f'EMA_{window}'], 
                    label=f'EMA_{window}', 
                    alpha=0.5 if window != 3 else 1,
                    linewidth=2 if window == 3 else 1, color='red')
        # Plot fitted EMA line
        ax1.plot(range(last_crossover_idx, len(data_for_check)), fit_values, 'b--', label=f'EMA Fit Slope: {slope:.3f}, MSE: {mse:.3f}')

        # Distinguish future days in the plot
        #ax1.axvline(x=len(data) - 0.5-future_days, linestyle='--', color='blue', label='Today')


        # MACD and Fitted Line Plot
        x_range = range(len(data))
        ax2.plot(x_range, data['MACD'], label='MACD', color='blue')
        ax2.plot(x_range, data['MACD_signal'], label='Signal', color='orange')

        # Plot MACD Histogram with Color Conditions
        color_condition = np.where(data['Close'].diff() > 0, 'green', 'red')
        ax2.bar(x_range, data['MACD_hist'], color=color_condition)
        ax2.axhline(0, color='black', linewidth=1, linestyle='-')
        #ax2.axvline(x=len(data) - 0.5- future_days, linestyle='--', color='blue', label='Today')

        # Extend x-axis with future dates and set custom format
        ax2.set_xlim([0, len(extended_dates) - 1])
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_date(x, p, trading_dates)))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='right')

        # Show tick markers every `n_ticks` intervals
        n_ticks = 10
        tick_locations = np.linspace(0, len(data)-1 -future_days, n_ticks, dtype=int)
        ax2.set_xticks(tick_locations)
        ax2.set_xticklabels([extended_dates[i].strftime('%Y-%m-%d') for i in tick_locations], rotation=0, ha='right')


        # New subplot for RSI, WR, and Volume
        #ax3 = fig.add_subplot(3, 1, 3)  # Create a new subplot below ax2

        # Plot mean
        ax3.plot(x_range, 100-meantrend, label='RSI 6', color='blue', linestyle='-',linewidth=0.8,marker='.',markersize=3)
        #ax3.plot(x_range, 100-data['WR_6'], label='WR 6', color='purple', linestyle='-', linewidth=0.8,marker='.',markersize=3)
        #ax3.plot(x_range, data['kdj_k'], label='kdj k', color='red', linestyle='-', linewidth=0.8,marker='.',markersize=3)
        ax3_right = ax3.twinx() 
        ax3_right.plot(x_range, data['Close'], label='RSI 6', color='orange', linestyle='-',linewidth=0.8,marker='.',markersize=3)

        #ax3.plot(x_range, data['RSI_12'], label='RSI 12', color='orange', linestyle='-',linewidth=0.8,marker='.',markersize=3)
        #ax3.plot(x_range, data['RSI_24'], label='RSI 24', color='green', linestyle='-', linewidth=0.8,marker='.',markersize=3)

        # Plot Williams %R
        ax4.plot(x_range, data['WR_6'], label='WR 6', color='purple', linestyle='-', linewidth=0.8,marker='.',markersize=3)
        ax4.plot(x_range, data['WR_10'], label='WR 10', color='red', linestyle='-', linewidth=0.8,marker='.',markersize=3)
        ax4.axhline(80, color='red', linestyle='--', linewidth=0.8)  # Overbought level for RSI
        ax4.axhline(20, color='green', linestyle='--', linewidth=0.8)  # Oversold level for RSI


        # Plot Volume
        #ax5 = ax3.twinx()  # Create a twin y-axis for volume
        ax5.bar(x_range, data['Volume'], alpha=1, color=color_condition, label='Volume', width=0.75)

        # Formatting ax3
        ax3.axhline(30, color='red', linestyle='--', linewidth=0.8)  # Overbought level for RSI
        ax3.axhline(70, color='red', linestyle='--', linewidth=0.8)  # Overbought level for RSI
        ax3.axhline(50, color='green', linestyle='--', linewidth=0.8)  # Oversold level for RSI
        ax3.set_ylabel('RSI', color='black')
        ax4.set_ylabel('WR', color='black')
        ax5.set_ylabel('Volume', color='black')

 
        # Set minor grid for additional lines every two data points
        for ax in [ax1,ax2, ax3, ax4, ax5]:
            ax.grid(True, alpha=1)
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))  # Set minor grid every 2 data points
            ax.grid(True, which='minor', alpha=0.5)  # Enable minor grid with desired transparency

     
        for bp in buy_points:
            ax1.axvline(x=bp, color='green', linestyle='--', label='Buy Point' if 'Buy Point' not in plt.gca().get_legend_handles_labels()[1] else "")
            ax2.axvline(x=bp, color='green', linestyle='--', label='Buy Point' if 'Buy Point' not in plt.gca().get_legend_handles_labels()[1] else "")
            #ax3.axvline(x=bp, color='green', linestyle='--', label='Buy Point' if 'Buy Point' not in plt.gca().get_legend_handles_labels()[1] else "")
            ax4.axvline(x=bp, color='green', linestyle='--', label='Buy Point' if 'Buy Point' not in plt.gca().get_legend_handles_labels()[1] else "")
        for sp in sell_points:
            ax1.axvline(x=sp, color='red', linestyle='--', label='Sell Point' if 'Sell Point' not in plt.gca().get_legend_handles_labels()[1] else "")
            ax2.axvline(x=sp, color='red', linestyle='--', label='Sell Point' if 'Sell Point' not in plt.gca().get_legend_handles_labels()[1] else "")
            #ax3.axvline(x=sp, color='red', linestyle='--', label='Sell Point' if 'Sell Point' not in plt.gca().get_legend_handles_labels()[1] else "")
            ax4.axvline(x=sp, color='red', linestyle='--', label='Sell Point' if 'Sell Point' not in plt.gca().get_legend_handles_labels()[1] else "")
        for bp in buy_points7:
            ax3.axvline(x=bp, color='green', linestyle='-', label='Buy Point' if 'Buy Point' not in plt.gca().get_legend_handles_labels()[1] else "")
        for sp in sell_points7:
            ax3.axvline(x=sp, color='red', linestyle='-', label='Sell Point' if 'Sell Point' not in plt.gca().get_legend_handles_labels()[1] else "")


        # Save plot
        rank_str = f"{sel_idx:03}"
        ax1.set_title(f'{rank_str}|{stockticker}|{stock_capitalizations[idx-1][1]:.1f}B|MACD Slope: {MACD_hist_slope:.3f}|Crossover Days: {crossover_days}|EMA Slope: {slope:.3f}, Error: {mse:.3f} - Inc: {percentage_change:.1f}  - {decrease_percentage:.1f}- {note}')
        plt.tight_layout()
        plt.savefig(f'./static/images/{today}/{rank_str}_{stockticker}.png')
        plt.close()
    filtered_file.close()

def run_function_twices(deploy_mode):
    if deploy_mode==1:
        today = datetime.now(edt).strftime('%Y%m%d')
        print('today',today)
        analyze_and_plot_stocks(today, future_days=0)
    else:
        today = '20241108'
    analyze_and_plot_stocks(today, future_days=0)
    if deploy_mode == 1:
        run_post_process(1)

if len(sys.argv) > 1:
    run_function_twices(1)
else:
    run_function_twices(0)
