# %%
# https://aihorizonforecast.substack.com/p/tabpfn-ts-a-surprising-new-breakthrough
import pandas as pd
import numpy as np

from autogluon.timeseries import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import to_gluonts_univariate, generate_test_X
from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode, FeatureTransformer, DefaultFeatures

import matplotlib.pyplot as plt

from utilsforecast.losses import mae, mse
from utilsforecast.evaluation import evaluate

pd.options.display.float_format = '{:.2f}'.format

import os
os.environ["NIXTLA_ID_AS_COL"] = "1"
# %%

# %%
df = pd.read_csv('/root/pfn/neuralforecast/datasets/ohiot1dm_exog_9_day_test.csv')
df.head()

# df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")
# df.head()

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="unique_id",
    timestamp_column="ds"
)
train_data.head()

# %%

df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
df['timestamp'] = df['ds'] # autogluon expects timestamp column
df['item_id'] = df['unique_id'].str.replace('#', '') # autogluon expects item_id column

df['unique_id'].value_counts()

df_563 = df[df['item_id'] == '563']
df_563.head()
plt.figure(figsize=(26, 5))
plt.plot(df_563['ds'], df_563['y'], linestyle='-', markersize=2, label=f'Item #563')

plt.title(f'Glucose levels for patient #563', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Glucose level', fontsize=12)
plt.xticks(rotation=45)
# plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()




# %%
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.splitter import MultiWindowSplitter

# Build ts with proper multi-index
ts = TimeSeriesDataFrame.from_data_frame(
    df, id_column='item_id', timestamp_column='timestamp'
).sort_index()

def tts_split(ts, split_ratio=0.8):
        
    # 80/20 per-item split
    split = 0.8
    train_parts, hold_parts = [], []

    for item in ts.item_ids:
        df_i = ts.loc[item]                           # DataFrame indexed by timestamp
        df_i = df_i.reset_index().assign(item_id=item)  # add item_id column back
        n = len(df_i)
        cut = int(n * split)
        train_parts.append(df_i.iloc[:cut])
        hold_parts.append(df_i.iloc[cut:])

    # Combine and rebuild TSDF
    train = TimeSeriesDataFrame.from_data_frame(
        pd.concat(train_parts, ignore_index=True),
        id_column='item_id',
        timestamp_column='timestamp'
    )

    holdout = TimeSeriesDataFrame.from_data_frame(
        pd.concat(hold_parts, ignore_index=True),
        id_column='item_id',
        timestamp_column='timestamp'
    )

    return train, holdout

train, holdout = tts_split(ts, split_ratio=0.8)

# Optional sanity-check
print(train.num_timesteps_per_item().describe(),
      holdout.num_timesteps_per_item().describe())


train.tail()

holdout.head()


# %%

# %%

# %%
from autogluon.timeseries import TimeSeriesDataFrame

# 1. Ensure timestamp is datetime
df_563['timestamp'] = pd.to_datetime(df_563['timestamp'])

# 2. Rename target
df_563 = df_563.rename(columns={'y': 'target'})

# 3. Build TSDF
tsdf_563 = TimeSeriesDataFrame.from_data_frame(
    df_563,
    id_column='item_id',
    timestamp_column='timestamp'
)

# 4. Inspect
print(tsdf_563.head())
print("New frequency:", tsdf_563.freq)
print("Number of items:", tsdf_563.num_items)

# %%
train_tsdf, test_tsdf = tts_split(tsdf_563, split_ratio=0.8)
# %%

def rolling_history_future(df, ctx=144, h=12, item_id=None):
    """
    Given a row-indexed pandas DataFrame for a single item (with columns including 'target'),
    yields (history, future, item_id, start_idx).
    - history: numpy array, shape=(ctx,)
    - future: numpy array, shape=(h,)
    - item_id can be provided as a parameter if not in DataFrame
    """
    arr = df['target'].values
    total = ctx + h
    
    # Get item_id from parameter if not in DataFrame
    if item_id is None and 'item_id' in df.columns:
        item = df['item_id'].iat[0]
    else:
        item = item_id
        
    n = len(arr)
    last_start = n - total
    for start in range(last_start + 1):
        window = arr[start : start + total]
        history = window[:ctx]
        future = window[ctx:]
        yield history, future, item, start


windows = []
for item in test_tsdf.item_ids:
    subdf = test_tsdf.loc[item].reset_index() 
    print(subdf.head()) # row-indexed DataFrame
    for hist, fut, uid, idx in rolling_history_future(subdf, ctx=144, h=12, item_id=item):
        windows.append({
            'item_id': uid,
            'start_idx': idx,
            'history': hist,
            'future': fut
        })

print(f"Total windows: {len(windows)}")







# %%
train_windows = []
for item in train_tsdf.item_ids:
    subdf = train_tsdf.loc[item].reset_index() 
    for hist, fut, uid, idx in rolling_history_future(subdf, ctx=144, h=12, item_id=item):
        train_windows.append({
            'item_id': uid,
            'start_idx': idx,
            'history': hist,
            'future': fut
        })

print(f"Total train windows: {len(train_windows)}")

# %%
# Visualization of train/test split and a random window example
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import timedelta

# Get the time series for visualization
item_id = '563'  # Use the same item we've been working with

# Step 1: Get the train and test data with timestamps
train_series = train_tsdf.loc[item_id].reset_index()
test_series = test_tsdf.loc[item_id].reset_index()

# Create the plot with proper date formatting
plt.figure(figsize=(20, 8))

# Step 2: Select a random window from test windows for this item
test_item_windows = [w for w in windows if w['item_id'] == item_id]
if test_item_windows:
    random_window = random.choice(test_item_windows)
    start_idx = random_window['start_idx']
    
    # Calculate the actual indices in the test set
    context_start = start_idx
    context_end = context_start + 144  # ctx length
    horizon_end = context_end + 12  # h length
    
    # Get the corresponding timestamps
    context_timestamps = test_series['timestamp'].iloc[context_start:context_end]
    horizon_timestamps = test_series['timestamp'].iloc[context_end:horizon_end]
    
    # Calculate the focus range - show only data from 10 days before the context window
    focus_start_date = context_timestamps.iloc[0] - timedelta(days=10)
    
    # Filter train data to show only the relevant part
    relevant_train = train_series[train_series['timestamp'] >= focus_start_date]
    
    # Plot the filtered train and test data
    plt.plot(relevant_train['timestamp'], relevant_train['target'], 'b-', label='Train')
    plt.plot(test_series['timestamp'], test_series['target'], 'r-', label='Test')
    
    # Add vertical line at train/test split
    split_point = train_series['timestamp'].iloc[-1]
    plt.axvline(x=split_point, color='g', linestyle='--', linewidth=2, label='Train/Test Split')
    
    # Plot the context and horizon with increased visibility
    plt.plot(context_timestamps, test_series['target'].iloc[context_start:context_end], 
             'g-', linewidth=3, label='Context Window (144 points)')
    plt.plot(horizon_timestamps, test_series['target'].iloc[context_end:horizon_end], 
             'm-', linewidth=3, label='Forecast Horizon (12 points)')
    
    # Highlight the window areas with increased opacity
    for i in range(len(context_timestamps)-1):
        plt.axvspan(context_timestamps.iloc[i], context_timestamps.iloc[i+1], 
                   alpha=0.15, color='green', edgecolor=None)
    
    for i in range(len(horizon_timestamps)-1):
        plt.axvspan(horizon_timestamps.iloc[i], horizon_timestamps.iloc[i+1], 
                   alpha=0.25, color='magenta', edgecolor=None)

    # Add annotation for context and horizon
    plt.annotate('Context Window', 
                xy=(context_timestamps.iloc[len(context_timestamps)//2], 
                    test_series['target'].iloc[context_start + len(context_timestamps)//2]),
                xytext=(0, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                fontsize=12, color='darkgreen')
    
    plt.annotate('Forecast Horizon', 
                xy=(horizon_timestamps.iloc[len(horizon_timestamps)//2], 
                    test_series['target'].iloc[context_end + len(horizon_timestamps)//2]),
                xytext=(0, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                fontsize=12, color='darkmagenta')

# Format the x-axis to show dates properly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.xticks(rotation=45)

plt.title(f'Glucose Time Series for Patient #{item_id}', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Glucose Level', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%

# %%
# Create a GIF showing the sliding window moving across the time series
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import timedelta
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import os

# Get the time series for visualization
item_id = '563'  # Use the same item we've been working with

# Get the train and test data with timestamps
train_series = train_tsdf.loc[item_id].reset_index()
test_series = test_tsdf.loc[item_id].reset_index()

# Get all windows for this item
test_item_windows = [w for w in windows if w['item_id'] == item_id]

# We'll create frames for a subset of windows to keep the GIF manageable
# Take every 5th window or similar to reduce the number of frames
step = 5  # Adjust based on how many windows you have
selected_windows = test_item_windows[::step]

# Set up the figure
fig, ax = plt.subplots(figsize=(20, 8))

# Function to create a frame for each window position
def create_frame(i):
    ax.clear()
    
    # Get the window data
    window = selected_windows[i]
    start_idx = window['start_idx']
    
    # Calculate indices
    context_start = start_idx
    context_end = context_start + 144  # ctx length
    horizon_end = context_end + 12  # h length
    
    # Get timestamps
    context_timestamps = test_series['timestamp'].iloc[context_start:context_end]
    horizon_timestamps = test_series['timestamp'].iloc[context_end:horizon_end]
    
    # Calculate focus range - show only data from 10 days before the context window
    focus_start_date = context_timestamps.iloc[0] - timedelta(days=10)
    
    # Filter train data to show only the relevant part
    relevant_train = train_series[train_series['timestamp'] >= focus_start_date]
    
    # Plot the filtered train and test data
    ax.plot(relevant_train['timestamp'], relevant_train['target'], 'b-', label='Train')
    ax.plot(test_series['timestamp'], test_series['target'], 'r-', label='Test')
    
    # Add vertical line at train/test split
    split_point = train_series['timestamp'].iloc[-1]
    ax.axvline(x=split_point, color='g', linestyle='--', linewidth=2, label='Train/Test Split')
    
    # Plot the context and horizon
    ax.plot(context_timestamps, test_series['target'].iloc[context_start:context_end], 
            'g-', linewidth=3, label='Context Window (144 points)')
    ax.plot(horizon_timestamps, test_series['target'].iloc[context_end:horizon_end], 
            'm-', linewidth=3, label='Forecast Horizon (12 points)')
    
    # Highlight the window areas
    for j in range(len(context_timestamps)-1):
        ax.axvspan(context_timestamps.iloc[j], context_timestamps.iloc[j+1], 
                  alpha=0.15, color='green', edgecolor=None)
    
    for j in range(len(horizon_timestamps)-1):
        ax.axvspan(horizon_timestamps.iloc[j], horizon_timestamps.iloc[j+1], 
                  alpha=0.25, color='magenta', edgecolor=None)
    
    # Format the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)
    
    # Add title and labels
    ax.set_title(f'Sliding Window Visualization - Frame {i+1}/{len(selected_windows)}', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Glucose Level', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add frame counter
    ax.text(0.02, 0.95, f"Window {i+1}/{len(selected_windows)}", 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    return ax

# Create the animation
frames = len(selected_windows)
ani = animation.FuncAnimation(fig, create_frame, frames=frames, interval=500)

# Save as GIF
output_path = os.path.join(os.path.dirname(os.getcwd()), 'sliding_window.gif')
ani.save(output_path, writer=PillowWriter(fps=2))

plt.close()
print(f"GIF saved to: {output_path}")

# %%
