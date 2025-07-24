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

df_563 = df[df['unique_id'] == '#563']
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
# from autogluon.timeseries import TimeSeriesDataFrame

# ts_data = TimeSeriesDataFrame.from_data_frame(
#     df,
#     id_column='item_id',
#     timestamp_column='timestamp'
# )

# split_ratio = 0.8
# def get_split_index(ts):
#     n = len(ts)
#     return ts.index[round(n * split_ratio) - 1]

# split_idx = ts_data.groupby('item_id').apply(get_split_index)

# # Create train & holdout
# train = TimeSeriesDataFrame(ts_data.loc[ts_data.index.get_level_values('timestamp') <= split_idx[ts_data.index.get_level_values('item_id')]])
# holdout = TimeSeriesDataFrame(ts_data.loc[ts_data.index.get_level_values('timestamp') > split_idx[ts_data.index.get_level_values('item_id')]])

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
def sliding_windows(s, context_window=144, horizon=12):
    """
    Yields (history, future) pairs from a sequence s (e.g., list, numpy array, pandas Series).
    
    - context_window: number of past points to use (history)
    - horizon: number of future points to predict
    - skips partial windows at the end
    """
    total = context_window + horizon
    n = len(s)
    last_start = n - total
    for i in range(last_start + 1):
        window = s[i : i + total]
        yield window[:context_window], window[context_window:]


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

def rolling_history_future(df, ctx=144, h=12):
    """
    From a pandas TimeSeriesDataFrame (single item),
    yields sliding (history, future, item_id, start_index).
    """
    arr = df['target'].values
    n = len(arr)
    total = ctx + h
    for start in range(0, n - total + 1):
        hist = arr[start : start + ctx]
        fut = arr[start + ctx : start + total]
        yield hist, fut, df.index[0][0], start

windows = []
for item in test_tsdf.item_ids:
    subdf = test_tsdf.loc[item].reset_index()  # row-indexed DataFrame
    for hist, fut, uid, idx in rolling_history_future(subdf, ctx=144, h=12):
        windows.append({
            'item_id': uid,
            'start_idx': idx,
            'history': hist,
            'future': fut
        })

print(f"Total windows: {len(windows)}")

# %%
