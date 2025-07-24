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
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={'task_name': 'embedding'}, # We are loading the model in `embedding` mode to learn representations
    # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
)
# %%
model.init()
print(model)

# %%
# Compute MOMENT embeddings for train_windows
import torch
import numpy as np
from tqdm import tqdm

print(f"Computing MOMENT embeddings for {len(train_windows)} train windows...")
print("Each 'history' has 144 timestamps and will be divided into 18 patches (144/8) for embedding.")

# Convert to torch and process embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Process in batches for efficiency
batch_size = 32
for i in tqdm(range(0, len(train_windows), batch_size), desc="Computing embeddings"):
    batch_end = min(i + batch_size, len(train_windows))
    batch_histories = []
    
    # Collect histories for this batch
    for j in range(i, batch_end):
        history = train_windows[j]['history']
        # Ensure history is float32 and has shape (1, 144) for univariate time series
        history_tensor = torch.FloatTensor(history).unsqueeze(0)  # Shape: (1, 144)
        batch_histories.append(history_tensor)
    
    # Stack into batch tensor: (batch_size, 1, 144)
    batch_tensor = torch.stack(batch_histories).to(device)
    
    # Get embeddings from MOMENT
    with torch.no_grad():
        outputs = model(x_enc=batch_tensor)
        embeddings = outputs.embeddings  # Shape: (batch_size, embedding_dim)
    
    # Add embeddings back to train_windows
    for j, embedding in enumerate(embeddings):
        train_windows[i + j]['embedding'] = embedding.cpu().numpy()

print(f"✅ Successfully computed embeddings!")
print(f"Each embedding has dimension: {train_windows[0]['embedding'].shape}")
print(f"Sample train_window keys: {list(train_windows[0].keys())}")

# %%
# Let's inspect a few examples
print("\n🔍 Sample train_window structure:")
for i in range(min(3, len(train_windows))):
    window = train_windows[i]
    print(f"\nWindow {i}:")
    print(f"  - item_id: {window['item_id']}")
    print(f"  - start_idx: {window['start_idx']}")
    print(f"  - history shape: {window['history'].shape}")
    print(f"  - future shape: {window['future'].shape}")
    print(f"  - embedding shape: {window['embedding'].shape}")
    print(f"  - embedding mean: {window['embedding'].mean():.4f}")
    print(f"  - embedding std: {window['embedding'].std():.4f}")

# %%
# 🔍 Retrieval System using Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MOMENTRetriever:
    def __init__(self, train_windows):
        """
        Initialize the retrieval system with train windows containing embeddings.
        
        Args:
            train_windows: List of dictionaries with 'embedding' key
        """
        self.train_windows = train_windows
        self.train_embeddings = np.stack([w['embedding'] for w in train_windows])
        print(f"📚 Initialized retriever with {len(train_windows)} train windows")
        print(f"📊 Embedding matrix shape: {self.train_embeddings.shape}")
    
    def retrieve_similar(self, query_embedding, top_k=5, return_similarities=True):
        """
        Retrieve top-k most similar train windows for a query embedding.
        
        Args:
            query_embedding: numpy array of shape (embedding_dim,)
            top_k: number of most similar windows to retrieve
            return_similarities: whether to return similarity scores
            
        Returns:
            List of tuples: (window_dict, similarity_score) if return_similarities=True
            List of window_dict if return_similarities=False
        """
        # Ensure query_embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.train_embeddings)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order
        
        # Retrieve corresponding windows and similarities
        results = []
        for idx in top_k_indices:
            window = self.train_windows[idx]
            sim_score = similarities[idx]
            
            if return_similarities:
                results.append((window, sim_score))
            else:
                results.append(window)
                
        return results
    
    def batch_retrieve(self, query_embeddings, top_k=5):
        """
        Batch retrieval for multiple query embeddings.
        
        Args:
            query_embeddings: numpy array of shape (n_queries, embedding_dim)
            top_k: number of most similar windows to retrieve for each query
            
        Returns:
            List of lists: each inner list contains top-k results for one query
        """
        # Compute all similarities at once
        similarities = cosine_similarity(query_embeddings, self.train_embeddings)
        
        batch_results = []
        for i, query_similarities in enumerate(similarities):
            # Get top-k indices for this query
            top_k_indices = np.argsort(query_similarities)[-top_k:][::-1]
            
            # Retrieve corresponding windows and similarities
            query_results = []
            for idx in top_k_indices:
                window = self.train_windows[idx]
                sim_score = query_similarities[idx]
                query_results.append((window, sim_score))
                
            batch_results.append(query_results)
            
        return batch_results

# Initialize the retriever
retriever = MOMENTRetriever(train_windows)

# %%
# 🧪 Test the retrieval system with a sample query
print("\n🧪 Testing retrieval system...")

# Use the first train window as a query to test (should be most similar to itself)
test_query = train_windows[0]['embedding']
print(f"🔍 Query from window 0: item_id={train_windows[0]['item_id']}, start_idx={train_windows[0]['start_idx']}")

# Retrieve top 5 most similar windows
similar_windows = retriever.retrieve_similar(test_query, top_k=5, return_similarities=True)

print(f"\n📋 Top 5 most similar windows:")
for i, (window, similarity) in enumerate(similar_windows):
    print(f"  {i+1}. Similarity: {similarity:.4f} | "
          f"item_id: {window['item_id']} | "
          f"start_idx: {window['start_idx']}")

# %%
# 📊 Visualization of Retrieved Similar Windows
import matplotlib.pyplot as plt

def visualize_retrieved_windows(query_window, retrieved_results, max_display=5):
    """
    Visualize the query window and its most similar retrieved windows.
    """
    n_display = min(len(retrieved_results), max_display)
    fig, axes = plt.subplots(2, n_display, figsize=(4*n_display, 8))
    
    if n_display == 1:
        axes = axes.reshape(2, 1)
    
    # Plot query window
    for i in range(n_display):
        # Query history (top row)
        axes[0, i].plot(query_window['history'], 'b-', linewidth=2, label='Query History')
        axes[0, i].plot(range(144, 144+12), query_window['future'], 'r--', linewidth=2, label='Query Future')
        axes[0, i].set_title(f'Query Window\nitem_id: {query_window["item_id"]}', fontsize=10)
        axes[0, i].set_ylabel('Glucose Level')
        axes[0, i].grid(True, alpha=0.3)
        if i == 0:
            axes[0, i].legend()
        
        # Retrieved window (bottom row)
        retrieved_window, similarity = retrieved_results[i]
        axes[1, i].plot(retrieved_window['history'], 'g-', linewidth=2, label='Retrieved History')
        axes[1, i].plot(range(144, 144+12), retrieved_window['future'], 'orange', linestyle='--', linewidth=2, label='Retrieved Future')
        axes[1, i].set_title(f'Similar Window {i+1}\nSim: {similarity:.3f}\nitem_id: {retrieved_window["item_id"]}', fontsize=10)
        axes[1, i].set_ylabel('Glucose Level')
        axes[1, i].set_xlabel('Time Steps')
        axes[1, i].grid(True, alpha=0.3)
        if i == 0:
            axes[1, i].legend()
    
    plt.tight_layout()
    plt.show()

# Visualize the test retrieval
print(f"\n📈 Visualizing query and retrieved similar windows...")
visualize_retrieved_windows(train_windows[0], similar_windows, max_display=5)

# %%
