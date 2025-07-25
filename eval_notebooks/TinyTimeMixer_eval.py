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
    
    def retrieve_similar(self, query_embedding, top_k=5, return_similarities=True, 
                        temporal_filter=False, temporal_threshold=20):
        """
        Retrieve top-k most similar train windows for a query embedding.
        
        Args:
            query_embedding: numpy array of shape (embedding_dim,)
            top_k: number of most similar windows to retrieve
            return_similarities: whether to return similarity scores
            temporal_filter: if True, filter out temporally close windows from same item_id
            temporal_threshold: minimum timestep difference to consider windows as different
            
        Returns:
            List of tuples: (window_dict, similarity_score) if return_similarities=True
            List of window_dict if return_similarities=False
        """
        # Ensure query_embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.train_embeddings)[0]
        
        if temporal_filter:
            # Get all candidates sorted by similarity
            all_indices = np.argsort(similarities)[::-1]  # All indices, best first
            
            # Apply temporal filtering first
            filtered_results = []
            used_signatures = set()
            remaining_indices = []
            
            for idx in all_indices:
                window = self.train_windows[idx]
                sim_score = similarities[idx]
                item_id = window['item_id']
                start_idx = window['start_idx']
                
                # Check if this window is too close to any already selected window
                is_too_close = any(
                    item_id == used_item and abs(start_idx - used_start) < temporal_threshold
                    for used_item, used_start in used_signatures
                )
                
                if not is_too_close:
                    used_signatures.add((item_id, start_idx))
                    if return_similarities:
                        filtered_results.append((window, sim_score))
                    else:
                        filtered_results.append(window)
                else:
                    # Keep track of filtered-out indices for potential backfill
                    remaining_indices.append(idx)
                
                # Stop filtering when we have enough or processed all
                if len(filtered_results) >= top_k:
                    break
            
            # If we don't have enough filtered results, backfill with remaining best matches
            results = filtered_results[:top_k]
            if len(results) < top_k:
                needed = top_k - len(results)
                for idx in remaining_indices[:needed]:
                    window = self.train_windows[idx]
                    sim_score = similarities[idx]
                    if return_similarities:
                        results.append((window, sim_score))
                    else:
                        results.append(window)
            
            return results
        else:
            # Original behavior without temporal filtering
            top_k_indices = np.argsort(similarities)[-top_k:][::-1]
            
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
similar_windows = retriever.retrieve_similar(test_query, top_k=10, return_similarities=True, temporal_filter=True, temporal_threshold=20)

print(f"\n📋 Top 5 most similar windows:")
for i, (window, similarity) in enumerate(similar_windows):
    print(f"  {i+1}. Similarity: {similarity:.4f} | "
          f"item_id: {window['item_id']} | "
          f"start_idx: {window['start_idx']}")

# %%
# 📊 Visualization of Retrieved Similar Windows
# import matplotlib.pyplot as plt

# def visualize_retrieved_windows(query_window, retrieved_results, max_display=5):
#     """
#     Visualize the query window and its most similar retrieved windows.
#     """
#     n_display = min(len(retrieved_results), max_display)
#     fig, axes = plt.subplots(2, n_display, figsize=(4*n_display, 8))
    
#     if n_display == 1:
#         axes = axes.reshape(2, 1)
    
#     # Plot query window
#     for i in range(n_display):
#         # Query history (top row)
#         axes[0, i].plot(query_window['history'], 'b-', linewidth=2, label='Query History')
#         axes[0, i].plot(range(144, 144+12), query_window['future'], 'r--', linewidth=2, label='Query Future')
#         axes[0, i].set_title(f'Query Window\nitem_id: {query_window["item_id"]}', fontsize=10)
#         axes[0, i].set_ylabel('Glucose Level')
#         axes[0, i].grid(True, alpha=0.3)
#         if i == 0:
#             axes[0, i].legend()
        
#         # Retrieved window (bottom row)
#         retrieved_window, similarity = retrieved_results[i]
#         axes[1, i].plot(retrieved_window['history'], 'g-', linewidth=2, label='Retrieved History')
#         axes[1, i].plot(range(144, 144+12), retrieved_window['future'], 'orange', linestyle='--', linewidth=2, label='Retrieved Future')
#         axes[1, i].set_title(f'Similar Window {i+1}\nSim: {similarity:.3f}\nitem_id: {retrieved_window["item_id"]}', fontsize=10)
#         axes[1, i].set_ylabel('Glucose Level')
#         axes[1, i].set_xlabel('Time Steps')
#         axes[1, i].grid(True, alpha=0.3)
#         if i == 0:
#             axes[1, i].legend()
    
#     plt.tight_layout()
#     plt.show()

# # Visualize the test retrieval
# print(f"\n📈 Visualizing query and retrieved similar windows...")
# visualize_retrieved_windows(train_windows[0], similar_windows, max_display=5)

# %%
# 🔍 VERIFICATION: Check uniqueness and test with random queries from test set
print("🔍 VERIFICATION: Testing retrieval system comprehensively...")

# First, let's compute embeddings for some test windows from 'windows'
def compute_test_embeddings(test_windows, model, batch_size=32):
    """Compute embeddings for test windows"""
    test_embeddings = []
    
    print(f"Computing embeddings for {len(test_windows)} test windows...")
    
    for i in tqdm(range(0, len(test_windows), batch_size), desc="Computing test embeddings"):
        batch_end = min(i + batch_size, len(test_windows))
        batch_histories = []
        
        for j in range(i, batch_end):
            history = test_windows[j]['history']
            history_tensor = torch.FloatTensor(history).unsqueeze(0)
            batch_histories.append(history_tensor)
        
        batch_tensor = torch.stack(batch_histories).to(device)
        
        with torch.no_grad():
            outputs = model(x_enc=batch_tensor)
            embeddings = outputs.embeddings.cpu().numpy()
            test_embeddings.extend(embeddings)
    
    return np.array(test_embeddings)

# Compute embeddings for a subset of test windows
n_test_samples = min(50, len(windows))  # Don't compute too many for efficiency
test_sample_indices = np.random.choice(len(windows), n_test_samples, replace=False)
test_sample_windows = [windows[i] for i in test_sample_indices]

print(f"\n🎲 Computing embeddings for {n_test_samples} random test windows...")
test_embeddings = compute_test_embeddings(test_sample_windows, model)

# %%
# 🧪 Uniqueness verification function
def verify_retrieval_uniqueness(retriever, test_windows, test_embeddings, n_queries=5):
    """
    Verify that retrieved windows are unique and different from query
    """
    print(f"\n🔍 UNIQUENESS VERIFICATION with {n_queries} random test queries:")
    
    # Select random test queries
    query_indices = np.random.choice(len(test_windows), n_queries, replace=False)
    
    for i, query_idx in enumerate(query_indices):
        query_window = test_windows[query_idx]
        query_embedding = test_embeddings[query_idx]
        
        print(f"\n📋 Query {i+1}: item_id={query_window['item_id']}, start_idx={query_window['start_idx']}")
        
        # Retrieve similar windows
        similar_results = retriever.retrieve_similar(query_embedding, top_k=5, return_similarities=True, temporal_filter=True, temporal_threshold=20)
        
        # Check uniqueness
        retrieved_signatures = []
        for j, (retrieved_window, similarity) in enumerate(similar_results):
            signature = f"{retrieved_window['item_id']}_{retrieved_window['start_idx']}"
            retrieved_signatures.append(signature)
            
            print(f"  {j+1}. Sim: {similarity:.4f} | "
                  f"item_id: {retrieved_window['item_id']} | "
                  f"start_idx: {retrieved_window['start_idx']}")
        
        # Check for duplicates
        unique_signatures = set(retrieved_signatures)
        if len(unique_signatures) != len(retrieved_signatures):
            print(f"  ⚠️  WARNING: Found duplicate retrievals!")
        else:
            print(f"  ✅ All retrieved windows are unique")
        
        # Check similarity distribution
        similarities = [sim for _, sim in similar_results]
        print(f"  📊 Similarity range: {min(similarities):.4f} - {max(similarities):.4f}")

# Run uniqueness verification
verify_retrieval_uniqueness(retriever, test_sample_windows, test_embeddings, n_queries=5)

# %%
# 📊 COMPREHENSIVE VISUALIZATION: Multiple queries and their retrievals
def visualize_multiple_queries_and_retrievals(test_windows, test_embeddings, retriever, n_queries=4, top_k=5):
    """
    Visualize multiple test queries and their top-k retrieved training windows
    Each row shows: [Query] [Retrieved 1] [Retrieved 2] [Retrieved 3] [Retrieved 4] [Retrieved 5]
    """
    # Select random queries
    query_indices = np.random.choice(len(test_windows), n_queries, replace=False)
    
    fig, axes = plt.subplots(n_queries, top_k + 1, figsize=(4*(top_k + 1), 4*n_queries))
    
    if n_queries == 1:
        axes = axes.reshape(1, -1)
    
    for row, query_idx in enumerate(query_indices):
        query_window = test_windows[query_idx]
        query_embedding = test_embeddings[query_idx]
        
        # Plot query window (first column)
        ax_query = axes[row, 0]
        ax_query.plot(query_window['history'], 'b-', linewidth=2, label='Query History')
        ax_query.plot(range(144, 144+12), query_window['future'], 'r--', linewidth=2, label='Query Future')
        ax_query.set_title(f'QUERY {row+1}\nitem_id: {query_window["item_id"]}\nstart_idx: {query_window["start_idx"]}', 
                          fontsize=10, fontweight='bold')
        ax_query.set_ylabel('Glucose Level')
        ax_query.grid(True, alpha=0.3)
        ax_query.legend(fontsize=8)
        
        # Retrieve and plot similar windows
        similar_results = retriever.retrieve_similar(query_embedding, top_k=top_k, return_similarities=True, temporal_filter=True, temporal_threshold=20)
        
        for col, (retrieved_window, similarity) in enumerate(similar_results, 1):
            ax_retrieved = axes[row, col]
            ax_retrieved.plot(retrieved_window['history'], 'g-', linewidth=2, label='Retrieved History')
            ax_retrieved.plot(range(144, 144+12), retrieved_window['future'], 'orange', 
                            linestyle='--', linewidth=2, label='Retrieved Future')
            ax_retrieved.set_title(f'RETRIEVED {col}\nSim: {similarity:.3f}\nitem_id: {retrieved_window["item_id"]}\nstart_idx: {retrieved_window["start_idx"]}', 
                                 fontsize=9)
            ax_retrieved.grid(True, alpha=0.3)
            
            if col == 1:  # Only show legend for first retrieved window
                ax_retrieved.legend(fontsize=8)
        
        # Set x-label only for bottom row
        if row == n_queries - 1:
            for col in range(top_k + 1):
                axes[row, col].set_xlabel('Time Steps')
    
    plt.suptitle('Test Queries and Retrieved Training Windows (Cosine Similarity)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Create comprehensive visualization
print(f"\n📊 Creating comprehensive visualization of queries and retrievals...")
visualize_multiple_queries_and_retrievals(test_sample_windows, test_embeddings, retriever, n_queries=4, top_k=5)

# %%
# 📈 Statistical analysis of retrieval quality
def analyze_retrieval_statistics(test_windows, test_embeddings, retriever, n_samples=20):
    """
    Analyze statistical properties of the retrieval system
    """
    print(f"\n📈 STATISTICAL ANALYSIS of retrieval system:")
    
    query_indices = np.random.choice(len(test_windows), n_samples, replace=False)
    
    all_similarities = []
    similarity_distributions = []
    
    for query_idx in query_indices:
        query_embedding = test_embeddings[query_idx]
        similar_results = retriever.retrieve_similar(query_embedding, top_k=10, return_similarities=True)
        
        similarities = [sim for _, sim in similar_results]
        all_similarities.extend(similarities)
        similarity_distributions.append(similarities)
    
    all_similarities = np.array(all_similarities)
    
    print(f"📊 Overall similarity statistics (across {n_samples} queries, top-10 each):")
    print(f"  • Mean similarity: {all_similarities.mean():.4f}")
    print(f"  • Std similarity: {all_similarities.std():.4f}")
    print(f"  • Min similarity: {all_similarities.min():.4f}")
    print(f"  • Max similarity: {all_similarities.max():.4f}")
    print(f"  • Median similarity: {np.median(all_similarities):.4f}")
    
    # Plot similarity distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_similarities, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(all_similarities.mean(), color='red', linestyle='--', label=f'Mean: {all_similarities.mean():.3f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Retrieval Similarities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    similarity_by_rank = np.array(similarity_distributions)
    mean_by_rank = similarity_by_rank.mean(axis=0)
    std_by_rank = similarity_by_rank.std(axis=0)
    
    ranks = np.arange(1, len(mean_by_rank) + 1)
    plt.errorbar(ranks, mean_by_rank, yerr=std_by_rank, marker='o', capsize=5)
    plt.xlabel('Retrieval Rank')
    plt.ylabel('Mean Cosine Similarity')
    plt.title('Similarity by Retrieval Rank')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return all_similarities, similarity_distributions

# Run statistical analysis
similarity_stats = analyze_retrieval_statistics(test_sample_windows, test_embeddings, retriever, n_samples=20)

# %%
# 🎯 NEW VISUALIZATION: Test Windows and Their Top-5 Training Retrievals
import matplotlib.pyplot as plt
import numpy as np

def visualize_test_queries_with_training_retrievals(test_windows, test_embeddings, retriever, n_queries=3, top_k=5):
    """
    Visualize random test windows and their top-k retrieved training windows.
    
    Args:
        test_windows: List of test window dictionaries
        test_embeddings: Array of test embeddings  
        retriever: MOMENTRetriever instance
        n_queries: Number of random test queries to show
        top_k: Number of training retrievals to show per query
    """
    # Select random test queries
    query_indices = np.random.choice(len(test_windows), n_queries, replace=False)
    
    # Create subplot grid: n_queries rows, (1 + top_k) columns
    fig, axes = plt.subplots(n_queries, 1 + top_k, figsize=(4*(1 + top_k), 4*n_queries))
    
    # Handle single query case
    if n_queries == 1:
        axes = axes.reshape(1, -1)
    
    for row, query_idx in enumerate(query_indices):
        # Get test query data
        test_window = test_windows[query_idx]
        test_embedding = test_embeddings[query_idx]
        
        # Retrieve top-k similar training windows
        similar_results = retriever.retrieve_similar(
            test_embedding, top_k=top_k, return_similarities=True, 
            temporal_filter=True, temporal_threshold=20
        )
        
        # Plot test query (first column)
        ax_query = axes[row, 0]
        ax_query.plot(test_window['history'], 'blue', linewidth=2.5, label='Test History')
        ax_query.plot(range(144, 144+12), test_window['future'], 'red', 
                     linestyle='--', linewidth=2.5, label='Test Future')
        ax_query.set_title(f'TEST QUERY {row+1}\nitem_id: {test_window["item_id"]}\nstart_idx: {test_window["start_idx"]}', 
                          fontsize=11, fontweight='bold', color='darkblue')
        ax_query.set_ylabel('Glucose Level', fontsize=10)
        ax_query.grid(True, alpha=0.3)
        ax_query.legend(fontsize=9)
        
        # Add colored border for test query
        for spine in ax_query.spines.values():
            spine.set_edgecolor('darkblue')
            spine.set_linewidth(2)
        
        # Plot retrieved training windows (remaining columns)
        for col, (train_window, similarity) in enumerate(similar_results, 1):
            ax_train = axes[row, col]
            
            # Use different colors for different similarity ranges
            if similarity > 0.8:
                color_hist, color_fut = 'darkgreen', 'green'
                border_color = 'darkgreen'
            elif similarity > 0.6:
                color_hist, color_fut = 'orange', 'darkorange' 
                border_color = 'orange'
            else:
                color_hist, color_fut = 'purple', 'mediumpurple'
                border_color = 'purple'
            
            ax_train.plot(train_window['history'], color_hist, linewidth=2, label='Train History')
            ax_train.plot(range(144, 144+12), train_window['future'], color_fut,
                         linestyle='--', linewidth=2, label='Train Future')
            ax_train.set_title(f'RANK {col}\nSim: {similarity:.3f}\nitem_id: {train_window["item_id"]}\nstart_idx: {train_window["start_idx"]}', 
                              fontsize=10, color=border_color)
            ax_train.grid(True, alpha=0.3)
            
            # Add colored border based on similarity
            for spine in ax_train.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(1.5)
            
            # Add legend only for first retrieved window
            if col == 1:
                ax_train.legend(fontsize=9)
        
        # Set x-labels for bottom row only
        if row == n_queries - 1:
            for col in range(1 + top_k):
                axes[row, col].set_xlabel('Time Steps', fontsize=10)
    
    # Set overall title and layout
    plt.suptitle(f'Test Queries and Top-{top_k} Retrieved Training Windows\n(Colors: Green=High Sim, Orange=Med Sim, Purple=Low Sim)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 RETRIEVAL SUMMARY:")
    print(f"🔍 DATA STRUCTURE: Intra-patient retrieval from continuous glucose monitoring")
    print(f"   • Test and train windows are different rolling windows from SAME patient's time series")
    print(f"   • High similarities (0.95+) are EXPECTED - overlapping/nearby glucose patterns")
    print(f"   • This demonstrates finding similar glucose dynamics within patient history")
    print()
    
    for i, query_idx in enumerate(query_indices):
        test_window = test_windows[query_idx]
        test_embedding = test_embeddings[query_idx]
        similar_results = retriever.retrieve_similar(
            test_embedding, top_k=top_k, return_similarities=True,
            temporal_filter=True, temporal_threshold=20
        )
        
        similarities = [sim for _, sim in similar_results]
        retrieved_items = [w['item_id'] for w, _ in similar_results]
        retrieved_starts = [w['start_idx'] for w, _ in similar_results]
        
        print(f"  Query {i+1} (item_id: {test_window['item_id']}, start_idx: {test_window['start_idx']}):")
        print(f"    • Similarities: {[f'{s:.3f}' for s in similarities]}")
        print(f"    • Retrieved start_indices: {retrieved_starts}")
        print(f"    • Temporal gaps from query: {[abs(test_window['start_idx'] - s) for s in retrieved_starts]}")
        print(f"    • Unique items retrieved: {len(set(retrieved_items))}/{top_k}")

def visualize_full_timeline_with_chunks(test_windows, test_embeddings, retriever, train_windows, n_queries=3, top_k=5):
    """
    Visualize full continuous glucose time series with highlighted query and retrieved chunks.
    """
    # Reconstruct the full continuous time series
    patient_id = train_windows[0]['item_id']
    
    # Get all windows sorted by start_idx
    train_patient_windows = [(w['start_idx'], w) for w in train_windows if w['item_id'] == patient_id]
    test_patient_windows = [(w['start_idx'], w) for w in test_windows if w['item_id'] == patient_id]
    all_windows = train_patient_windows + test_patient_windows
    all_windows.sort(key=lambda x: x[0])
    
    # Find series range
    min_start = min(start for start, _ in all_windows)
    max_start = max(start for start, _ in all_windows)
    series_length = max_start + 156  # 144 + 12 for future
    
    # Reconstruct continuous series using overlapping windows
    full_series = np.full(series_length, np.nan)
    for start_idx, window in all_windows:
        # Fill history
        end_hist = start_idx + 144
        if 0 <= start_idx < series_length and end_hist <= series_length:
            full_series[start_idx:end_hist] = window['history']
        # Fill future
        start_fut = start_idx + 144
        end_fut = start_fut + 12
        if 0 <= start_fut < series_length and end_fut <= series_length:
            full_series[start_fut:end_fut] = window['future']
    
    # Select random test queries
    query_indices = np.random.choice(len(test_windows), n_queries, replace=False)
    
    # Create plots
    fig, axes = plt.subplots(n_queries, 1, figsize=(20, 4*n_queries))
    if n_queries == 1:
        axes = [axes]
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    
    for row, query_idx in enumerate(query_indices):
        ax = axes[row]
        test_window = test_windows[query_idx]
        test_embedding = test_embeddings[query_idx]
        query_start = test_window['start_idx']
        
        # Get retrieved windows
        similar_results = retriever.retrieve_similar(
            test_embedding, top_k=top_k, return_similarities=True, 
            temporal_filter=True, temporal_threshold=20
        )
        
        # Plot full series
        time_axis = np.arange(len(full_series))
        ax.plot(time_axis, full_series, 'lightgray', linewidth=1, alpha=0.7, label='Full Glucose Series')
        
        # Highlight query window
        query_end = query_start + 144
        query_mask = (time_axis >= query_start) & (time_axis < query_end)
        ax.plot(time_axis[query_mask], full_series[query_mask], 'blue', linewidth=4, 
               label=f'QUERY (start: {query_start})')
        ax.axvspan(query_start, query_end, alpha=0.3, color='blue')
        ax.axvline(query_start, color='blue', linestyle='--', linewidth=2)
        ax.axvline(query_end, color='blue', linestyle='--', linewidth=2)
        
        # Highlight retrieved windows
        for i, (retrieved_window, similarity) in enumerate(similar_results):
            color = colors[i % len(colors)]
            retr_start = retrieved_window['start_idx']
            retr_end = retr_start + 144
            
            retr_mask = (time_axis >= retr_start) & (time_axis < retr_end)
            ax.plot(time_axis[retr_mask], full_series[retr_mask], color, linewidth=3, 
                   label=f'RANK {i+1} (sim: {similarity:.3f}, start: {retr_start})')
            ax.axvspan(retr_start, retr_end, alpha=0.2, color=color)
            ax.axvline(retr_start, color=color, linestyle=':', linewidth=1.5)
            ax.axvline(retr_end, color=color, linestyle=':', linewidth=1.5)
        
        # Formatting
        ax.set_title(f'Query {row+1}: Full Glucose Timeline - Patient {patient_id}\n'
                    f'Query at timestep {query_start} with {top_k} similar chunks highlighted', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps (Continuous Glucose Monitoring)', fontsize=11)
        ax.set_ylabel('Glucose Level', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Focus on relevant region
        focus_start = max(0, query_start - 300)
        focus_end = min(len(full_series), query_start + 500)
        ax.set_xlim(focus_start, focus_end)
    
    plt.tight_layout()
    plt.show()

# Create the new timeline visualization
print("🎯 Creating Full Timeline Visualization with Highlighted Chunks...")
print("   📝 Shows complete glucose time series with query and retrieved chunks highlighted")
visualize_full_timeline_with_chunks(
    test_sample_windows, test_embeddings, retriever, train_windows, n_queries=3, top_k=5
)

# Create the original visualization  
print("\n🎯 Creating Test Queries with Training Retrievals Visualization...")
print("   📝 Note: This shows intra-patient glucose pattern retrieval")
print("   📝 Test and train are rolling windows from the same continuous glucose time series")
visualize_test_queries_with_training_retrievals(
    test_sample_windows, test_embeddings, retriever, n_queries=3, top_k=5
)

# %%
# 📅 NEW TIMESTAMP-BASED VISUALIZATION: Real dates with train/test split and retrievals
def visualize_timestamp_based_retrieval(test_windows, test_embeddings, retriever, train_tsdf, test_tsdf, n_queries=3, top_k=5):
    """
    Visualize glucose time series with real timestamps, showing train/test split and highlighted retrieval results.
    
    Args:
        test_windows: List of test window dictionaries
        test_embeddings: Array of test embeddings
        retriever: MOMENTRetriever instance
        train_tsdf: Training time series dataframe with timestamps
        test_tsdf: Test time series dataframe with timestamps  
        n_queries: Number of random test queries to show
        top_k: Number of training retrievals to highlight
    """
    import random
    import matplotlib.dates as mdates
    
    # Get the patient data
    item_id = '563'
    train_series = train_tsdf.loc[item_id].reset_index()
    test_series = test_tsdf.loc[item_id].reset_index()
    
    # Select random test queries
    query_indices = np.random.choice(len(test_windows), n_queries, replace=False)
    
    # Create subplot grid
    fig, axes = plt.subplots(n_queries, 1, figsize=(24, 6*n_queries))
    if n_queries == 1:
        axes = [axes]
    
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    
    for row, query_idx in enumerate(query_indices):
        ax = axes[row]
        
        # Get test query data
        test_window = test_windows[query_idx]
        test_embedding = test_embeddings[query_idx]
        query_start_idx = test_window['start_idx']
        
        # Plot train and test data with timestamps
        ax.plot(train_series['timestamp'], train_series['target'], 'b-', linewidth=1.5, label='Train Data')
        ax.plot(test_series['timestamp'], test_series['target'], 'gray', linewidth=1.5, label='Test Data')
        
        # Add vertical line at train/test split
        split_point = train_series['timestamp'].iloc[-1]
        ax.axvline(x=split_point, color='black', linestyle='--', linewidth=2, label='Train/Test Split')
        
        # Highlight the test query window
        context_start = query_start_idx
        context_end = context_start + 144
        horizon_end = context_end + 12
        
        # Get corresponding timestamps for the query
        context_timestamps = test_series['timestamp'].iloc[context_start:context_end]
        horizon_timestamps = test_series['timestamp'].iloc[context_end:horizon_end]
        
        # Plot query context and horizon
        ax.plot(context_timestamps, test_series['target'].iloc[context_start:context_end], 
                'blue', linewidth=4, label=f'TEST QUERY (start_idx: {query_start_idx})')
        ax.plot(horizon_timestamps, test_series['target'].iloc[context_end:horizon_end], 
                'darkblue', linewidth=4, linestyle='--', label='Query Horizon (12 points)')
        
        # Add shading for query window
        if len(context_timestamps) > 1:
            ax.axvspan(context_timestamps.iloc[0], context_timestamps.iloc[-1], 
                      alpha=0.3, color='blue', edgecolor=None)
        if len(horizon_timestamps) > 1:
            ax.axvspan(horizon_timestamps.iloc[0], horizon_timestamps.iloc[-1], 
                      alpha=0.2, color='darkblue', edgecolor=None)
        
        # Retrieve similar training windows
        similar_results = retriever.retrieve_similar(
            test_embedding, top_k=top_k, return_similarities=True,
            temporal_filter=True, temporal_threshold=20
        )
        
        # Highlight each retrieved training window
        for i, (retrieved_window, similarity) in enumerate(similar_results):
            color = colors[i % len(colors)]
            retr_start_idx = retrieved_window['start_idx']
            retr_context_end = retr_start_idx + 144
            retr_horizon_end = retr_context_end + 12
            
            # Get corresponding timestamps in training data
            if retr_start_idx < len(train_series) and retr_context_end <= len(train_series):
                retr_context_timestamps = train_series['timestamp'].iloc[retr_start_idx:retr_context_end]
                retr_context_values = train_series['target'].iloc[retr_start_idx:retr_context_end]
                
                # Plot retrieved context window
                ax.plot(retr_context_timestamps, retr_context_values, color, linewidth=3, 
                       label=f'RANK {i+1} (sim: {similarity:.3f}, start: {retr_start_idx})')
                
                # Add shading for retrieved window
                if len(retr_context_timestamps) > 1:
                    ax.axvspan(retr_context_timestamps.iloc[0], retr_context_timestamps.iloc[-1], 
                              alpha=0.15, color=color, edgecolor=None)
                
                # Add vertical lines at boundaries
                ax.axvline(retr_context_timestamps.iloc[0], color=color, linestyle=':', linewidth=1.5, alpha=0.8)
                ax.axvline(retr_context_timestamps.iloc[-1], color=color, linestyle=':', linewidth=1.5, alpha=0.8)
        
        # Format the x-axis to show dates properly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.tick_params(axis='x', rotation=45)
        
        # Formatting
        ax.set_title(f'Query {row+1}: Glucose Time Series with Retrieval Results\n'
                    f'Patient #{item_id} - Test Query at start_idx {query_start_idx} + Top {top_k} Retrieved Training Windows', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Glucose Level', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable time range to focus on relevant period
        query_timestamp = test_series['timestamp'].iloc[query_start_idx]
        focus_start = query_timestamp - pd.Timedelta(days=3)
        focus_end = query_timestamp + pd.Timedelta(days=2)
        ax.set_xlim(focus_start, focus_end)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary with timestamps
    print(f"\n📊 TIMESTAMP-BASED RETRIEVAL SUMMARY:")
    print(f"🔍 Real-time glucose monitoring with date/time context")
    print()
    
    for i, query_idx in enumerate(query_indices):
        test_window = test_windows[query_idx]
        test_embedding = test_embeddings[query_idx]
        query_start_idx = test_window['start_idx']
        query_timestamp = test_series['timestamp'].iloc[query_start_idx]
        
        similar_results = retriever.retrieve_similar(
            test_embedding, top_k=top_k, return_similarities=True,
            temporal_filter=True, temporal_threshold=20
        )
        
        print(f"  Query {i+1}: Test window at {query_timestamp.strftime('%Y-%m-%d %H:%M')} (start_idx: {query_start_idx})")
        
        for j, (retrieved_window, similarity) in enumerate(similar_results):
            retr_start_idx = retrieved_window['start_idx']
            if retr_start_idx < len(train_series):
                retr_timestamp = train_series['timestamp'].iloc[retr_start_idx]
                time_diff = query_timestamp - retr_timestamp
                print(f"    Rank {j+1}: {retr_timestamp.strftime('%Y-%m-%d %H:%M')} "
                      f"(sim: {similarity:.3f}, {time_diff.days} days before query)")

# Create the timestamp-based visualization
print("📅 Creating Timestamp-Based Retrieval Visualization...")
print("   📝 Shows real dates/times with train/test split and highlighted retrievals")
visualize_timestamp_based_retrieval(
    test_sample_windows, test_embeddings, retriever, train_tsdf, test_tsdf, n_queries=3, top_k=5
)

# %%
