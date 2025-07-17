import pandas as pd
import numpy as np
from neuralforecast.core import NeuralForecast
from neuralforecast.models.tabpfn import TabPFN
from neuralforecast.models.mlp import MLP
from neuralforecast.losses.pytorch import MAE

# Load the same data as the training script
from train_models.data_parameters import get_data_parameters

class Args:
    dataset = 'ohiot1dm'

args = Args()
data_dir, static_dir, val_size, test_size, freq, exog = get_data_parameters(args)

# Load a small subset of data for debugging
Y_df = pd.read_csv(data_dir)
if Y_df.ds.dtype != '<M8[ns]':
    Y_df.ds = pd.to_datetime(Y_df.ds, format='%Y-%m-%d %H:%M:%S')

# Take only first 2 series and limit data for faster debugging
unique_ids = Y_df.unique_id.unique()[:2]  
Y_df = Y_df[Y_df.unique_id.isin(unique_ids)].reset_index(drop=True)

# Reduce data size to make debugging faster
Y_df = Y_df.groupby('unique_id').head(1000).reset_index(drop=True)

static_df = None
if static_dir is not None:
    static_df = pd.read_csv(static_dir)
    static_df = static_df[static_df.unique_id.isin(unique_ids)].reset_index(drop=True)

print(f"Dataset shape: {Y_df.shape}")
print(f"Number of series: {len(unique_ids)}")

# Create models
h = 12
input_size = 288

# Test MLP first (known working model)
print("\n=== Testing MLP ===")
mlp_model = MLP(h=h, input_size=input_size, loss=MAE(), max_steps=5)
fcst_mlp = NeuralForecast(freq=freq, models=[mlp_model])

try:
    fcst_df_mlp = fcst_mlp.cross_validation(
        df=Y_df, 
        static_df=static_df,
        val_size=100,
        test_size=100,
        step_size=1,
        n_windows=None,
    )
    print("MLP cross-validation succeeded!")
    print(f"MLP result shape: {fcst_df_mlp.shape}")
except Exception as e:
    print(f"MLP cross-validation failed: {e}")
    import traceback
    traceback.print_exc()

# Test TabPFN
print("\n=== Testing TabPFN ===")
tabpfn_model = TabPFN(h=h, input_size=input_size, loss=MAE())
fcst_tabpfn = NeuralForecast(freq=freq, models=[tabpfn_model])

try:
    fcst_df_tabpfn = fcst_tabpfn.cross_validation(
        df=Y_df, 
        static_df=static_df,
        val_size=100,
        test_size=100,
        step_size=1,
        n_windows=None,
    )
    print("TabPFN cross-validation succeeded!")
    print(f"TabPFN result shape: {fcst_df_tabpfn.shape}")
except Exception as e:
    print(f"TabPFN cross-validation failed: {e}")
    import traceback
    traceback.print_exc() 