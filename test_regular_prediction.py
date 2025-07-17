import pandas as pd
import numpy as np
from neuralforecast.core import NeuralForecast
from neuralforecast.models.tabpfn import TabPFN
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

# Create TabPFN model
h = 12
input_size = 288

print("\n=== Testing TabPFN Regular Prediction ===")
tabpfn_model = TabPFN(h=h, input_size=input_size, loss=MAE())
fcst_tabpfn = NeuralForecast(freq=freq, models=[tabpfn_model])

# Test regular fit and predict (not cross-validation)
try:
    fcst_tabpfn.fit(df=Y_df, static_df=static_df)
    print("TabPFN fit succeeded!")
    
    # Test regular prediction
    predictions = fcst_tabpfn.predict()
    print(f"TabPFN regular prediction succeeded!")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions sample:")
    print(predictions.head(10))
    
except Exception as e:
    print(f"TabPFN regular prediction failed: {e}")
    import traceback
    traceback.print_exc() 