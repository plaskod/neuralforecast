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
print(f"Data per series: {Y_df.groupby('unique_id').size()}")

# Create TabPFN model
h = 12
input_size = 288
tabpfn_model = TabPFN(h=h, input_size=input_size, loss=MAE())

# Create NeuralForecast with debugging
fcst = NeuralForecast(freq=freq, models=[tabpfn_model])

# Let me debug by trying a regular fit first
print("Fitting model first...")
fcst.fit(df=Y_df, static_df=static_df)

print(f"Dataset n_groups: {fcst.dataset.n_groups}")
print(f"Dataset indptr: {fcst.dataset.indptr}")
print(f"Dataset min_size: {fcst.dataset.min_size}")
print(f"Dataset temporal shape: {fcst.dataset.temporal.shape}")

# Let me debug what happens during regular prediction first
print("\n=== Testing regular prediction ===")
predictions = fcst.models[0].predict(fcst.dataset, step_size=1)
print(f"Regular predictions shape: {predictions.shape}")
print(f"Regular predictions dtype: {predictions.dtype}")

# Let me monkey-patch the _no_refit_cross_validation method to see what's happening
original_cv = fcst._no_refit_cross_validation

def debug_cv(*args, **kwargs):
    print(f"\n=== DEBUG CROSS-VALIDATION ===")
    
    # Import needed utilities  
    import utilsforecast.compat as ufp
    from coreforecast.grouped_array import GroupedArray
    
    # Call part of the original method to get the fcsts_df
    result = None
    try:
        # We need to manually replicate some of the CV logic to debug it
        return original_cv(*args, **kwargs)
    except Exception as e:
        # If it fails, let's see what the issue is
        print(f"CV failed with: {e}")
        
        # Let's manually debug the issue
        # Get the arguments
        df = kwargs.get('df') or args[0] if args else None
        static_df = kwargs.get('static_df') or (args[1] if len(args) > 1 else None)
        val_size = kwargs.get('val_size', 0)
        test_size = kwargs.get('test_size', 0)
        step_size = kwargs.get('step_size', 1)
        id_col = kwargs.get('id_col', 'unique_id')
        time_col = kwargs.get('time_col', 'ds')
        
        # Create CV times manually
        fcsts_df = ufp.cv_times(
            times=fcst.ds,
            uids=fcst.uids,
            indptr=fcst.dataset.indptr,
            h=fcst.h,
            test_size=test_size,
            step_size=step_size,
            id_col=id_col,
            time_col=time_col,
        )
        
        print(f"CV DataFrame shape: {fcsts_df.shape}")
        print(f"CV DataFrame head:")
        print(fcsts_df.head(15))
        
        # Get effective sizes
        effective_sizes = ufp.counts_by_id(fcsts_df, id_col)["counts"].to_numpy()
        print(f"Effective sizes: {effective_sizes}")
        print(f"Sum of effective sizes: {effective_sizes.sum()}")
        
        # Calculate n_windows
        n_windows = len(fcsts_df) // (fcst.dataset.n_groups * fcst.h)
        print(f"Calculated n_windows: {n_windows}")
        print(f"len(fcsts_df): {len(fcsts_df)}")
        print(f"fcst.dataset.n_groups: {fcst.dataset.n_groups}")
        print(f"fcst.h: {fcst.h}")
        
        # Get model predictions
        model_fcsts = fcst.models[0].predict(fcst.dataset, step_size=step_size)
        print(f"Model predictions shape: {model_fcsts.shape}")
        
        # Calculate expected indptr
        expected_indptr = np.arange(
            0,
            n_windows * fcst.h * (fcst.dataset.n_groups + 1),
            n_windows * fcst.h,
            dtype=np.int32,
        )
        print(f"Expected indptr: {expected_indptr}")
        print(f"Model predictions should have shape: ({expected_indptr[-1]}, 1)")
        print(f"Actual model predictions shape: {model_fcsts.shape}")
        print(f"Shape mismatch: {model_fcsts.shape[0]} != {expected_indptr[-1]}")
        
        raise e

fcst._no_refit_cross_validation = debug_cv

# Try cross-validation with reduced parameters
try:
    print("\nStarting cross-validation...")
    fcst_df = fcst.cross_validation(
        df=Y_df, 
        static_df=static_df,
        val_size=100,  # Reduced
        test_size=100,  # Reduced
        step_size=1,
        n_windows=None,  # Like in the original training script
    )
    print("Cross-validation completed successfully!")
    print(f"Result shape: {fcst_df.shape}")
except Exception as e:
    print(f"Cross-validation failed: {e}")
    import traceback
    traceback.print_exc() 