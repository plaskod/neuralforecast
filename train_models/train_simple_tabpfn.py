import os
import pickle
import time
import argparse
import pandas as pd
import numpy as np
import sys

from .data_parameters import get_data_parameters

from neuralforecast.models.simple_tabpfn import TabPFN
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

def main(args):

    #----------------------------------------------- Load Data -----------------------------------------------#
    data_dir, static_dir, val_size, test_size, freq, exog = get_data_parameters(args)

    Y_df = pd.read_csv(data_dir)
    if Y_df.ds.dtype != '<M8[ns]':
        Y_df.ds = pd.to_datetime(Y_df.ds, format='%Y-%m-%d %H:%M:%S')

    static_df = None
    if static_dir is not None:
        static_df = pd.read_csv(static_dir)
        
    args.exog = exog
    args.n_series = len(Y_df.unique_id.unique())
    args.freq = freq

    #----------------------------------------------- Training -----------------------------------------------#

    print(50*'-', args.dataset, 50*'-')
    print(50*'-', args.horizon, 50*'-')
    print(50*'-', args.input_size, 50*'-')
    print("Using Simple TabPFN implementation for cross-validation")
    print(f"Dataset: {args.dataset}")
    print(f"Number of series: {args.n_series}")
    print(f"Max context length: {args.max_context_length}")
    print(f"Horizon: {args.horizon}")
    print(f"Input size: {args.input_size}")
    print(f"Val size: {val_size}, Test size: {test_size}")
    
    start = time.time()
        
    results_dir = f'{args.results_dir}/{args.dataset}_{args.horizon}/simple_tabpfn/trial_{args.experiment_id}'
    os.makedirs(results_dir, exist_ok = True)

    # Initialize Simple TabPFN model with optimized parameters
    print("Initializing TabPFN model...")
    simple_tabpfn_model = TabPFN(
        h=args.horizon,
        input_size=args.input_size,
        max_context_length=args.max_context_length,
        loss=MAE()
    )

    fcst = NeuralForecast(freq=freq,
                          models=[simple_tabpfn_model])

    print("Starting cross-validation with Simple TabPFN...")
    cv_start = time.time()
    
    # Follow the exact same pattern as other baseline models
    fcst_df = fcst.cross_validation(df=Y_df, 
                                    static_df=static_df,
                                    val_size=val_size,
                                    test_size=test_size,
                                    step_size=1,
                                    n_windows=None,
                                    )
    
    cv_time = time.time() - cv_start
    print(f"Cross-validation completed in {cv_time:.2f} seconds")
    
    # Save results
    print("Saving results...")
    fcst_df.to_csv(results_dir+f'/forecasts.csv', index=False)
    print(f"Forecasts saved to: {results_dir}/forecasts.csv")
    print(f"Forecast shape: {fcst_df.shape}")
    
    # Save the model following the same pattern as other baselines
    fcst.save(path=results_dir,
              model_index=None,
              overwrite=True,
              save_dataset=False)
    print(f"Model saved to: {results_dir}")

    total_time = time.time() - start
    print(f'Cross-validation completed successfully')
    print(f'Total time: {total_time:.2f} seconds')
    print(f'Average time per series: {total_time/args.n_series:.2f} seconds')
        
def parse_args():
    desc = "Example of Simple TabPFN cross-validation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--results_dir', type=str, default='./results', help='results_dir')
    parser.add_argument('--dataset', type=str, default='ohiot1dm', help='dataset name')
    parser.add_argument('--horizon', type=int, default=12, help='forecast horizon')
    parser.add_argument('--input_size', type=int, default=288, help='input size')
    parser.add_argument('--max_context_length', type=int, default=8000, help='maximum context length for TabPFN')
    parser.add_argument('--experiment_id', default=1, required=False, type=int, help='string to identify experiment')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args) 