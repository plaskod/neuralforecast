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
    print("Using Simple TabPFN implementation")
    start = time.time()
        
    results_dir = f'{args.results_dir}/{args.dataset}_{args.horizon}/simple_tabpfn/trial_{args.experiment_id}'
    os.makedirs(results_dir, exist_ok = True)

    # Initialize Simple TabPFN model with optimized parameters
    simple_tabpfn_model = TabPFN(
        h=args.horizon,
        input_size=args.input_size,
        max_context_length=args.max_context_length,
        loss=MAE()
    )

    fcst = NeuralForecast(freq=freq,
                          models=[simple_tabpfn_model])

    print("Starting cross-validation with Simple TabPFN")
    try:
        fcst_df = fcst.cross_validation(df=Y_df, 
                                        static_df=static_df,
                                        val_size=val_size,
                                        test_size=test_size,
                                        step_size=1,
                                        n_windows=None,
                                        )
        fcst_df.to_csv(results_dir+f'/forecasts.csv', index=False)
        print(f'Cross-validation completed successfully')
    except Exception as e:
        print(f'Cross-validation failed: {e}')
        print("Attempting regular prediction instead...")
        
        # Fallback to regular prediction if cross-validation fails
        try:
            # Fit the model first
            fcst.fit(df=Y_df, static_df=static_df, val_size=val_size)
            
            # Make regular predictions
            fcst_df = fcst.predict(df=Y_df, static_df=static_df)
            fcst_df.to_csv(results_dir+f'/predictions.csv', index=False)
            print(f'Regular prediction completed successfully')
        except Exception as e2:
            print(f'Regular prediction also failed: {e2}')
            # Save error information
            with open(results_dir+f'/error_log.txt', 'w') as f:
                f.write(f"Cross-validation error: {e}\n")
                f.write(f"Regular prediction error: {e2}\n")

    print('Time: ', time.time() - start)
        
def parse_args():
    desc = "Example of Simple TabPFN evaluation"
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