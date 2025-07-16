import os
import pickle
import time
import argparse
import pandas as pd
import numpy as np
import sys

from .data_parameters import get_data_parameters

from neuralforecast.models.tabpfn import TabPFN
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
    start = time.time()
        
    results_dir = f'{args.results_dir}/{args.dataset}_{args.horizon}/tabpfn/trial_{args.experiment_id}'
    os.makedirs(results_dir, exist_ok = True)

    # Initialize TabPFN model
    # Note: TabPFN does not require hyperparameter tuning like other models.
    tabpfn_model = TabPFN(h=args.horizon,
                          input_size=args.input_size,
                          loss=MAE())

    fcst = NeuralForecast(freq=freq,
                          models=[tabpfn_model])

    print("Starting cross-validation")
    fcst_df = fcst.cross_validation(df=Y_df, 
                                    static_df=static_df,
                                    val_size=val_size,
                                    test_size=test_size,
                                    step_size=1,
                                    n_windows=None,
                                    )
    fcst_df.to_csv(results_dir+f'/forecasts.csv', index=False)

    print('Time: ', time.time() - start)
        
def parse_args():
    desc = "Example of TabPFN evaluation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--results_dir', type=str, default='./results', help='results_dir')
    parser.add_argument('--dataset', type=str, default='ohiot1dm', help='dataset name')
    parser.add_argument('--horizon', type=int, default=12, help='forecast horizon')
    parser.add_argument('--input_size', type=int, default=288, help='input size')
    parser.add_argument('--experiment_id', default=1, required=False, type=int, help='string to identify experiment')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args) 