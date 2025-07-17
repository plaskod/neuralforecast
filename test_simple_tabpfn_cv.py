#!/usr/bin/env python3
"""
Quick test script for simple_tabpfn cross-validation
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
from neuralforecast.models.simple_tabpfn import TabPFN
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE

def test_simple_tabpfn_cv():
    print("Testing Simple TabPFN Cross-Validation...")
    
    # Create a small test dataset
    n_series = 3
    n_points = 1000
    horizon = 12
    
    # Generate test data
    data = []
    for i in range(n_series):
        dates = pd.date_range(start='2020-01-01', periods=n_points, freq='5T')
        # Simple trend + noise
        trend = np.linspace(10 + i*5, 50 + i*5, n_points)
        noise = np.random.normal(0, 2, n_points)
        values = trend + noise
        
        series_data = pd.DataFrame({
            'unique_id': f'series_{i}',
            'ds': dates,
            'y': values
        })
        data.append(series_data)
    
    Y_df = pd.concat(data, ignore_index=True)
    print(f"Created test dataset with {n_series} series, {n_points} points each")
    
    # Initialize TabPFN with small context for quick testing
    tabpfn_model = TabPFN(
        h=horizon,
        input_size=144,  # Smaller input size for faster testing
        max_context_length=500,  # Much smaller context for quick testing
        loss=MAE()
    )
    
    fcst = NeuralForecast(freq='5T', models=[tabpfn_model])
    
    print("Starting cross-validation test...")
    
    try:
        # Test cross-validation
        fcst_df = fcst.cross_validation(
            df=Y_df,
            val_size=100,
            test_size=100,
            step_size=1,
            n_windows=None,
        )
        
        print(f"✅ Cross-validation successful!")
        print(f"   Output shape: {fcst_df.shape}")
        print(f"   Columns: {list(fcst_df.columns)}")
        
        # Check if we have the expected structure
        expected_models = ['TabPFN']
        for model in expected_models:
            if model in fcst_df.columns:
                print(f"   ✅ {model} predictions found")
            else:
                print(f"   ❌ {model} predictions missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_tabpfn_cv()
    if success:
        print("\n🎉 Simple TabPFN cross-validation test passed!")
    else:
        print("\n💥 Simple TabPFN cross-validation test failed!")
        sys.exit(1) 