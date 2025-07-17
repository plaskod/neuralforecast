#!/usr/bin/env python3
"""
Quick test for TabPFN cross-validation on Ohio dataset
Tests both with and without exogenous variables using a small subset
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('.')
sys.path.append('./train_models')

try:
    from neuralforecast.models.simple_tabpfn import TabPFN
    from neuralforecast.core import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    print("✅ Successfully imported neuralforecast modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_quick_cv():
    """Quick test of TabPFN cross-validation"""
    
    print("🔄 Loading Ohio T1DM dataset...")
    
    # Load data
    try:
        data_path = 'datasets/ohiot1dm_exog_9_day_test.csv'
        static_path = 'datasets/ohiot1dm_static.csv'
        
        Y_df = pd.read_csv(data_path)
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        static_df = pd.read_csv(static_path)
        
        print(f"✅ Loaded dataset: {Y_df.shape} rows, {Y_df['unique_id'].nunique()} patients")
        print(f"   Date range: {Y_df['ds'].min()} to {Y_df['ds'].max()}")
        print(f"   Columns: {list(Y_df.columns)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Test with subset of data (first 2 patients, smaller window)
    print("\n🔄 Preparing test subset...")
    unique_ids = Y_df['unique_id'].unique()[:2]  # First 2 patients
    Y_df_subset = Y_df[Y_df['unique_id'].isin(unique_ids)].copy()
    
    # Take first 1000 rows per patient for faster testing
    Y_df_test = []
    for uid in unique_ids:
        patient_data = Y_df_subset[Y_df_subset['unique_id'] == uid].head(1000)
        Y_df_test.append(patient_data)
    Y_df_test = pd.concat(Y_df_test, ignore_index=True)
    
    print(f"✅ Test subset: {Y_df_test.shape} rows, {Y_df_test['unique_id'].nunique()} patients")
    
    # Test 1: Without exogenous variables
    print("\n🧪 Test 1: TabPFN WITHOUT exogenous variables")
    try:
        Y_df_no_exog = Y_df_test[['unique_id', 'ds', 'y', 'available_mask']].copy()
        
        tabpfn_no_exog = TabPFN(
            h=6,  # Smaller horizon for quick test
            input_size=72,  # Smaller input size 
            max_context_length=500,  # Smaller context for quick test
            stat_exog_list=['#559', '#563', '#570', '#575', '#588', '#591', '#540', '#544', '#552', '#567', '#584'],
            loss=MAE(),
            alias="TabPFN_no_exog"
        )
        
        fcst_no_exog = NeuralForecast(freq='5T', models=[tabpfn_no_exog])
        
        cv_results_no_exog = fcst_no_exog.cross_validation(
            df=Y_df_no_exog,
            static_df=static_df,
            n_windows=2,  # Just 2 windows for quick test
            step_size=6,
            val_size=50,  # Smaller validation set
            refit=True,
            verbose=False
        )
        
        print(f"✅ CV without exog completed: {cv_results_no_exog.shape}")
        print(f"   Columns: {list(cv_results_no_exog.columns)}")
        
        # Check if predictions exist
        if 'TabPFN_no_exog' in cv_results_no_exog.columns:
            preds = cv_results_no_exog['TabPFN_no_exog']
            print(f"   Predictions range: {preds.min():.2f} to {preds.max():.2f}")
            print(f"   Non-null predictions: {preds.count()}/{len(preds)}")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: With exogenous variables
    print("\n🧪 Test 2: TabPFN WITH exogenous variables")
    try:
        tabpfn_with_exog = TabPFN(
            h=6,  # Smaller horizon for quick test
            input_size=72,  # Smaller input size
            max_context_length=500,  # Smaller context for quick test
            stat_exog_list=['#559', '#563', '#570', '#575', '#588', '#591', '#540', '#544', '#552', '#567', '#584',
                           'insulin_type_novalog', 'female', 'age_20_40', 'age_40_60', 'pump_model_630G'],
            hist_exog_list=['CHO', 'basal_insulin', 'bolus_insulin'],
            loss=MAE(),
            alias="TabPFN_with_exog"
        )
        
        fcst_with_exog = NeuralForecast(freq='5T', models=[tabpfn_with_exog])
        
        cv_results_with_exog = fcst_with_exog.cross_validation(
            df=Y_df_test,  # Full exog data
            static_df=static_df,
            n_windows=2,  # Just 2 windows for quick test
            step_size=6,
            val_size=50,  # Smaller validation set
            refit=True,
            verbose=False
        )
        
        print(f"✅ CV with exog completed: {cv_results_with_exog.shape}")
        print(f"   Columns: {list(cv_results_with_exog.columns)}")
        
        # Check if predictions exist
        if 'TabPFN_with_exog' in cv_results_with_exog.columns:
            preds = cv_results_with_exog['TabPFN_with_exog']
            print(f"   Predictions range: {preds.min():.2f} to {preds.max():.2f}")
            print(f"   Non-null predictions: {preds.count()}/{len(preds)}")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Quick comparison
    print("\n📊 Quick comparison:")
    try:
        if 'TabPFN_no_exog' in cv_results_no_exog.columns and 'TabPFN_with_exog' in cv_results_with_exog.columns:
            # Calculate simple MAE for comparison
            y_true_no_exog = cv_results_no_exog['y'].values
            y_pred_no_exog = cv_results_no_exog['TabPFN_no_exog'].values
            mae_no_exog = np.mean(np.abs(y_true_no_exog - y_pred_no_exog))
            
            y_true_with_exog = cv_results_with_exog['y'].values
            y_pred_with_exog = cv_results_with_exog['TabPFN_with_exog'].values
            mae_with_exog = np.mean(np.abs(y_true_with_exog - y_pred_with_exog))
            
            print(f"   MAE without exog: {mae_no_exog:.4f}")
            print(f"   MAE with exog: {mae_with_exog:.4f}")
            
            improvement = ((mae_no_exog - mae_with_exog) / mae_no_exog) * 100
            direction = "↓ BETTER" if improvement > 0 else "↑ WORSE"
            print(f"   Improvement: {improvement:+.2f}% {direction}")
            
    except Exception as e:
        print(f"⚠️  Could not calculate comparison metrics: {e}")
    
    return True

def main():
    """Main test function"""
    print("="*60)
    print("QUICK TABPFN CROSS-VALIDATION TEST")
    print("Ohio T1DM Dataset - Subset Test")
    print("="*60)
    
    success = test_quick_cv()
    
    if success:
        print("\n🎉 Quick test completed successfully!")
        print("✅ TabPFN cross-validation is working")
        print("✅ Ready to run full evaluation")
    else:
        print("\n💥 Quick test failed!")
        print("❌ Please check the setup and try again")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 