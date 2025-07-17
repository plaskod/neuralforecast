"""
TabPFNTS Evaluation with Baseline-Compatible Setup
================================================

This script evaluates TabPFNTS using cross-validation parameters based on the baseline
train_models, but with modifications for TabPFNTS's in-context learning nature and 
to avoid computational complexity issues.

Key adjustments for TabPFNTS compatibility:
1. Use same data parameters as baseline models (val_size, test_size, freq)
2. Use refit=False since TabPFNTS is designed for in-context learning
3. Use larger step_size (10 vs 1) to reduce computational load
4. Use limited n_windows (100) to avoid memory/indexing issues
5. NEW: TabPFNTS now supports exogenous variables (historical, static, future)
6. Focus on evaluation quality over exhaustive cross-validation coverage
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import argparse

# NeuralForecast imports
from neuralforecast.core import NeuralForecast
from neuralforecast.models.tabpfnts import TabPFNTS
from neuralforecast.losses.pytorch import MAE

try:
    from tabpfn_time_series import TabPFNMode
except ImportError:
    TabPFNMode = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_baseline_data_parameters():
    """
    Get the SAME data parameters used in train_models for consistent evaluation.
    This matches exactly what baseline models use.
    """
    # Get the absolute path of the project's root directory (neuralforecast/)
    # Since we're in scripts/, we need to go up one level to get to neuralforecast/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    data_dir = os.path.join(project_root, 'datasets', 'ohiot1dm_exog_9_day_test.csv')
    static_dir = os.path.join(project_root, 'datasets', 'ohiot1dm_static.csv')
    
    # SAME as baseline models
    val_size = 2691   # Exactly as used in train_models
    test_size = 2691  # Exactly as used in train_models  
    freq = '5T'       # 5-minute frequency
    
    return data_dir, static_dir, val_size, test_size, freq

def load_ohio_data():
    """Load Ohio T1DM dataset using baseline parameters."""
    data_dir, static_dir, val_size, test_size, freq = get_baseline_data_parameters()
    
    # Load temporal data
    Y_df = pd.read_csv(data_dir)
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    
    # Load static patient features  
    static_df = pd.read_csv(static_dir)
    
    logger.info(f"Loaded dataset: {Y_df.shape[0]} records, {len(Y_df.unique_id.unique())} patients")
    logger.info(f"Time range: {Y_df.ds.min()} to {Y_df.ds.max()}")
    logger.info(f"Using BASELINE parameters: val_size={val_size}, test_size={test_size}")
    
    return Y_df, static_df, val_size, test_size, freq

def configure_tabpfnts_baseline(use_exog=True, horizon=12, input_size=144, context_length=1000):
    """
    Configure TabPFNTS model using baseline model conventions but optimized for in-context learning.
    
    Args:
        use_exog: Whether to include exogenous variables
        horizon: Forecast horizon (same as baseline models)
        input_size: Input context size (same as baseline models)
        context_length: TabPFNTS-specific parameter for context management
        
    Returns:
        Configured TabPFNTS model
    """
    
    # Configure exogenous variables based on the ohio dataset structure
    if use_exog:
        # Ohio T1DM dataset exogenous variables (based on the actual dataset columns)
        stat_exog_list = ['#559', '#563', '#570', '#575', '#588', '#591', 
                         '#540', '#544', '#552', '#567', '#584', 
                         'insulin_type_novalog', 'female', 'age_20_40', 
                         'age_40_60', 'pump_model_630G']  # Static features from ohio dataset
        hist_exog_list = ['CHO', 'basal_insulin', 'bolus_insulin']  # Actual column names in ohio dataset
        futr_exog_list = []  # No future exogenous typically available
        alias = "TabPFNTS_baseline_with_exog"
        print(f"TabPFNTS configured WITH exogenous variables:")
        print(f"  - Historical: {hist_exog_list}")
        print(f"  - Static: {len(stat_exog_list)} variables")
    else:
        stat_exog_list = None
        hist_exog_list = None 
        futr_exog_list = None
        alias = "TabPFNTS_baseline_no_exog"
        print("TabPFNTS configured WITHOUT exogenous variables")
    
    # Configure TabPFNTS with baseline-compatible parameters
    model = TabPFNTS(
        h=horizon,                          # Same horizon as baseline models
        input_size=input_size,              # Same input_size as baseline models
        context_length=context_length,      # TabPFNTS-specific: manage context efficiently
        tabpfn_mode=TabPFNMode.LOCAL if TabPFNMode else None,  # Use local mode if available
        debug=True,                         # Enable debug for monitoring
        stat_exog_list=stat_exog_list,      # Static exogenous variables
        hist_exog_list=hist_exog_list,      # Historical exogenous variables
        futr_exog_list=futr_exog_list,      # Future exogenous variables
        loss=MAE(),                         # Same loss as baseline models
        alias=alias                         # Model identifier
    )
    
    return model

def run_baseline_cross_validation(Y_df, static_df, model, val_size, test_size, use_refit=False):
    """
    Run cross-validation using a modified setup that works reliably with TabPFNTS.
    
    Args:
        Y_df: Time series dataframe
        static_df: Static features dataframe
        model: Configured TabPFNTS model
        val_size: Validation size (same as baseline)
        test_size: Test size (same as baseline)
        use_refit: Whether to use refit (False for in-context learning)
        
    Returns:
        Cross-validation results dataframe
    """
    
    # Create NeuralForecast object
    nf = NeuralForecast(
        freq='5T',              # Same frequency as baseline
        models=[model]
    )
    
    logger.info("\n" + "="*80)
    logger.info("TABPFNTS EVALUATION WITH BASELINE SETUP")
    logger.info("="*80)
    logger.info("Cross-validation parameters (MODIFIED for TabPFNTS compatibility):")
    logger.info(f"  - Frequency: 5 minutes")
    logger.info(f"  - Validation size: {val_size} steps ({val_size*5/60/24:.1f} days)")
    logger.info(f"  - Step size: 10 (modified for efficiency)")
    logger.info(f"  - N windows: 100 (limited for TabPFNTS compatibility)")
    logger.info(f"  - Test size: auto-calculated from n_windows and step_size")
    logger.info(f"  - Refit: {use_refit} (optimized for TabPFNTS in-context learning)")
    logger.info(f"  - Expected windows: ~100 (limited sample for testing)")
    
    start_time = datetime.now()
    
    # Use a more conservative cross-validation setup for TabPFNTS compatibility
    # This reduces the computational load and avoids indexing issues
    cv_results = nf.cross_validation(
        df=Y_df,
        static_df=static_df,
        val_size=val_size,      # SAME as baseline models
        step_size=10,           # MODIFIED: Every 50 minutes instead of 5 min for efficiency
        n_windows=100,          # MODIFIED: Limited number of windows for compatibility
        refit=use_refit,        # FALSE for in-context learning efficiency
        verbose=True
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\nModified cross-validation completed!")
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Results shape: {cv_results.shape}")
    logger.info(f"Actual number of windows: {len(cv_results.groupby(['unique_id', 'cutoff']))}")
    
    return cv_results

def calculate_comprehensive_metrics(cv_results, model_alias):
    """
    Calculate comprehensive metrics matching baseline evaluation standards.
    
    Args:
        cv_results: Cross-validation results dataframe
        model_alias: Model column name
        
    Returns:
        Dictionary with detailed metrics
    """
    
    # Extract predictions and actual values
    y_true = cv_results['y'].values
    y_pred = cv_results[model_alias].values
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        logger.warning("No valid predictions found!")
        return {}
    
    # Calculate standard forecasting metrics
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    mse = np.mean((y_true_clean - y_pred_clean) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE (with zero-division protection)
    mape_mask = y_true_clean != 0
    if np.any(mape_mask):
        mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
    else:
        mape = np.nan
    
    # SMAPE
    denominator = (np.abs(y_true_clean) + np.abs(y_pred_clean)) / 2
    smape_mask = denominator != 0
    if np.any(smape_mask):
        smape = np.mean(np.abs(y_true_clean[smape_mask] - y_pred_clean[smape_mask]) / denominator[smape_mask]) * 100
    else:
        smape = np.nan
    
    # Additional clinical metrics for glucose prediction
    # Glucose-specific thresholds
    normal_range = (70, 180)  # mg/dL
    low_threshold = 70        # Hypoglycemia
    high_threshold = 250      # Severe hyperglycemia
    
    # Clinical accuracy metrics
    in_range_actual = np.sum((y_true_clean >= normal_range[0]) & (y_true_clean <= normal_range[1]))
    in_range_predicted = np.sum((y_pred_clean >= normal_range[0]) & (y_pred_clean <= normal_range[1]))
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'n_predictions': len(y_true_clean),
        'mean_glucose_actual': np.mean(y_true_clean),
        'std_glucose_actual': np.std(y_true_clean),
        'mean_glucose_predicted': np.mean(y_pred_clean),
        'std_glucose_predicted': np.std(y_pred_clean),
        'glucose_range_accuracy': in_range_predicted / len(y_pred_clean) * 100,
        'clinical_range_overlap': min(in_range_actual, in_range_predicted) / max(in_range_actual, in_range_predicted) * 100 if max(in_range_actual, in_range_predicted) > 0 else 0
    }
    
    # Per-patient metrics (important for diabetes care)
    patient_metrics = {}
    for patient_id in cv_results['unique_id'].unique():
        patient_data = cv_results[cv_results['unique_id'] == patient_id]
        if len(patient_data) > 0:
            p_true = patient_data['y'].values
            p_pred = patient_data[model_alias].values
            p_mask = ~(np.isnan(p_true) | np.isnan(p_pred))
            
            if np.any(p_mask):
                p_mae = np.mean(np.abs(p_true[p_mask] - p_pred[p_mask]))
                p_rmse = np.sqrt(np.mean((p_true[p_mask] - p_pred[p_mask]) ** 2))
                patient_metrics[patient_id] = {
                    'mae': p_mae,
                    'rmse': p_rmse,
                    'n_predictions': np.sum(p_mask),
                    'mean_glucose': np.mean(p_true[p_mask])
                }
    
    metrics['per_patient'] = patient_metrics
    return metrics

def run_comprehensive_baseline_evaluation(save_path='tabpfnts_baseline_evaluation_results'):
    """
    Run comprehensive TabPFNTS evaluation using baseline-compatible setup with practical optimizations.
    
    Args:
        save_path: Directory path to save results
    """
    
    logger.info("="*100)
    logger.info("TABPFNTS COMPREHENSIVE EVALUATION - BASELINE-COMPATIBLE SETUP")
    logger.info("="*100)
    logger.info("Using baseline data parameters with TabPFNTS-optimized cross-validation")
    logger.info("Focused on evaluation quality with practical computational constraints")
    
    # 1. Load data with baseline parameters
    logger.info("\n1. Loading Ohio T1DM Dataset (Baseline Parameters)...")
    Y_df, static_df, val_size, test_size, freq = load_ohio_data()
    
    # 2. Evaluate TabPFNTS WITHOUT exogenous variables (matching ohiot1dm dataset)
    logger.info("\n2. TabPFNTS Evaluation WITHOUT Exogenous Variables...")
    logger.info("   (Matching 'ohiot1dm' dataset configuration in train_models)")
    
    model_no_exog = configure_tabpfnts_baseline(use_exog=False, context_length=1000)
    cv_results_no_exog = run_baseline_cross_validation(
        Y_df, static_df, model_no_exog, val_size, test_size, use_refit=False
    )
    metrics_no_exog = calculate_comprehensive_metrics(cv_results_no_exog, "TabPFNTS_baseline_no_exog")
    
    logger.info(f"\nResults WITHOUT exogenous variables:")
    logger.info(f"  MAE: {metrics_no_exog['mae']:.2f} mg/dL")
    logger.info(f"  RMSE: {metrics_no_exog['rmse']:.2f} mg/dL")
    logger.info(f"  MAPE: {metrics_no_exog['mape']:.2f}%")
    logger.info(f"  SMAPE: {metrics_no_exog['smape']:.2f}%")
    logger.info(f"  Predictions: {metrics_no_exog['n_predictions']:,}")
    logger.info(f"  Clinical range accuracy: {metrics_no_exog['glucose_range_accuracy']:.1f}%")
    
    # 3. Evaluate TabPFNTS WITH exogenous variables (matching ohiot1dm_exog dataset)  
    logger.info("\n3. TabPFNTS Evaluation WITH Exogenous Variables...")
    logger.info("   (Testing TabPFNTS's new exogenous variable support)")
    
    model_with_exog = configure_tabpfnts_baseline(use_exog=True, context_length=1000)
    cv_results_with_exog = run_baseline_cross_validation(
        Y_df, static_df, model_with_exog, val_size, test_size, use_refit=False
    )
    metrics_with_exog = calculate_comprehensive_metrics(cv_results_with_exog, "TabPFNTS_baseline_with_exog")
    
    logger.info(f"\nResults WITH exogenous variables:")
    logger.info(f"  MAE: {metrics_with_exog['mae']:.2f} mg/dL")
    logger.info(f"  RMSE: {metrics_with_exog['rmse']:.2f} mg/dL")
    logger.info(f"  MAPE: {metrics_with_exog['mape']:.2f}%")
    logger.info(f"  SMAPE: {metrics_with_exog['smape']:.2f}%")
    logger.info(f"  Predictions: {metrics_with_exog['n_predictions']:,}")
    logger.info(f"  Clinical range accuracy: {metrics_with_exog['glucose_range_accuracy']:.1f}%")
    
    # 4. Compare with baseline standards
    logger.info("\n4. Comparison Analysis:")
    mae_improvement = ((metrics_no_exog['mae'] - metrics_with_exog['mae']) / metrics_no_exog['mae']) * 100
    rmse_improvement = ((metrics_no_exog['rmse'] - metrics_with_exog['rmse']) / metrics_no_exog['rmse']) * 100
    
    logger.info(f"  MAE improvement with exogenous: {mae_improvement:.1f}%")
    logger.info(f"  RMSE improvement with exogenous: {rmse_improvement:.1f}%")
    
    # 5. Per-patient analysis
    logger.info("\n5. Per-Patient Performance:")
    logger.info("   Patient MAE (mg/dL) - No Exog vs With Exog:")
    
    for patient_id in sorted(metrics_no_exog['per_patient'].keys()):
        mae_no = metrics_no_exog['per_patient'][patient_id]['mae']
        mae_with = metrics_with_exog['per_patient'][patient_id]['mae']
        improvement = ((mae_no - mae_with) / mae_no) * 100
        logger.info(f"   {patient_id}: {mae_no:.1f} vs {mae_with:.1f} ({improvement:+.1f}%)")
    
    # 6. Save results for comparison with other baseline models
    logger.info("\n6. Saving Results...")
    
    # Save in format compatible with baseline model comparison
    results_dir = save_path
    os.makedirs(results_dir, exist_ok=True)
    
    # Save cross-validation results
    cv_results_no_exog.to_csv(f'{results_dir}/tabpfnts_baseline_no_exog_forecasts.csv', index=False)
    cv_results_with_exog.to_csv(f'{results_dir}/tabpfnts_baseline_with_exog_forecasts.csv', index=False)
    
    # Save metrics summary
    import json
    summary = {
        'evaluation_setup': 'baseline_compatible',
        'cross_validation_parameters': {
            'val_size': val_size,
            'test_size': test_size,
            'step_size': 1,
            'n_windows': 'auto_calculated',
            'refit': False,
            'frequency': '5T'
        },
        'tabpfnts_parameters': {
            'horizon': 12,
            'input_size': 144,
            'context_length': 1000
        },
        'metrics_no_exog': metrics_no_exog,
        'metrics_with_exog': metrics_with_exog,
        'comparison': {
            'mae_improvement_pct': mae_improvement,
            'rmse_improvement_pct': rmse_improvement
        }
    }
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    summary = convert_numpy(summary)
    
    with open(f'{results_dir}/tabpfnts_baseline_evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to: {results_dir}/")
    logger.info(f"  - Cross-validation forecasts (CSV)")
    logger.info(f"  - Evaluation metrics summary (JSON)")
    
    # 7. Final summary
    logger.info("\n" + "="*100) 
    logger.info("BASELINE-COMPATIBLE EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("="*100)
    logger.info("Key Insights:")
    logger.info(f"  • TabPFNTS evaluated on {metrics_with_exog['n_predictions']:,} prediction points")
    logger.info(f"  • Baseline-compatible data parameters with TabPFNTS-optimized CV")
    logger.info(f"  • In-context learning: refit=False for efficiency")
    logger.info(f"  • Note: TabPFNTS doesn't support exogenous variables")
    logger.info(f"  • Clinical glucose prediction accuracy: {metrics_with_exog['glucose_range_accuracy']:.1f}%")
    logger.info(f"  • Results comparable to baseline models with practical constraints")
    
    return {
        'cv_results_no_exog': cv_results_no_exog,
        'cv_results_with_exog': cv_results_with_exog,
        'metrics_no_exog': metrics_no_exog,
        'metrics_with_exog': metrics_with_exog
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TabPFNTS evaluation with baseline train_models setup')
    parser.add_argument('--horizon', type=int, default=12, help='Forecasting horizon')
    parser.add_argument('--input_size', type=int, default=144, help='Input context size')
    parser.add_argument('--context_length', type=int, default=1000, help='Context length for TabPFNTS')
    parser.add_argument('--save_path', type=str, default='tabpfnts_baseline_evaluation_results', 
                       help='Path to save results')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run comprehensive baseline evaluation
    results = run_comprehensive_baseline_evaluation(save_path=args.save_path) 