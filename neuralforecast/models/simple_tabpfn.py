import torch
import numpy as np
import pandas as pd
from typing import Optional, List

from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import MAE

try:
    from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode
    from tabpfn_time_series.data_preparation import to_gluonts_univariate, generate_test_X
    from tabpfn_time_series import FeatureTransformer
    from tabpfn_time_series.features import (
        RunningIndexFeature,
        CalendarFeature,
        AutoSeasonalFeature,
    )
    from autogluon.timeseries import TimeSeriesDataFrame
except ImportError:
    print("TabPFN Time Series not installed. Please install with: pip install tabpfn-time-series")
    TabPFNTimeSeriesPredictor = None

class TabPFN(BaseModel):
    """Simple TabPFN Time Series wrapper for NeuralForecast"""

    def __init__(
        self,
        h: int,
        input_size: int,
        max_context_length: int = 8000,
        stat_exog_list: Optional[List[str]] = None,
        hist_exog_list: Optional[List[str]] = None,
        futr_exog_list: Optional[List[str]] = None,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled: bool = False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        alias: str = None,
        **trainer_kwargs
    ):
            
        super().__init__(
            h=h,
            input_size=input_size,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            **trainer_kwargs
        )
        
        if TabPFNTimeSeriesPredictor is None:
            raise ImportError("TabPFN Time Series not installed. Please install with: pip install tabpfn-time-series")
        
        # TabPFN specific settings
        self.max_context_length = max_context_length
        
        # Lazy initialization of TabPFN predictor
        self._predictor = None
        
    @property
    def predictor(self):
        """Lazy initialization of TabPFN predictor"""
        if self._predictor is None:
            self._predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)
        return self._predictor
        
    def fit(self, dataset, val_size=0, test_size=0, random_seed=None, distributed_config=None):
        """No training needed for TabPFN"""
        self._fitted = True
        return self
    
    def training_step(self, batch, batch_idx):
        """No-op training step for PyTorch Lightning compatibility"""
        return torch.tensor(0.0, requires_grad=True)
    
    def _predict_single_series_window(self, series_data, series_id=0):
        """
        Predict a single time series window with TabPFN
        """
        # Handle context window limitation
        if len(series_data) > self.max_context_length:
            # Use the most recent data points
            series_data = series_data[-self.max_context_length:]
        
        # Skip if too few data points
        if len(series_data) < 5:
            return np.full(self.h, series_data[-1] if len(series_data) > 0 else 0.0)
        
        try:
            # Create DataFrame for TabPFN
            timestamps = pd.date_range(start='2020-01-01', periods=len(series_data), freq='5T')
            
            train_df = pd.DataFrame({
                'item_id': f'series_{series_id}',
                'timestamp': timestamps,
                'target': series_data
            })
            
            # Convert to AutoGluon TimeSeriesDataFrame
            train_tsdf = TimeSeriesDataFrame.from_data_frame(
                train_df, id_column="item_id", timestamp_column="timestamp"
            )
            
            # Generate test periods for prediction
            test_tsdf = generate_test_X(train_tsdf, self.h)
            
            # Feature transformation according to demo code snippet
            selected_features = [
                RunningIndexFeature(),
                CalendarFeature(),
                AutoSeasonalFeature(),
            ]
            
            feature_transformer = FeatureTransformer(selected_features)
            train_tsdf_transformed, test_tsdf_transformed = feature_transformer.transform(
                train_tsdf, test_tsdf
            )

            # Make prediction with TabPFN
            pred_tsdf = self.predictor.predict(
                train_tsdf_transformed, 
                test_tsdf_transformed,
                ignore_pretraining_limits=True  # Always use this to avoid errors
            )
            
            # Extract predictions
            pred_values = pred_tsdf['target'].values
            if len(pred_values) >= self.h:
                return pred_values[:self.h]
            else:
                # Pad with last value if needed
                padded_predictions = np.zeros(self.h)
                padded_predictions[:len(pred_values)] = pred_values
                if len(pred_values) > 0:
                    padded_predictions[len(pred_values):] = pred_values[-1]
                else:
                    padded_predictions[:] = series_data[-1] if len(series_data) > 0 else 0.0
                return padded_predictions
                
        except Exception as e:
            # If TabPFN fails, use simple naive forecast for robustness
            if len(series_data) >= 3:
                naive_forecast = np.mean(series_data[-3:])
            else:
                naive_forecast = series_data[-1] if len(series_data) > 0 else 0.0
            return np.full(self.h, naive_forecast)
    
    def forward(self, windows_batch):
        """
        Forward method that works with PyTorch Lightning's predict_step.
        This is called automatically by the framework for cross-validation.
        """
        # Extract insample data
        insample_y = windows_batch["insample_y"]  # [B, L, 1]
        batch_size = insample_y.shape[0]
        
        # Initialize predictions array
        predictions = torch.zeros(batch_size, self.h, 1, device=insample_y.device, dtype=insample_y.dtype)
        
        # Process each series/window in the batch
        for i in range(batch_size):
            series_data = insample_y[i, :, 0]  # [L]
            
            # Check for valid data
            valid_mask = ~torch.isnan(series_data)
            if not valid_mask.any():
                continue
                
            # Find last valid index and convert to numpy
            valid_indices = valid_mask.nonzero().flatten()
            if len(valid_indices) == 0:
                continue
                
            last_valid_idx = valid_indices[-1].item()
            train_data = series_data[:last_valid_idx+1].detach().cpu().numpy()
            
            # Get predictions
            series_pred = self._predict_single_series_window(train_data, i)
            
            # Convert back to tensor
            pred_tensor = torch.tensor(series_pred, dtype=insample_y.dtype, device=insample_y.device)
            predictions[i, :, 0] = pred_tensor
        
        return predictions