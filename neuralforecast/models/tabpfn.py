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
    """TabPFN Time Series wrapper for NeuralForecast"""

    def __init__(
        self,
        h: int,
        input_size: int,
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
        
        # Initialize TabPFN predictor
        self.predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=TabPFNMode.LOCAL,
        )

        print(type(self.predictor))
        
        # Cache for predictions to avoid re-computation
        self._prediction_cache = {}
        
    def fit(self, dataset, val_size=0, test_size=0, random_seed=None, distributed_config=None):
        """No training needed for TabPFN"""
        self._fitted = True
        return self
    
    def training_step(self, batch, batch_idx):
        """No-op training step for PyTorch Lightning compatibility"""
        # Return a dummy loss to satisfy PyTorch Lightning
        return torch.tensor(0.0, requires_grad=True)
    
    def predict(self, dataset, step_size=1, **data_module_kwargs):
        """
        Custom predict method that works directly with the dataset 
        to bypass PyTorch Lightning issues.
        """
        # DEBUG: Print dataset information
        print(f"TabPFN predict: dataset.n_groups={dataset.n_groups}")
        print(f"TabPFN predict: dataset.indptr={dataset.indptr}")
        print(f"TabPFN predict: dataset.temporal.shape={dataset.temporal.shape}")
        
        # Extract data from the dataset
        temporal_data = dataset.temporal.numpy()  # [total_time_points, features]
        indptr = dataset.indptr  # [n_series + 1] - indices for each series
        n_series = dataset.n_groups
        
        # Calculate number of prediction windows per series
        predictions_list = []
        
        for i in range(n_series):
            # Get this series data
            start_idx = indptr[i]
            end_idx = indptr[i + 1]
            series_temporal = temporal_data[start_idx:end_idx]  # [series_length, features]
            
            print(f"Series {i}: length={len(series_temporal)}, start_idx={start_idx}, end_idx={end_idx}")
            
            # Extract target values (assuming y is at index 0)
            y_values = series_temporal[:, 0]
            
            # Remove NaN values
            valid_mask = ~np.isnan(y_values)
            if not valid_mask.any():
                # No valid data, use zeros
                series_predictions = np.zeros(self.h)
                predictions_list.append(series_predictions.reshape(-1, 1))
                continue
            
            # Get valid data
            valid_indices = np.where(valid_mask)[0]
            last_valid_idx = valid_indices[-1]
            valid_y = y_values[:last_valid_idx + 1]
            
            print(f"Series {i}: valid_y length={len(valid_y)}")
            
            # Only proceed if we have enough data
            if len(valid_y) < 2:
                # Use simple naive forecast
                if len(valid_y) > 0:
                    naive_forecast = valid_y[-1]
                else:
                    naive_forecast = 0.0
                series_predictions = np.full(self.h, naive_forecast)
                predictions_list.append(series_predictions.reshape(-1, 1))
                continue
            
            try:
                # Create DataFrame for TabPFN
                timestamps = pd.date_range(start='2020-01-01', periods=len(valid_y), freq='5T')
                
                train_df = pd.DataFrame({
                    'item_id': f'series_{i}',
                    'timestamp': timestamps,
                    'target': valid_y
                })
                
                # Convert to AutoGluon TimeSeriesDataFrame
                train_tsdf = TimeSeriesDataFrame.from_data_frame(
                    train_df, id_column="item_id", timestamp_column="timestamp"
                )
                
                # Generate test periods for prediction
                test_tsdf = generate_test_X(train_tsdf, self.h)
                
                # Feature transformation
                selected_features = [
                    RunningIndexFeature(),
                    CalendarFeature(),
                    AutoSeasonalFeature(),
                ]
                
                feature_transformer = FeatureTransformer(selected_features)
                train_tsdf_transformed, test_tsdf_transformed = feature_transformer.transform(
                    train_tsdf, test_tsdf
                )

                # Make prediction
                pred_tsdf = self.predictor.predict(
                    train_tsdf_transformed, test_tsdf_transformed
                )
                
                # Extract predictions
                pred_values = pred_tsdf['target'].values
                if len(pred_values) >= self.h:
                    series_predictions = pred_values[:self.h]
                else:
                    # Pad with last value if needed
                    series_predictions = np.zeros(self.h)
                    series_predictions[:len(pred_values)] = pred_values
                    if len(pred_values) > 0:
                        series_predictions[len(pred_values):] = pred_values[-1]
                    else:
                        series_predictions[:] = valid_y[-1] if len(valid_y) > 0 else 0.0
                
                predictions_list.append(series_predictions.reshape(-1, 1))
                
            except Exception as e:
                print(f"TabPFN failed for series {i}: {e}")
                # If TabPFN fails, use simple naive forecast
                if len(valid_y) >= 5:
                    naive_forecast = np.mean(valid_y[-5:])
                elif len(valid_y) > 0:
                    naive_forecast = np.mean(valid_y)
                else:
                    naive_forecast = 0.0
                series_predictions = np.full(self.h, naive_forecast)
                predictions_list.append(series_predictions.reshape(-1, 1))
        
        # Stack all predictions: [n_series * h, 1]
        all_predictions = np.vstack(predictions_list)
        print(f"TabPFN predict: returning shape {all_predictions.shape}")
        return all_predictions
        
    def forward(self, windows_batch):
        """
        TabPFN forward method that follows BaseModel pattern.
        This is kept for compatibility but may not be used in cross-validation.
        """
        # Extract insample data
        insample_y = windows_batch["insample_y"]  # [B, L, 1]
        batch_size = insample_y.shape[0]
        
        # Initialize predictions array
        predictions = torch.zeros(batch_size, self.h, 1, device=insample_y.device, dtype=insample_y.dtype)
        
        # Process each series in the batch
        for i in range(batch_size):
            series_data = insample_y[i, :, 0]  # [L]
            
            # Check for valid data
            valid_mask = ~torch.isnan(series_data)
            if not valid_mask.any():
                # No valid data, use zeros (will be handled by loss function)
                continue
                
            # Find last valid index
            valid_indices = valid_mask.nonzero().flatten()
            if len(valid_indices) == 0:
                continue
                
            last_valid_idx = valid_indices[-1].item()
            train_data = series_data[:last_valid_idx+1]
            
            # Only proceed if we have enough data
            if len(train_data) < 2:
                # Use simple naive forecast
                if len(train_data) > 0:
                    naive_forecast = train_data[-1].item()
                else:
                    naive_forecast = 0.0
                predictions[i, :, 0] = naive_forecast
                continue
                
            try:
                # Convert to pandas DataFrame with proper timestamps
                timestamps = pd.date_range(start='2020-01-01', periods=len(train_data), freq='5T')
                
                train_df = pd.DataFrame({
                    'item_id': f'series_{i}',
                    'timestamp': timestamps,
                    'target': train_data.detach().cpu().numpy()
                })
                
                # Convert to AutoGluon TimeSeriesDataFrame
                train_tsdf = TimeSeriesDataFrame.from_data_frame(
                    train_df, id_column="item_id", timestamp_column="timestamp"
                )
                
                # Generate test periods for prediction
                test_tsdf = generate_test_X(train_tsdf, self.h)
                
                # Feature transformation
                selected_features = [
                    RunningIndexFeature(),
                    CalendarFeature(),
                    AutoSeasonalFeature(),
                ]
                
                feature_transformer = FeatureTransformer(selected_features)
                train_tsdf_transformed, test_tsdf_transformed = feature_transformer.transform(
                    train_tsdf, test_tsdf
                )

                # Make prediction
                pred_tsdf = self.predictor.predict(
                    train_tsdf_transformed, test_tsdf_transformed
                )
                
                # Extract predictions and convert back to tensor
                pred_values = pred_tsdf['target'].values
                if len(pred_values) >= self.h:
                    pred_tensor = torch.tensor(pred_values[:self.h], dtype=insample_y.dtype, device=insample_y.device)
                    predictions[i, :, 0] = pred_tensor
                else:
                    # Pad with last value if needed
                    pred_tensor = torch.tensor(pred_values, dtype=insample_y.dtype, device=insample_y.device)
                    predictions[i, :len(pred_values), 0] = pred_tensor
                    if len(pred_values) > 0:
                        predictions[i, len(pred_values):, 0] = pred_values[-1]
                
            except Exception as e:
                # If TabPFN fails, use simple naive forecast
                if len(train_data) >= 5:
                    naive_forecast = torch.mean(train_data[-5:]).item()
                elif len(train_data) > 0:
                    naive_forecast = torch.mean(train_data).item()
                else:
                    naive_forecast = 0.0
                predictions[i, :, 0] = naive_forecast
        
        return predictions