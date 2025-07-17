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
        max_steps: int = 0,
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
        if TabPFNTimeSeriesPredictor is None:
            raise ImportError("TabPFN Time Series not available")
            
        super().__init__(
            h=h,
            input_size=input_size,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=0,
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
        
        # Initialize TabPFN predictor
        self.predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=TabPFNMode.LOCAL,
        )
        
        # Mark as fitted (no training needed)
        self._fitted = True
        
    def fit(self, dataset, val_size=0, test_size=0, random_seed=None):
        """No training needed for TabPFN"""
        self._fitted = True
        return self
        
    def predict(self, dataset, step_size=1, **data_module_kwargs):
        """TabPFN prediction using proper API"""
        
        # Convert dataset to pandas DataFrame
        temporal_df = pd.DataFrame(dataset.temporal, columns=dataset.temporal_cols)
        
        # Convert ds from int to timestamp
        # The stored integer is an offset from min_ds in terms of frequency periods
        temporal_df['ds'] = pd.to_datetime(dataset.min_ds) + pd.to_timedelta(temporal_df['ds'] * dataset.freq_int, unit='ns')
        
        # Rename columns for TabPFN
        temporal_df = temporal_df.rename(columns={'unique_id': 'item_id', 'ds': 'timestamp', 'y': 'target'})
        
        # Split into train and test
        train_df = temporal_df.dropna(subset=['target'])
        test_df = temporal_df[temporal_df['target'].isna()]

        if test_df.empty:
            # When future covariates are not available, we need to generate test_X
            train_tsdf = TimeSeriesDataFrame.from_data_frame(
                train_df, id_column="item_id", timestamp_column="timestamp"
            )
            test_tsdf = generate_test_X(train_tsdf, self.h)
        else:
            # When future covariates are available
            train_tsdf = TimeSeriesDataFrame.from_data_frame(
                train_df, id_column="item_id", timestamp_column="timestamp"
            )
            test_tsdf = TimeSeriesDataFrame.from_data_frame(
                test_df, id_column="item_id", timestamp_column="timestamp"
            )

        # Feature transformation
        # Note: TabPFN-time-series can generate features from the timestamp,
        # but also uses other columns as is.
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
        pred_tsdf = self.predictor.predict(train_tsdf_transformed, test_tsdf_transformed, known_covariates=test_tsdf_transformed)

        # Reshape predictions for NeuralForecast
        predictions = pred_tsdf.unstack(level='item_id')['target'].T
        
        # ensure order is correct
        predictions = predictions.reindex(dataset.uids)
            
        return predictions.values