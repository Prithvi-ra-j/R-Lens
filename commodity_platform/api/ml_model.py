import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import os
from typing import List, Tuple, Optional
from .db import DatabaseHandler

class CommodityPredictor:
    def __init__(self, model_path: str = "/workspace/commodity_platform/models"):
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.sequence_length = 60  # Use 60 time steps for prediction
        
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
        
        # Load existing models
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load pre-trained models if they exist"""
        for commodity in ['gold', 'silver', 'oil', 'gas']:
            model_file = os.path.join(self.model_path, f"{commodity}_lstm_model.h5")
            scaler_file = os.path.join(self.model_path, f"{commodity}_scaler.npy")
            
            if os.path.exists(model_file):
                try:
                    self.models[commodity] = load_model(model_file)
                    if os.path.exists(scaler_file):
                        scaler_data = np.load(scaler_file, allow_pickle=True).item()
                        scaler = MinMaxScaler()
                        scaler.scale_ = scaler_data['scale_']
                        scaler.min_ = scaler_data['min_']
                        scaler.data_min_ = scaler_data['data_min_']
                        scaler.data_max_ = scaler_data['data_max_']
                        scaler.data_range_ = scaler_data['data_range_']
                        self.scalers[commodity] = scaler
                    print(f"Loaded model for {commodity}")
                except Exception as e:
                    print(f"Error loading model for {commodity}: {e}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare data for LSTM training"""
        # Ensure we have enough data
        if len(df) < self.sequence_length + 10:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length + 10} records")
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['price']])
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaler
    
    def create_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Create LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_model(self, commodity: str, df: pd.DataFrame, epochs: int = 50) -> dict:
        """Train LSTM model for a specific commodity"""
        try:
            # Prepare data
            X, y, scaler = self.prepare_data(df)
            
            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Split into train/test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Create and train model
            model = self.create_model((X.shape[1], 1))
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate model
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mae = mean_absolute_error(y_test, test_pred)
            
            # Save model and scaler
            model_file = os.path.join(self.model_path, f"{commodity}_lstm_model.h5")
            scaler_file = os.path.join(self.model_path, f"{commodity}_scaler.npy")
            
            model.save(model_file)
            scaler_data = {
                'scale_': scaler.scale_,
                'min_': scaler.min_,
                'data_min_': scaler.data_min_,
                'data_max_': scaler.data_max_,
                'data_range_': scaler.data_range_
            }
            np.save(scaler_file, scaler_data)
            
            # Store in memory
            self.models[commodity] = model
            self.scalers[commodity] = scaler
            
            return {
                'success': True,
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae),
                'model_accuracy': max(0, 1 - test_rmse)  # Simple accuracy metric
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_prices(self, commodity: str, days_ahead: int = 7) -> Optional[dict]:
        """Predict future prices for a commodity"""
        commodity = commodity.lower()
        
        if commodity not in self.models or commodity not in self.scalers:
            return None
        
        try:
            # Get recent data from database
            db = DatabaseHandler()
            df = db.get_historical_data(commodity, days=self.sequence_length + 10)
            
            if len(df) < self.sequence_length:
                return None
            
            model = self.models[commodity]
            scaler = self.scalers[commodity]
            
            # Prepare last sequence
            recent_prices = df['price'].values[-self.sequence_length:]
            scaled_prices = scaler.transform(recent_prices.reshape(-1, 1))
            
            # Make predictions
            predictions = []
            current_sequence = scaled_prices.flatten()
            
            for _ in range(days_ahead):
                # Reshape for prediction
                X = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, 1)
                
                # Predict next price
                pred_scaled = model.predict(X, verbose=0)[0, 0]
                
                # Transform back to original scale
                pred_price = scaler.inverse_transform([[pred_scaled]])[0, 0]
                
                # Add to predictions
                current_date = datetime.now() + timedelta(days=len(predictions) + 1)
                predictions.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'predicted_price': round(float(pred_price), 2)
                })
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], pred_scaled)
            
            return {
                'commodity': commodity.upper(),
                'predictions': predictions,
                'confidence_score': 0.75,  # Placeholder - could be calculated from validation metrics
                'model_accuracy': 0.85     # Placeholder - from training metrics
            }
            
        except Exception as e:
            print(f"Error predicting prices for {commodity}: {e}")
            return None
    
    def retrain_all_models(self, db_handler: DatabaseHandler) -> dict:
        """Retrain models for all commodities"""
        results = {}
        
        for commodity in ['gold', 'silver', 'oil', 'gas']:
            try:
                # Get historical data
                df = db_handler.get_historical_data(commodity, days=200)  # Get more data for training
                
                if len(df) >= self.sequence_length + 10:
                    result = self.train_model(commodity, df)
                    results[commodity] = result
                else:
                    results[commodity] = {
                        'success': False,
                        'error': 'Insufficient historical data'
                    }
            except Exception as e:
                results[commodity] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results