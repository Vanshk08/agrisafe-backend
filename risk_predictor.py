"""
Risk Prediction Module
Uses scikit-learn Random Forest for contamination risk prediction
"""
import numpy as np
import pickle
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class RiskPredictor:
    """
    Risk predictor for food contamination
    Predicts contamination risk based on food type, storage time, and temperature
    """
    
    def __init__(self, model_path=None):
        """
        Initialize risk predictor
        
        Args:
            model_path: path to saved model
        """
        self.model = None
        self.food_type_encoder = LabelEncoder()
        self.food_types = ['dairy', 'meat', 'seafood', 'produce', 'other']
        self.food_type_encoder.fit(self.food_types)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning(f"Model not found at {model_path}")

    def build_model(self, n_estimators=100, random_state=42):
        """
        Build Random Forest model for risk prediction
        
        Args:
            n_estimators: number of trees in forest
            random_state: random seed
        """
        try:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=15,
                random_state=random_state,
                n_jobs=-1
            )
            logger.info("Risk prediction model built successfully")
            return self.model
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def prepare_features(self, food_type, storage_time_hours, temperature):
        """
        Prepare features for prediction
        
        Args:
            food_type: type of food (dairy, meat, seafood, produce, other)
            storage_time_hours: hours stored
            temperature: storage temperature in Celsius
            
        Returns:
            feature array
        """
        # Encode food type
        food_type_encoded = self.food_type_encoder.transform([food_type])[0]
        
        # Create feature array
        features = np.array([[
            food_type_encoded,
            storage_time_hours,
            temperature
        ]])
        
        return features

    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: training features (N, 3) - [food_type_encoded, storage_time, temperature]
            y_train: training target - risk percentages (0-100)
        """
        try:
            if self.model is None:
                self.build_model()
            
            self.model.fit(X_train, y_train)
            logger.info("Model training completed")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def save_model(self, filepath):
        """Save model to file"""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            # Create directory if not exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model and encoder
            model_data = {
                'model': self.model,
                'food_type_encoder': self.food_type_encoder,
                'food_types': self.food_types
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath):
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.food_type_encoder = model_data['food_type_encoder']
            self.food_types = model_data['food_types']
            
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, food_type, storage_time_hours, temperature):
        """
        Predict contamination risk
        
        Args:
            food_type: type of food
            storage_time_hours: storage time in hours
            temperature: storage temperature in Celsius
            
        Returns:
            risk_percentage: contamination risk (0-100)
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            # Prepare features
            features = self.prepare_features(food_type, storage_time_hours, temperature)
            
            # Make prediction
            risk_percentage = self.model.predict(features)[0]
            
            # Ensure within valid range
            risk_percentage = max(0, min(100, risk_percentage))
            
            logger.info(f"Risk prediction: {risk_percentage:.2f}%")
            return risk_percentage
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def predict_batch(self, food_types, storage_times, temperatures):
        """
        Predict batch of samples
        
        Args:
            food_types: list of food types
            storage_times: list of storage times in hours
            temperatures: list of temperatures
            
        Returns:
            list of risk percentages
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            # Encode food types
            food_types_encoded = self.food_type_encoder.transform(food_types)
            
            # Create feature matrix
            features = np.column_stack([
                food_types_encoded,
                storage_times,
                temperatures
            ])
            
            # Make predictions
            risk_percentages = self.model.predict(features)
            
            # Ensure within valid range
            risk_percentages = np.clip(risk_percentages, 0, 100)
            
            return risk_percentages.tolist()
            
        except Exception as e:
            logger.error(f"Error making batch prediction: {str(e)}")
            raise

    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        feature_names = ['food_type', 'storage_time_hours', 'temperature']
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances))
