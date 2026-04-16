"""Database Models for AgriSafe AI Agricultural Contamination System
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import json

db = SQLAlchemy()


class AgriculturalInput(db.Model):
    """Store agricultural input data"""
    __tablename__ = 'agricultural_inputs'
    
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    crop_type = db.Column(db.String(100), nullable=False)
    pesticide_used = db.Column(db.String(255), nullable=True)
    pesticide_quantity = db.Column(db.Float, nullable=True)  # in kg/hectare
    days_since_pesticide = db.Column(db.Integer, nullable=True)  # days since application
    fertilizer_used = db.Column(db.String(255), nullable=True)
    fertilizer_quantity = db.Column(db.Float, nullable=True)  # in kg/hectare
    irrigation_source = db.Column(db.String(100), nullable=True)  # river, groundwater, rain
    farm_location = db.Column(db.String(255), nullable=True)
    days_since_harvest = db.Column(db.Integer, nullable=False, default=0)
    farm_area = db.Column(db.Float, nullable=True)  # in hectares
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    predictions = db.relationship('PredictionHistory', foreign_keys='PredictionHistory.agricultural_input_id', backref='agricultural_input', lazy=True)

    def __repr__(self):
        return f'<AgriculturalInput {self.batch_id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'crop_type': self.crop_type,
            'pesticide_used': self.pesticide_used,
            'pesticide_quantity': self.pesticide_quantity,
            'days_since_pesticide': self.days_since_pesticide,
            'fertilizer_used': self.fertilizer_used,
            'fertilizer_quantity': self.fertilizer_quantity,
            'irrigation_source': self.irrigation_source,
            'farm_location': self.farm_location,
            'days_since_harvest': self.days_since_harvest,
            'farm_area': self.farm_area,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class EnvironmentalData(db.Model):
    """Store environmental/weather data for risk analysis"""
    __tablename__ = 'environmental_data'
    
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(50), db.ForeignKey('agricultural_inputs.batch_id'), nullable=False)
    temperature = db.Column(db.Float, nullable=False)  # Celsius
    humidity = db.Column(db.Float, nullable=True)  # 0-100
    rainfall = db.Column(db.Float, nullable=True)  # mm
    soil_moisture = db.Column(db.Float, nullable=True)  # percentage
    light_exposure = db.Column(db.Integer, nullable=True)  # hours per day
    wind_speed = db.Column(db.Float, nullable=True)  # km/h
    date_recorded = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<EnvironmentalData batch={self.batch_id}>'

    def to_dict(self):
        data = {
            'id': self.id,
            'batch_id': self.batch_id,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'rainfall': self.rainfall,
            'soil_moisture': self.soil_moisture,
            'light_exposure': self.light_exposure,
            'wind_speed': self.wind_speed,
            'date_recorded': self.date_recorded.isoformat() if self.date_recorded else None
        }
        return {k: v for k, v in data.items() if v is not None}


class ContaminationRisk(db.Model):
    """Store computed contamination risks"""
    __tablename__ = 'contamination_risks'
    
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(50), db.ForeignKey('agricultural_inputs.batch_id'), nullable=False)
    contamination_type = db.Column(db.String(50), nullable=False)  # chemical, biological, environmental
    risk_score = db.Column(db.Float, nullable=False)  # 0-100
    risk_level = db.Column(db.String(20), nullable=False)  # low, medium, high
    primary_cause = db.Column(db.String(255), nullable=True)  # what caused the risk
    probability_score = db.Column(db.Float, nullable=False)  # 0-1 probability
    harvest_safe = db.Column(db.Boolean, nullable=True)  # can harvest now?
    days_until_safe = db.Column(db.Integer, nullable=True)  # days to wait
    calculated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ContaminationRisk batch={self.batch_id} type={self.contamination_type}>'

    def to_dict(self):
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'contamination_type': self.contamination_type,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'primary_cause': self.primary_cause,
            'probability_score': self.probability_score,
            'harvest_safe': self.harvest_safe,
            'days_until_safe': self.days_until_safe,
            'calculated_at': self.calculated_at.isoformat() if self.calculated_at else None
        }


class FoodSafetyScore(db.Model):
    """Store overall food safety scores with explanations"""
    __tablename__ = 'food_safety_scores'
    
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(50), db.ForeignKey('agricultural_inputs.batch_id'), nullable=False, unique=True)
    overall_score = db.Column(db.Integer, nullable=False)  # 0-100
    agricultural_practices_score = db.Column(db.Integer, nullable=False)  # 0-100
    environmental_risk_score = db.Column(db.Integer, nullable=False)  # 0-100
    ai_prediction_score = db.Column(db.Integer, nullable=False)  # 0-100
    safe_for_consumption = db.Column(db.Boolean, nullable=False)
    explanation = db.Column(db.Text, nullable=True)
    recommendations = db.Column(db.JSON, nullable=True)  # JSON array of recommendations
    calculated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<FoodSafetyScore batch={self.batch_id} score={self.overall_score}>'

    def to_dict(self):
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'overall_score': self.overall_score,
            'agricultural_practices_score': self.agricultural_practices_score,
            'environmental_risk_score': self.environmental_risk_score,
            'ai_prediction_score': self.ai_prediction_score,
            'safe_for_consumption': self.safe_for_consumption,
            'explanation': self.explanation,
            'recommendations': self.recommendations,
            'calculated_at': self.calculated_at.isoformat() if self.calculated_at else None
        }


class PredictionHistory(db.Model):
    """Track all predictions made"""
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(50), db.ForeignKey('agricultural_inputs.batch_id'), nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)  # image, risk, agricultural
    image_path = db.Column(db.String(255), nullable=True)
    image_prediction = db.Column(db.String(50), nullable=True)  # fresh, spoiled
    image_confidence = db.Column(db.Float, nullable=True)
    risk_percentage = db.Column(db.Float, nullable=True)
    contamination_type = db.Column(db.String(50), nullable=True)
    contamination_risk = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    agricultural_input_id = db.Column(db.Integer, db.ForeignKey('agricultural_inputs.id'), nullable=True)

    def __repr__(self):
        return f'<PredictionHistory batch={self.batch_id} type={self.prediction_type}>'

    def to_dict(self):
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'prediction_type': self.prediction_type,
            'image_path': self.image_path,
            'image_prediction': self.image_prediction,
            'image_confidence': self.image_confidence,
            'risk_percentage': self.risk_percentage,
            'contamination_type': self.contamination_type,
            'contamination_risk': self.contamination_risk,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
