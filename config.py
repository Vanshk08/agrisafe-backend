"""
AgriSafe AI Backend Configuration
"""

import os

# Flask Configuration
DEBUG = True
DEVELOPMENT = True

# Database Configuration
# Use PostgreSQL on Vercel (DATABASE_URL env var), SQLite locally
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///agrisafe.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = DEBUG

# Upload Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Model Paths
MODEL_PATHS = {
    'image_classifier': 'models/food_classifier.pkl',
    'risk_predictor': 'models/risk_predictor.pkl'
}

# Food Types
FOOD_TYPES = ['dairy', 'meat', 'seafood', 'produce', 'other']
CROP_TYPES = ['grain', 'vegetables', 'fruits', 'legumes', 'herbs', 'spices', 'nuts', 'root_crops', 'leafy_greens', 'other']
IRRIGATION_SOURCES = ['river', 'groundwater', 'rain', 'pumped', 'well', 'canal']

# Risk Thresholds
RISK_THRESHOLDS = {
    'low': 30,
    'medium': 70,
    'high': 100
}

# Pesticide Toxicity Levels (WHO Classification)
PESTICIDE_TOXICITY = {
    'ia': 5.0,      # Extremely Hazardous
    'ib': 4.5,      # Highly Hazardous
    'ii': 3.5,      # Moderately Hazardous
    'iii': 2.0,     # Slightly Hazardous
    'u': 1.0,       # Unlikely to present hazard
}

# Environmental Risk Factors
ENVIRONMENTAL_RISK_FACTORS = {
    'high_temperature_threshold': 25,  # Celsius - encourages microbial growth
    'high_humidity_threshold': 75,      # Percentage - increases fungal risk
    'waterborne_contamination_risk': {
        'river': 0.8,
        'groundwater': 0.3,
        'rain': 0.5,
        'pumped': 0.6,
        'well': 0.4,
        'canal': 0.7
    }
}

# Contamination Types
CONTAMINATION_TYPES = ['chemical', 'biological', 'environmental']

# CORS Configuration
CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5000']

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

