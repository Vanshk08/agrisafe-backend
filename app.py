"""
AgriSafe AI Backend - Agricultural Contamination Detection System
Flask API for food contamination and agricultural risk detection
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
import uuid

# Import configuration and models
from config import (
    SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS, UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS, MAX_FILE_SIZE, FOOD_TYPES, CROP_TYPES
)
from models import (
    db, AgriculturalInput, EnvironmentalData, ContaminationRisk, 
    FoodSafetyScore, PredictionHistory
)
from image_classifier import ImageClassifier
from risk_predictor import RiskPredictor
from agricultural_risk_calculator import AgriculturalRiskCalculator
from safety_score_calculator import FoodSafetyScoreCalculator, PreventionAdvisorySystem

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize database
db.init_app(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
try:
    image_classifier = ImageClassifier(model_path='models/food_classifier.pkl')
    risk_predictor = RiskPredictor(model_path='models/risk_predictor.pkl')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.warning(f"Models not found yet: {e}. Train models first.")
    image_classifier = None
    risk_predictor = None

# Initialize risk calculators
agricultural_risk_calc = AgriculturalRiskCalculator()
food_safety_score_calc = FoodSafetyScoreCalculator()
prevention_advisor = PreventionAdvisorySystem()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.before_request
def create_tables():
    """Create database tables if they don't exist"""
    db.create_all()


# ==================== BATCH TRACEABILITY ENDPOINTS ====================

@app.route('/api/batch/<batch_id>', methods=['GET'])
def get_batch_details(batch_id):
    """
    Get complete traceability information for a batch
    
    Returns:
        - agricultural_input: farming data
        - environmental_data: weather/environmental data
        - contamination_risks: all risk assessments
        - food_safety_score: overall safety score
        - prediction_history: all predictions made
    """
    try:
        ag_input = AgriculturalInput.query.filter_by(batch_id=batch_id).first()
        if not ag_input:
            return jsonify({'error': f'Batch {batch_id} not found'}), 404
        
        env_data = EnvironmentalData.query.filter_by(batch_id=batch_id).first()
        contamination_risks = ContaminationRisk.query.filter_by(batch_id=batch_id).all()
        food_safety_score = FoodSafetyScore.query.filter_by(batch_id=batch_id).first()
        prediction_history = PredictionHistory.query.filter_by(batch_id=batch_id).all()
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'agricultural_input': ag_input.to_dict(),
            'environmental_data': env_data.to_dict() if env_data else None,
            'contamination_risks': [r.to_dict() for r in contamination_risks],
            'food_safety_score': food_safety_score.to_dict() if food_safety_score else None,
            'prediction_history': [p.to_dict() for p in prediction_history],
            'traceability': {
                'farm_location': ag_input.farm_location,
                'crop_type': ag_input.crop_type,
                'days_in_supply_chain': ag_input.days_since_harvest,
                'agricultural_inputs': {
                    'pesticide': ag_input.pesticide_used,
                    'fertilizer': ag_input.fertilizer_used,
                    'water_source': ag_input.irrigation_source
                },
                'created_at': ag_input.created_at.isoformat() if ag_input.created_at else None
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error retrieving batch details: {str(e)}")
        return jsonify({'error': f'Failed to retrieve batch: {str(e)}'}), 500


@app.route('/api/batch/<batch_id>/history', methods=['GET'])
def get_batch_history(batch_id):
    """
    Get prediction history for a batch
    
    Returns:
        - list of all predictions made for this batch
    """
    try:
        ag_input = AgriculturalInput.query.filter_by(batch_id=batch_id).first()
        if not ag_input:
            return jsonify({'error': f'Batch {batch_id} not found'}), 404
        
        prediction_history = PredictionHistory.query.filter_by(batch_id=batch_id)\
            .order_by(PredictionHistory.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'prediction_count': len(prediction_history),
            'predictions': [p.to_dict() for p in prediction_history],
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error retrieving batch history: {str(e)}")
        return jsonify({'error': f'Failed to retrieve history: {str(e)}'}), 500


@app.route('/api/batches', methods=['GET'])
def list_batches():
    """
    Get all batches with summary information
    
    Query parameters:
        - limit: max results (default 50)
        - offset: pagination offset (default 0)
    
    Returns:
        - list of batches with summary data
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        batches = AgriculturalInput.query.limit(limit).offset(offset).all()
        total_count = AgriculturalInput.query.count()
        
        batch_summaries = []
        for batch in batches:
            safety_score = FoodSafetyScore.query.filter_by(batch_id=batch.batch_id).first()
            
            batch_summaries.append({
                'batch_id': batch.batch_id,
                'crop_type': batch.crop_type,
                'farm_location': batch.farm_location,
                'days_since_harvest': batch.days_since_harvest,
                'safety_score': safety_score.overall_score if safety_score else None,
                'safe_for_consumption': safety_score.safe_for_consumption if safety_score else None,
                'created_at': batch.created_at.isoformat() if batch.created_at else None
            })
        
        return jsonify({
            'success': True,
            'total_batches': total_count,
            'returned': len(batch_summaries),
            'batches': batch_summaries,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total': total_count
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error listing batches: {str(e)}")
        return jsonify({'error': f'Failed to list batches: {str(e)}'}), 500


# ==================== METADATA ENDPOINTS ====================

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Get available crop types, irrigation sources, etc."""
    return jsonify({
        'crop_types': CROP_TYPES,
        'irrigation_sources': ['river', 'groundwater', 'rain', 'pumped', 'well', 'canal'],
        'food_types': FOOD_TYPES,
        'contamination_types': ['chemical', 'biological', 'environmental'],
        'risk_levels': ['low', 'medium', 'high']
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint to check if API is running"""
    return jsonify({
        'status': 'online',
        'message': 'AgriSafe AI API is running',
        'version': '1.0.0',
        'endpoints': {
            'health_check': '/health',
            'predict_image': '/predict-image',
            'predict_risk': '/predict-risk'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': image_classifier is not None and risk_predictor is not None,
        'database': 'connected'
    }), 200



@app.route('/api/predict-image', methods=['POST'])
def predict_image():
    """
    Predict food contamination from image
    
    Returns:
        - prediction: 'fresh' or 'spoiled'
        - confidence: confidence score (0-1)
        - confidence_percentage: confidence percentage (0-100)
    """
    try:
        # Check if models are loaded
        if image_classifier is None:
            return jsonify({
                'error': 'Image classification model not loaded. Please train the model first.'
            }), 503

        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Optional: batch_id for linking to agricultural data
        batch_id = request.form.get('batch_id')

        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Allowed: png, jpg, jpeg, gif'}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        prediction, confidence = image_classifier.predict(filepath)

        # Store prediction in history if batch_id provided
        if batch_id:
            ag_input = AgriculturalInput.query.filter_by(batch_id=batch_id).first()
            if ag_input:
                pred_history = PredictionHistory(
                    batch_id=batch_id,
                    prediction_type='image',
                    image_path=filename,
                    image_prediction=prediction,
                    image_confidence=float(confidence),
                    agricultural_input_id=ag_input.id
                )
                db.session.add(pred_history)
                db.session.commit()
                logger.info(f"Image prediction stored: batch_id={batch_id}")

        # Clean up - remove temporary file
        os.remove(filepath)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'confidence_percentage': float(confidence * 100),
            'message': f'Food classified as {prediction} with {confidence*100:.2f}% confidence'
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/predict-risk', methods=['POST'])
def predict_risk():
    """
    Predict contamination risk based on food properties
    
    Expected JSON:
        - food_type: string ('dairy', 'meat', 'seafood', 'produce', 'other')
        - storage_time_hours: int (0-72)
        - temperature: float (Celsius)
        - batch_id: optional, links to agricultural data
    
    Returns:
        - risk_percentage: contamination risk percentage (0-100)
        - risk_level: 'low', 'medium', or 'high'
        - safe_to_eat: boolean
        - agricultural_context: if batch_id provided
    """
    try:
        # Check if model is loaded
        if risk_predictor is None:
            return jsonify({
                'error': 'Risk prediction model not loaded. Please train the model first.'
            }), 503

        # Validate input
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['food_type', 'storage_time_hours', 'temperature']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400

        # Validate field types and ranges
        food_type = data['food_type'].lower()
        storage_time_hours = float(data['storage_time_hours'])
        temperature = float(data['temperature'])
        batch_id = data.get('batch_id')

        valid_food_types = FOOD_TYPES
        if food_type not in valid_food_types:
            return jsonify({
                'error': f'Invalid food type. Valid options: {valid_food_types}'
            }), 400

        if storage_time_hours < 0 or storage_time_hours > 240:
            return jsonify({'error': 'Storage time must be between 0-240 hours'}), 400

        if temperature < -20 or temperature > 50:
            return jsonify({'error': 'Temperature must be between -20°C to 50°C'}), 400

        # Make prediction
        risk_percentage = risk_predictor.predict(food_type, storage_time_hours, temperature)

        # Determine risk level and safety
        if risk_percentage < 30:
            risk_level = 'low'
            safe_to_eat = True
        elif risk_percentage < 70:
            risk_level = 'medium'
            safe_to_eat = False
        else:
            risk_level = 'high'
            safe_to_eat = False

        # Store prediction in history if batch_id provided
        agricultural_context = None
        if batch_id:
            ag_input = AgriculturalInput.query.filter_by(batch_id=batch_id).first()
            if ag_input:
                pred_history = PredictionHistory(
                    batch_id=batch_id,
                    prediction_type='risk',
                    risk_percentage=float(risk_percentage),
                    contamination_risk=risk_level,
                    agricultural_input_id=ag_input.id
                )
                db.session.add(pred_history)
                db.session.commit()
                logger.info(f"Risk prediction stored: batch_id={batch_id}")
                
                # Get agricultural context
                agricultural_context = {
                    'crop_type': ag_input.crop_type,
                    'pesticide_used': ag_input.pesticide_used,
                    'days_since_pesticide': ag_input.days_since_pesticide,
                    'irrigation_source': ag_input.irrigation_source,
                    'days_since_harvest': ag_input.days_since_harvest
                }

        return jsonify({
            'success': True,
            'risk_percentage': float(risk_percentage),
            'risk_level': risk_level,
            'safe_to_eat': safe_to_eat,
            'food_type': food_type,
            'storage_time_hours': storage_time_hours,
            'temperature': temperature,
            'agricultural_context': agricultural_context,
            'message': f'Risk: {risk_percentage:.2f}% - {risk_level.upper()}'
        }), 200

    except ValueError as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Risk prediction error: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Risk prediction failed: {str(e)}'}), 500


@app.route('/api/food-types', methods=['GET'])
def get_food_types():
    """Get list of available food types"""
    return jsonify({
        'food_types': FOOD_TYPES,
        'crop_types': CROP_TYPES
    }), 200


# ==================== AGRICULTURAL INPUT ENDPOINTS ====================

@app.route('/api/agricultural-input', methods=['POST'])
def submit_agricultural_input():
    """
    Submit agricultural input data and store in database
    
    Expected JSON:
        - batch_id: unique batch identifier (will be generated if not provided)
        - crop_type: type of crop
        - pesticide_used: name of pesticide (optional)
        - pesticide_quantity: kg/hectare (optional)
        - days_since_pesticide: days since application (optional)
        - fertilizer_used: name of fertilizer (optional)
        - fertilizer_quantity: kg/hectare (optional)
        - irrigation_source: water source
        - farm_location: location of farm
        - days_since_harvest: days elapsed since harvest
        - farm_area: hectares (optional)
        - temperature: current temperature (optional)
        - humidity: humidity percentage (optional)
        - rainfall: rainfall in mm (optional)
        - soil_moisture: soil moisture percentage (optional)
    
    Returns:
        - batch_id: assigned batch identifier
        - agricultural_data: stored agricultural data
        - environmental_data: stored environmental data
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Generate batch ID if not provided
        batch_id = data.get('batch_id') or str(uuid.uuid4())[:8]
        
        # Validate required fields
        required_fields = ['crop_type', 'irrigation_source', 'days_since_harvest']
        missing = [f for f in required_fields if f not in data or data[f] is None]
        
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400
        
        # Check if batch already exists
        existing = AgriculturalInput.query.filter_by(batch_id=batch_id).first()
        if existing:
            return jsonify({'error': f'Batch ID {batch_id} already exists'}), 409
        
        # Create agricultural input record
        ag_input = AgriculturalInput(
            batch_id=batch_id,
            crop_type=data.get('crop_type'),
            pesticide_used=data.get('pesticide_used'),
            pesticide_quantity=data.get('pesticide_quantity'),
            days_since_pesticide=data.get('days_since_pesticide', 0),
            fertilizer_used=data.get('fertilizer_used'),
            fertilizer_quantity=data.get('fertilizer_quantity'),
            irrigation_source=data.get('irrigation_source'),
            farm_location=data.get('farm_location'),
            days_since_harvest=int(data.get('days_since_harvest', 0)),
            farm_area=data.get('farm_area')
        )
        
        db.session.add(ag_input)
        db.session.commit()
        
        # Store environmental data if provided
        env_data = None
        if any(k in data for k in ['temperature', 'humidity', 'rainfall', 'soil_moisture', 'wind_speed']):
            env_data = EnvironmentalData(
                batch_id=batch_id,
                temperature=data.get('temperature'),
                humidity=data.get('humidity'),
                rainfall=data.get('rainfall'),
                soil_moisture=data.get('soil_moisture'),
                wind_speed=data.get('wind_speed')
            )
            db.session.add(env_data)
            db.session.commit()
        
        logger.info(f"Agricultural input stored: batch_id={batch_id}")
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'agricultural_data': ag_input.to_dict(),
            'environmental_data': env_data.to_dict() if env_data else None,
            'message': f'Agricultural data recorded for batch {batch_id}'
        }), 201
    
    except Exception as e:
        logger.error(f"Error storing agricultural input: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Failed to store agricultural data: {str(e)}'}), 500


@app.route('/api/agricultural-risk/<batch_id>', methods=['GET'])
def get_agricultural_risk(batch_id):
    """
    Calculate agricultural contamination risk for a batch
    
    Returns:
        - overall_risk
        - chemical_risk
        - biological_risk
        - environmental_risk
        - harvest_safety
    """
    try:
        # Get agricultural input
        ag_input = AgriculturalInput.query.filter_by(batch_id=batch_id).first()
        if not ag_input:
            return jsonify({'error': f'Batch {batch_id} not found'}), 404
        
        # Get environmental data
        env_data = EnvironmentalData.query.filter_by(batch_id=batch_id).first()
        
        # Build data dictionaries
        ag_data = {
            'crop_type': ag_input.crop_type,
            'pesticide_used': ag_input.pesticide_used,
            'days_since_pesticide': ag_input.days_since_pesticide,
            'pesticide_quantity': ag_input.pesticide_quantity,
            'fertilizer_used': ag_input.fertilizer_used,
            'irrigation_source': ag_input.irrigation_source,
            'farm_location': ag_input.farm_location,
            'days_since_harvest': ag_input.days_since_harvest,
            'farm_area': ag_input.farm_area
        }
        
        env_dict = None
        if env_data:
            env_dict = {
                'temperature': env_data.temperature or 20,
                'humidity': env_data.humidity or 50,
                'rainfall': env_data.rainfall or 0,
                'soil_moisture': env_data.soil_moisture or 50,
                'wind_speed': env_data.wind_speed or 0
            }
        
        # Calculate risks
        risk_results = agricultural_risk_calc.calculate_overall_risk(ag_data, env_dict)
        
        # Store contamination risks
        for contamination_type in ['chemical', 'biological', 'environmental']:
            risk_data = risk_results[f'{contamination_type}_risk']
            
            existing = ContaminationRisk.query.filter_by(
                batch_id=batch_id,
                contamination_type=contamination_type
            ).first()
            
            if existing:
                existing.risk_score = risk_data['risk_score']
                existing.risk_level = risk_data['risk_level']
                existing.primary_cause = risk_data['primary_cause']
                existing.probability_score = risk_data['probability_score']
            else:
                contamination_risk = ContaminationRisk(
                    batch_id=batch_id,
                    contamination_type=contamination_type,
                    risk_score=risk_data['risk_score'],
                    risk_level=risk_data['risk_level'],
                    primary_cause=risk_data['primary_cause'],
                    probability_score=risk_data['probability_score'],
                    harvest_safe=risk_results['harvest_safety']['harvest_safe'] if contamination_type == 'chemical' else None,
                    days_until_safe=risk_results['harvest_safety']['days_until_safe'] if contamination_type == 'chemical' else None
                )
                db.session.add(contamination_risk)
        
        db.session.commit()
        logger.info(f"Agricultural risks calculated: batch_id={batch_id}")
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'risks': risk_results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error calculating agricultural risk: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Risk calculation failed: {str(e)}'}), 500


# ==================== FOOD SAFETY SCORE ENDPOINTS ====================

@app.route('/api/food-safety-score/<batch_id>', methods=['POST'])
def calculate_food_safety_score(batch_id):
    """
    Calculate overall food safety score for a batch
    
    Expected JSON (optional):
        - image_prediction: result from image analysis
        - risk_prediction: result from risk model
    
    Returns:
        - overall_score (0-100)
        - component scores
        - safe_for_consumption: boolean
        - explanation
        - recommendations
    """
    try:
        # Get agricultural input
        ag_input = AgriculturalInput.query.filter_by(batch_id=batch_id).first()
        if not ag_input:
            return jsonify({'error': f'Batch {batch_id} not found'}), 404
        
        # Get contamination risks
        contamination_risks = ContaminationRisk.query.filter_by(batch_id=batch_id).all()
        if not contamination_risks:
            return jsonify({'error': f'No risk data found for batch {batch_id}'}), 404
        
        # Convert contamination risks to dict
        risks_dict = {
            risk.contamination_type + '_risk': {
                'risk_score': risk.risk_score,
                'risk_level': risk.risk_level,
                'probability_score': risk.probability_score,
                'primary_cause': risk.primary_cause
            }
            for risk in contamination_risks
        }
        
        # Get harvest safety
        chemical_risk = next((r for r in contamination_risks if r.contamination_type == 'chemical'), None)
        harvest_safety = {
            'harvest_safe': chemical_risk.harvest_safe if chemical_risk else True,
            'days_until_safe': chemical_risk.days_until_safe if chemical_risk else 0
        }
        risks_dict['harvest_safety'] = harvest_safety
        
        # Parse optional image and risk predictions
        data = request.get_json(silent=True) or {}
        image_pred = data.get('image_prediction')
        risk_pred = data.get('risk_prediction')
        
        # Convert ag_input to dict
        ag_data = {
            'crop_type': ag_input.crop_type,
            'pesticide_used': ag_input.pesticide_used,
            'days_since_pesticide': ag_input.days_since_pesticide,
            'irrigation_source': ag_input.irrigation_source,
            'days_since_harvest': ag_input.days_since_harvest
        }
        
        # Calculate score
        score_result = food_safety_score_calc.calculate_overall_score(
            ag_data, risks_dict, image_pred, risk_pred
        )
        
        # Calculate recommendations
        recommendations = prevention_advisor.generate_recommendations(
            ag_data, risks_dict, harvest_safety
        )
        
        # Generate explanation
        explanation = food_safety_score_calc.generate_explanation(ag_data, risks_dict, score_result)
        
        # Store food safety score
        existing_score = FoodSafetyScore.query.filter_by(batch_id=batch_id).first()
        
        score_data = FoodSafetyScore(
            batch_id=batch_id,
            overall_score=score_result['overall_score'],
            agricultural_practices_score=score_result['agricultural_practices_score'],
            environmental_risk_score=score_result['environmental_risk_score'],
            ai_prediction_score=score_result['ai_prediction_score'],
            safe_for_consumption=score_result['safe_for_consumption'],
            explanation=explanation,
            recommendations=[r for r in recommendations]
        )
        
        if existing_score:
            existing_score.overall_score = score_data.overall_score
            existing_score.agricultural_practices_score = score_data.agricultural_practices_score
            existing_score.environmental_risk_score = score_data.environmental_risk_score
            existing_score.ai_prediction_score = score_data.ai_prediction_score
            existing_score.safe_for_consumption = score_data.safe_for_consumption
            existing_score.explanation = score_data.explanation
            existing_score.recommendations = score_data.recommendations
        else:
            db.session.add(score_data)
        
        db.session.commit()
        logger.info(f"Food safety score calculated: batch_id={batch_id}, score={score_result['overall_score']}")
        
        advisory_summary = prevention_advisor.generate_advisory_summary(recommendations)
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'safety_score': score_result,
            'explanation': explanation,
            'recommendations': recommendations[:5],  # Top 5 recommendations
            'advisory_summary': advisory_summary,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error calculating food safety score: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Score calculation failed: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    
    # Run development server
    app.run(debug=True, host='0.0.0.0', port=5000)

