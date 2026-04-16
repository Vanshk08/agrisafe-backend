"""
Agricultural Contamination Risk Engine
Calculates contamination risks from agricultural inputs and environmental factors
"""
import numpy as np
from config import PESTICIDE_TOXICITY, ENVIRONMENTAL_RISK_FACTORS, CONTAMINATION_TYPES
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AgriculturalRiskCalculator:
    """
    Calculate contamination risk from agricultural inputs
    Considers:
    - Pesticide usage and degradation
    - Irrigation water quality
    - Environmental conditions
    - Harvest timing
    """
    
    def __init__(self):
        self.contamination_types = CONTAMINATION_TYPES
        self.pesticide_toxicity = PESTICIDE_TOXICITY
        self.environmental_factors = ENVIRONMENTAL_RISK_FACTORS

    def calculate_chemical_risk(self, agricultural_data, environmental_data=None):
        """
        Calculate chemical contamination risk from pesticides and fertilizers
        
        Args:
            agricultural_data: dict with crop_type, pesticide_used, days_since_pesticide, pesticide_quantity
            environmental_data: dict with temperature, humidity, rainfall
            
        Returns:
            dict with risk_score, probability, explanation
        """
        risk_factors = {}
        
        # 1. Pesticide degradation risk
        pesticide_risk = self._calculate_pesticide_risk(
            agricultural_data.get('pesticide_used'),
            agricultural_data.get('days_since_pesticide', 0),
            agricultural_data.get('pesticide_quantity', 1.0)
        )
        risk_factors['pesticide_risk'] = pesticide_risk
        
        # 2. Environmental acceleration of degradation
        if environmental_data:
            env_acceleration = self._get_environmental_acceleration(environmental_data)
            pesticide_risk = pesticide_risk * env_acceleration
        
        # 3. Days since harvest impact (fresher = higher residue)
        days_since_harvest = agricultural_data.get('days_since_harvest', 0)
        harvest_factor = self._get_harvest_time_factor(days_since_harvest)
        pesticide_risk *= harvest_factor
        
        risk_score = min(pesticide_risk * 100, 100)  # Cap at 100
        probability = self._risk_to_probability(risk_score)
        
        return {
            'contamination_type': 'chemical',
            'risk_score': risk_score,
            'risk_level': self._score_to_level(risk_score),
            'probability_score': probability,
            'primary_cause': self._get_chemical_cause(agricultural_data),
            'risk_factors': risk_factors
        }

    def calculate_biological_risk(self, agricultural_data, environmental_data=None):
        """
        Calculate biological contamination risk from water sources and environmental conditions
        
        Args:
            agricultural_data: dict with irrigation_source
            environmental_data: dict with temperature, humidity, rainfall
            
        Returns:
            dict with risk_score, probability, explanation
        """
        risk_factors = {}
        
        # 1. Water source contamination risk
        water_risk = self._calculate_water_source_risk(
            agricultural_data.get('irrigation_source', 'unknown')
        )
        risk_factors['water_source_risk'] = water_risk
        
        # 2. Environmental conditions (temperature, humidity favor microbial growth)
        env_biological_risk = 0
        if environmental_data:
            env_biological_risk = self._calculate_environmental_biological_risk(environmental_data)
            risk_factors['environmental_risk'] = env_biological_risk
        
        # 3. Combine factors
        combined_risk = (water_risk * 0.6) + (env_biological_risk * 0.4)
        
        # 4. Days since harvest (longer = higher spoilage risk without intervention)
        days_since_harvest = agricultural_data.get('days_since_harvest', 0)
        storage_factor = min(1 + (days_since_harvest / 100), 1.5)  # Max 1.5x multiplier
        
        risk_score = min(combined_risk * storage_factor * 100, 100)
        probability = self._risk_to_probability(risk_score)
        
        return {
            'contamination_type': 'biological',
            'risk_score': risk_score,
            'risk_level': self._score_to_level(risk_score),
            'probability_score': probability,
            'primary_cause': self._get_biological_cause(agricultural_data, environmental_data),
            'risk_factors': risk_factors
        }

    def calculate_environmental_risk(self, agricultural_data, environmental_data=None):
        """
        Calculate environmental contamination from weather and soil conditions
        
        Args:
            agricultural_data: dict with farm_location, crop_type
            environmental_data: dict with temperature, humidity, soil_moisture, wind_speed
            
        Returns:
            dict with risk_score, probability, explanation
        """
        risk_factors = {}
        
        # Default low risk
        risk_score = 0
        
        if environmental_data:
            # Extreme weather events damage crops and create contamination routes
            if environmental_data.get('temperature', 20) > 35:
                risk_score += 20
                risk_factors['extreme_heat'] = True
            
            if environmental_data.get('temperature', 20) < 0:
                risk_score += 15
                risk_factors['frost_damage'] = True
            
            if environmental_data.get('rainfall', 0) > 100:
                risk_score += 25  # Heavy rain increases contamination risk
                risk_factors['heavy_rainfall'] = True
            
            if environmental_data.get('wind_speed', 0) > 40:
                risk_score += 15  # Strong wind can damage crops
                risk_factors['strong_wind'] = True
            
            # Humidity related to disease
            humidity = environmental_data.get('humidity', 60)
            if humidity > 80:
                risk_score += 20
                risk_factors['high_humidity'] = True
        
        probability = self._risk_to_probability(risk_score)
        
        return {
            'contamination_type': 'environmental',
            'risk_score': min(risk_score, 100),
            'risk_level': self._score_to_level(risk_score),
            'probability_score': probability,
            'primary_cause': self._get_environmental_cause(environmental_data),
            'risk_factors': risk_factors
        }

    def calculate_harvest_safety(self, agricultural_data, environmental_data=None):
        """
        Determine if crop is safe to harvest now based on pesticide residue
        
        Returns:
            dict with harvest_safe, days_until_safe, reason
        """
        pesticide_used = agricultural_data.get('pesticide_used')
        days_since_pesticide = agricultural_data.get('days_since_pesticide', 0)
        
        if not pesticide_used:
            return {
                'harvest_safe': True,
                'days_until_safe': 0,
                'reason': 'No pesticides used'
            }
        
        # Standard pre-harvest interval (conservative estimate: 14-30 days depending on pesticide)
        # In real application, this would be based on specific pesticide name
        standard_phi = 21  # days (default)
        
        # Adjust based on environmental conditions (heat accelerates degradation)
        adjustment = 0
        if environmental_data:
            temp = environmental_data.get('temperature', 20)
            if temp > 25:
                adjustment = -(temp - 25) * 0.3  # Roughly 0.3 days per degree C
        
        safe_days = max(0, standard_phi + adjustment - days_since_pesticide)
        
        return {
            'harvest_safe': safe_days <= 0,
            'days_until_safe': int(max(0, safe_days)),
            'reason': f'Pre-harvest interval requires {standard_phi} days, {days_since_pesticide} days passed'
        }

    def calculate_overall_risk(self, agricultural_data, environmental_data=None):
        """
        Calculate overall contamination risk across all types
        
        Returns:
            dict with all risk types and overall assessment
        """
        chemical_risk = self.calculate_chemical_risk(agricultural_data, environmental_data)
        biological_risk = self.calculate_biological_risk(agricultural_data, environmental_data)
        environmental_risk = self.calculate_environmental_risk(agricultural_data, environmental_data)
        harvest_safety = self.calculate_harvest_safety(agricultural_data, environmental_data)
        
        # Weighted average of the three risk types
        overall_score = (
            chemical_risk['risk_score'] * 0.35 +
            biological_risk['risk_score'] * 0.40 +
            environmental_risk['risk_score'] * 0.25
        )
        
        return {
            'overall_risk': {
                'risk_score': overall_score,
                'risk_level': self._score_to_level(overall_score),
                'probability_score': self._risk_to_probability(overall_score)
            },
            'chemical_risk': chemical_risk,
            'biological_risk': biological_risk,
            'environmental_risk': environmental_risk,
            'harvest_safety': harvest_safety,
            'timestamp': datetime.utcnow().isoformat()
        }

    # ==================== Private Methods ====================

    def _calculate_pesticide_risk(self, pesticide_name, days_since_application, quantity):
        """
        Calculate risk from pesticide residue based on degradation
        
        Assumes exponential decay of pesticide: Risk = Initial * 0.5^(days / half_life)
        Default half-life: 14 days
        """
        if not pesticide_name:
            return 0.0
        
        initial_risk = min(quantity / 10, 1.0)  # Normalize quantity to 0-1
        days = max(days_since_application, 0)
        half_life = 14  # days (typical)
        
        # Exponential decay
        remaining_risk = initial_risk * (0.5 ** (days / half_life))
        
        return remaining_risk

    def _get_environmental_acceleration(self, environmental_data):
        """
        Get multiplication factor for pesticide degradation based on environment
        Higher temperature = faster degradation
        """
        temp = environmental_data.get('temperature', 20)
        humidity = environmental_data.get('humidity', 50)
        
        # Temperature effect (rough approximation)
        # Every 10°C increase roughly doubles degradation rate
        temp_factor = 2 ** ((temp - 20) / 10)
        
        # Humidity effect
        humidity_factor = 1 + (humidity - 50) / 100 * 0.2  # Max 0.2 additional factor
        
        return max(temp_factor * humidity_factor, 0.1)  # Min 0.1x

    def _get_harvest_time_factor(self, days_since_harvest):
        """
        Fresh crops have higher pesticide residue
        Factor decreases as time passes (crops are stored/processed)
        """
        # Linear decay: starts at 1.0, decreases by 0.05 per day
        factor = max(1.0 - (days_since_harvest * 0.02), 0.2)
        return factor

    def _calculate_water_source_risk(self, irrigation_source):
        """Get baseline contamination risk for water source"""
        source_lower = irrigation_source.lower() if irrigation_source else 'unknown'
        
        risk_map = {
            'river': 0.8,
            'groundwater': 0.2,
            'rain': 0.3,
            'pumped': 0.5,
            'well': 0.3,
            'canal': 0.7,
            'unknown': 0.5
        }
        
        return risk_map.get(source_lower, 0.5)

    def _calculate_environmental_biological_risk(self, environmental_data):
        """
        Calculate biological risk from temperature and humidity
        Microbes thrive in warm, humid environments
        """
        temp = environmental_data.get('temperature', 20)
        humidity = environmental_data.get('humidity', 50)
        
        # Optimal microbial growth: 20-37°C, >70% humidity
        temp_factor = 0
        if 5 <= temp <= 45:  # Viable range for microbial growth
            if 20 <= temp <= 37:  # Optimal range
                temp_factor = 1.0
            elif 10 <= temp < 20:
                temp_factor = 0.5 + (temp - 10) / 20  # 0.5 to 1.0
            elif 37 < temp <= 45:
                temp_factor = 1.0 - (temp - 37) / 8  # 1.0 to 0
        
        humidity_factor = max(0, (humidity - 50) / 50) if humidity > 50 else 0
        
        combined = (temp_factor * 0.6) + (humidity_factor * 0.4)
        return min(combined, 1.0)

    def _score_to_level(self, score):
        """Convert numerical score to risk level"""
        if score < 30:
            return 'low'
        elif score < 70:
            return 'medium'
        else:
            return 'high'

    def _risk_to_probability(self, risk_score):
        """Convert risk score (0-100) to probability (0-1)"""
        return min(risk_score / 100.0, 1.0)

    def _get_chemical_cause(self, agricultural_data):
        """Generate explanation for chemical risk"""
        pesticide = agricultural_data.get('pesticide_used')
        days = agricultural_data.get('days_since_pesticide', 0)
        
        if not pesticide:
            return "No chemical pesticides detected"
        
        if days < 7:
            return f"Recent pesticide application ({days} days ago) - residue levels still high"
        elif days < 21:
            return f"Pesticide {pesticide} applied {days} days ago - residue present"
        else:
            return f"Pesticide applied {days} days ago - residue degrading"

    def _get_biological_cause(self, agricultural_data, environmental_data):
        """Generate explanation for biological risk"""
        source = agricultural_data.get('irrigation_source', 'unknown')
        causes = []
        
        if source == 'river':
            causes.append("River water irrigation increases pathogen exposure")
        elif source == 'canal':
            causes.append("Canal water may contain agricultural runoff")
        
        if environmental_data:
            temp = environmental_data.get('temperature', 20)
            humidity = environmental_data.get('humidity', 50)
            
            if temp > 25 and humidity > 70:
                causes.append("Warm, humid conditions favor microbial growth")
            elif temp > 30:
                causes.append("High temperature accelerates spoilage")
        
        return " | ".join(causes) if causes else "Potential biological contamination risk"

    def _get_environmental_cause(self, environmental_data):
        """Generate explanation for environmental risk"""
        if not environmental_data:
            return "Insufficient environmental data"
        
        causes = []
        temp = environmental_data.get('temperature', 20)
        humidity = environmental_data.get('humidity', 50)
        rainfall = environmental_data.get('rainfall', 0)
        wind = environmental_data.get('wind_speed', 0)
        
        if temp > 35:
            causes.append("Extreme heat")
        if temp < 0:
            causes.append("Frost damage risk")
        if humidity > 80:
            causes.append("High humidity increases disease risk")
        if rainfall > 100:
            causes.append("Heavy rainfall may cause soil contamination")
        if wind > 40:
            causes.append("Strong winds can damage crops")
        
        return " | ".join(causes) if causes else "Favorable environmental conditions"
