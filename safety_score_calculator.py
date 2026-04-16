"""
Food Safety Score Calculator & Prevention Advisory System
Generates comprehensive safety scores and recommendations
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FoodSafetyScoreCalculator:
    """Calculate comprehensive food safety scores"""
    
    def __init__(self):
        self.score_weights = {
            'agricultural_practices': 0.35,
            'environmental_risk': 0.30,
            'ai_prediction': 0.35
        }

    def calculate_agricultural_practices_score(self, agricultural_data, contamination_risks):
        """
        Score agricultural practices (0-100, higher is better)
        
        Assessment criteria:
        - Pesticide usage and frequency
        - Fertilizer compliance
        - Water source quality
        - Harvest timing (pre-harvest interval compliance)
        """
        score = 100
        
        # Pesticide penalty
        if agricultural_data.get('pesticide_used'):
            days_since = agricultural_data.get('days_since_pesticide', 0)
            if days_since < 7:
                score -= 30  # Recent application = high penalty
            elif days_since < 14:
                score -= 15
            elif days_since < 21:
                score -= 5
        
        # Water source assessment
        irrigation_source = agricultural_data.get('irrigation_source', '').lower()
        if irrigation_source in ['river', 'canal']:
            score -= 15
        elif irrigation_source in ['well', 'groundwater']:
            score += 5
        
        # Harvest timing assessment
        harvest_safety = contamination_risks.get('harvest_safety', {})
        if not harvest_safety.get('harvest_safe', False):
            score -= 20
        
        # Best practices bonus
        if not agricultural_data.get('pesticide_used'):
            score += 20  # Organic practices bonus
        
        return max(min(score, 100), 0)

    def calculate_environmental_risk_score(self, contamination_risks):
        """
        Convert environmental contamination risk to score (0-100, higher is better)
        """
        env_risk = contamination_risks.get('environmental_risk', {})
        risk_score = env_risk.get('risk_score', 0)
        
        # Convert risk score (0-100 where higher=worse) to safety score (0-100 where higher=better)
        safety_score = 100 - risk_score
        return max(min(safety_score, 100), 0)

    def calculate_ai_prediction_score(self, image_prediction, risk_prediction):
        """
        Combine AI model predictions into score (0-100, higher is better)
        """
        score = 100
        
        # Image analysis
        if image_prediction and image_prediction.get('prediction') == 'spoiled':
            score -= 40
        
        # Risk prediction
        if risk_prediction and risk_prediction.get('risk_percentage'):
            risk_pct = risk_prediction['risk_percentage']
            score -= (risk_pct * 0.6)  # Risk score heavily impacts AI score
        
        return max(min(score, 100), 0)

    def calculate_overall_score(self, agricultural_data, contamination_risks, 
                               image_prediction=None, risk_prediction=None):
        """
        Calculate overall food safety score (0-100)
        
        Args:
            agricultural_data: dict with farming practices
            contamination_risks: dict from agricultural_risk_calculator
            image_prediction: optional image analysis result
            risk_prediction: optional risk model prediction
            
        Returns:
            dict with overall score and component scores
        """
        ag_score = self.calculate_agricultural_practices_score(agricultural_data, contamination_risks)
        env_score = self.calculate_environmental_risk_score(contamination_risks)
        ai_score = self.calculate_ai_prediction_score(image_prediction, risk_prediction)
        
        # Weighted average
        overall_score = (
            ag_score * self.score_weights['agricultural_practices'] +
            env_score * self.score_weights['environmental_risk'] +
            ai_score * self.score_weights['ai_prediction']
        )
        
        # Determine safety status
        safe_for_consumption = overall_score >= 60
        
        return {
            'overall_score': round(overall_score),
            'agricultural_practices_score': round(ag_score),
            'environmental_risk_score': round(env_score),
            'ai_prediction_score': round(ai_score),
            'safe_for_consumption': safe_for_consumption,
            'score_breakdown': {
                'agricultural_practices': {
                    'score': round(ag_score),
                    'weight': self.score_weights['agricultural_practices']
                },
                'environmental_risk': {
                    'score': round(env_score),
                    'weight': self.score_weights['environmental_risk']
                },
                'ai_prediction': {
                    'score': round(ai_score),
                    'weight': self.score_weights['ai_prediction']
                }
            }
        }

    def generate_explanation(self, agricultural_data, contamination_risks, overall_score):
        """Generate human-readable explanation of the score"""
        explanations = []
        
        # Agricultural practices
        ag_risks = []
        if agricultural_data.get('pesticide_used'):
            days = agricultural_data.get('days_since_pesticide', 0)
            if days < 21:
                ag_risks.append(f"pesticide applied {days} days ago")
        
        if agricultural_data.get('irrigation_source', '').lower() in ['river', 'canal']:
            ag_risks.append("irrigation from potentially contaminated water source")
        
        if ag_risks:
            explanations.append(f"Agricultural concerns: {', '.join(ag_risks)}")
        
        # Environmental risks
        chemical_risk = contamination_risks.get('chemical_risk', {})
        biological_risk = contamination_risks.get('biological_risk', {})
        env_risk = contamination_risks.get('environmental_risk', {})
        
        risks_detected = []
        if chemical_risk.get('risk_level') == 'high':
            risks_detected.append("high chemical contamination risk")
        if biological_risk.get('risk_level') == 'high':
            risks_detected.append("high biological contamination risk")
        if env_risk.get('risk_level') == 'high':
            risks_detected.append("adverse environmental conditions")
        
        if risks_detected:
            explanations.append(f"Risk assessment: {', '.join(risks_detected)}")
        
        final_assessment = "SAFE" if overall_score['safe_for_consumption'] else "NOT RECOMMENDED"
        explanations.append(f"Overall assessment: {final_assessment} for consumption")
        
        return " | ".join(explanations)


class PreventionAdvisorySystem:
    """Generate prevention and mitigation recommendations"""
    
    def __init__(self):
        self.advisory_rules = {
            'pesticide': self._pesticide_advisories,
            'water': self._water_advisories,
            'environmental': self._environmental_advisories,
            'harvest': self._harvest_advisories,
            'storage': self._storage_advisories
        }

    def generate_recommendations(self, agricultural_data, contamination_risks, harvest_safety):
        """
        Generate actionable prevention recommendations
        
        Returns:
            list of recommendation dicts with priority and action
        """
        recommendations = []
        
        # Pesticide recommendations
        if agricultural_data.get('pesticide_used'):
            recs = self._pesticide_advisories(agricultural_data, contamination_risks)
            recommendations.extend(recs)
        
        # Water source recommendations
        recs = self._water_advisories(agricultural_data, contamination_risks)
        recommendations.extend(recs)
        
        # Environmental recommendations
        recs = self._environmental_advisories(contamination_risks)
        recommendations.extend(recs)
        
        # Harvest recommendations
        recs = self._harvest_advisories(harvest_safety, contamination_risks)
        recommendations.extend(recs)
        
        # Storage recommendations
        recs = self._storage_advisories(agricultural_data)
        recommendations.extend(recs)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations[:10]  # Return top 10 recommendations

    def _pesticide_advisories(self, agricultural_data, contamination_risks):
        """Generate pesticide-related advisories"""
        recommendations = []
        chemical_risk = contamination_risks.get('chemical_risk', {})
        
        days_since = agricultural_data.get('days_since_pesticide', 0)
        pesticide = agricultural_data.get('pesticide_used')
        
        if chemical_risk.get('risk_level') == 'high':
            if days_since < 7:
                recommendations.append({
                    'priority': 9,
                    'category': 'pesticide',
                    'action': f"DELAY HARVEST: {pesticide} applied only {days_since} days ago",
                    'reason': f"Standard pre-harvest interval is 14-21 days. Wait at least {max(0, 21-days_since)} more days.",
                    'impact': 'high'
                })
            elif days_since < 14:
                recommendations.append({
                    'priority': 8,
                    'category': 'pesticide',
                    'action': f"Delay harvest by {max(0, 14-days_since)} days if possible",
                    'reason': "Pesticide residue levels still elevated",
                    'impact': 'medium'
                })
        
        if days_since > 0 and pesticide:
            recommendations.append({
                'priority': 7,
                'category': 'pesticide',
                'action': "Consider alternative pest management for next cycle",
                'reason': "Reduce chemical residue accumulation with integrated pest management (IPM)",
                'impact': 'medium'
            })
        
        return recommendations

    def _water_advisories(self, agricultural_data, contamination_risks):
        """Generate water source related advisories"""
        recommendations = []
        biological_risk = contamination_risks.get('biological_risk', {})
        
        source = agricultural_data.get('irrigation_source', '').lower()
        
        if biological_risk.get('risk_level') == 'high':
            if source in ['river', 'canal']:
                recommendations.append({
                    'priority': 9,
                    'category': 'water',
                    'action': "Switch to groundwater or well source for irrigation",
                    'reason': "River/canal water has high contamination risk",
                    'impact': 'high'
                })
            
            recommendations.append({
                'priority': 8,
                'category': 'water',
                'action': "Treat irrigation water with filtration or chlorination",
                'reason': "Current water source shows high biological contamination risk",
                'impact': 'high'
            })
        
        if source == 'unknown':
            recommendations.append({
                'priority': 6,
                'category': 'water',
                'action': "Document and test irrigation water source",
                'reason': "Water source quality unknown - implement testing program",
                'impact': 'medium'
            })
        
        return recommendations

    def _environmental_advisories(self, contamination_risks):
        """Generate environment-based advisories"""
        recommendations = []
        env_risk = contamination_risks.get('environmental_risk', {})
        risk_factors = env_risk.get('risk_factors', {})
        
        if risk_factors.get('extreme_heat'):
            recommendations.append({
                'priority': 7,
                'category': 'environmental',
                'action': "Increase irrigation frequency to prevent crop stress",
                'reason': "Extreme heat increases disease susceptibility",
                'impact': 'medium'
            })
        
        if risk_factors.get('high_humidity'):
            recommendations.append({
                'priority': 7,
                'category': 'environmental',
                'action': "Improve crop spacing and ventilation to reduce fungal diseases",
                'reason': "High humidity promotes fungal growth",
                'impact': 'medium'
            })
        
        if risk_factors.get('heavy_rainfall'):
            recommendations.append({
                'priority': 8,
                'category': 'environmental',
                'action': "Inspect field for soil contamination and wash produce before consumption",
                'reason': "Heavy rain increases soil pathogen splash contamination",
                'impact': 'high'
            })
        
        return recommendations

    def _harvest_advisories(self, harvest_safety, contamination_risks):
        """Generate harvest timing advisories"""
        recommendations = []
        
        days_until_safe = harvest_safety.get('days_until_safe', 0)
        
        if days_until_safe > 0:
            recommendations.append({
                'priority': 9,
                'category': 'harvest',
                'action': f"Wait {days_until_safe} more days before harvest",
                'reason': f"Pre-harvest interval not yet complete. Current safety: {harvest_safety.get('reason', 'N/A')}",
                'impact': 'high'
            })
        else:
            recommendations.append({
                'priority': 5,
                'category': 'harvest',
                'action': "Safe to harvest - product meets pre-harvest intervals",
                'reason': "All pesticide residue thresholds met",
                'impact': 'low'
            })
        
        return recommendations

    def _storage_advisories(self, agricultural_data):
        """Generate storage and handling advisories"""
        recommendations = []
        crop_type = agricultural_data.get('crop_type', '').lower()
        
        if crop_type in ['vegetables', 'fruits', 'leafy_greens', 'produce']:
            recommendations.append({
                'priority': 6,
                'category': 'storage',
                'action': "Store in cool conditions (4-10°C) for optimal preservation",
                'reason': "Fresh produce requires refrigeration to slow spoilage",
                'impact': 'medium'
            })
        
        recommendations.append({
            'priority': 5,
            'category': 'storage',
            'action': "Wash produce thoroughly before consumption",
            'reason': "Standard food safety practice to remove surface contaminants",
            'impact': 'medium'
        })
        
        return recommendations

    def generate_advisory_summary(self, recommendations):
        """Generate a summary of critical recommendations"""
        if not recommendations:
            return {
                'critical_count': 0,
                'critical_actions': [],
                'summary': "No critical recommendations at this time"
            }
        
        critical = [r for r in recommendations if r['priority'] >= 8]
        
        return {
            'critical_count': len(critical),
            'critical_actions': [r['action'] for r in critical[:5]],
            'summary': f"Multiple critical safety concerns detected. {len(critical)} high-priority actions required."
        }
