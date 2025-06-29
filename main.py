import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Optional, Dict
import random
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from datetime import datetime, time
import math

# Initialize the FastAPI App
app = FastAPI(title="Enhanced Emotional Wellbeing API", version="5.0")

# Enhanced Pydantic Models
class PredictionInput(BaseModel):
    sleepHours: float = Field(..., ge=0, le=24, description="Hours of sleep")
    stepsCount: int = Field(..., ge=0, description="Daily step count")
    caloriesBurnt: int = Field(..., ge=0, description="Calories burnt")
    heartRate: int = Field(..., ge=30, le=200, description="Average heart rate")
    songsSkipped: int = Field(..., ge=0, description="Songs skipped")
    avg_valence: float = Field(..., ge=0, le=1, description="Music valence (positivity)")
    avg_energy: float = Field(..., ge=0, le=1, description="Music energy level")
    avg_danceability: float = Field(..., ge=0, le=1, description="Music danceability")
    socialTime: int = Field(..., description="Total social media time in minutes")
    instagramTime: Optional[int] = Field(None, ge=0, description="Instagram time")
    xTime: Optional[int] = Field(None, ge=0, description="X/Twitter time")
    redditTime: Optional[int] = Field(None, ge=0, description="Reddit time")
    youtubeTime: Optional[int] = Field(None, ge=0, description="YouTube time")
    musicListeningTime: Optional[int] = Field(None, ge=0, description="Music listening time")
    currentHour: Optional[int] = Field(None, ge=0, le=23, description="Current hour (0-23)")

class SmartRecommendation(BaseModel):
    category: str
    text: str
    priority: int = Field(..., ge=1, le=5, description="Priority (1=highest, 5=lowest)")
    actionable: bool = Field(True, description="Whether recommendation is actionable")
    impact_score: float = Field(..., ge=0, le=1, description="Expected impact on wellbeing")
    time_to_implement: str = Field(..., description="Time needed to implement")

class DetailedPrediction(BaseModel):
    predicted_emotion: str
    confidence_score: float = Field(..., ge=0, le=1)
    wellbeing_score: int = Field(..., ge=0, le=100)
    wellbeing_breakdown: Dict[str, float]
    recommendations: List[SmartRecommendation]
    risk_factors: List[str]
    positive_factors: List[str]
    next_check_in: str

# Load enhanced model artifacts
try:
    model = joblib.load('models/emotion_model_v5.joblib')
    scaler = joblib.load('models/scaler_v5.joblib')
    label_encoder = joblib.load('models/label_encoder_v5.joblib')
    model_features = joblib.load('models/features_v5.joblib')
    print("✅ Enhanced model artifacts loaded successfully")
except FileNotFoundError:
    raise RuntimeError("Enhanced model artifacts not found. Please run Cell 2 first.")

class SmartRecommendationEngine:
    """Enhanced recommendation engine with contextual intelligence"""

    def __init__(self):
        self.recommendation_templates = {
            'sleep': {
                'insufficient': [
                    "Your sleep is below optimal. Try setting a consistent bedtime routine.",
                    "Consider reducing screen time 1 hour before bed for better sleep quality.",
                    "A cool, dark environment can significantly improve sleep quality."
                ],
                'excessive': [
                    "Excessive sleep might indicate underlying fatigue. Consider sleep quality over quantity.",
                    "Long sleep periods can disrupt circadian rhythm. Try waking up at a consistent time."
                ]
            },
            'activity': {
                'low': [
                    "Low physical activity detected. Start with a 10-minute walk to boost mood.",
                    "Try the 2-minute rule: commit to just 2 minutes of exercise to build momentum.",
                    "Dancing to your favorite music counts as exercise and can boost mood instantly."
                ],
                'optimal': [
                    "Great activity level! Consider varying your routine to prevent plateaus.",
                    "Your activity is excellent. Focus on recovery and stretching now."
                ]
            },
            'social_media': {
                'excessive': [
                    "High social media usage detected. Try the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds.",
                    "Consider using app timers to limit social media consumption.",
                    "Replace some social media time with real-world social interactions."
                ],
                'platform_specific': [
                    "Instagram: Try following accounts that inspire rather than compare.",
                    "Twitter/X: Consider curating your feed to reduce negative content.",
                    "Reddit: Focus on educational or hobby-related subreddits."
                ]
            },
            'music': {
                'restless': [
                    "Frequent song skipping suggests restlessness. Try a 'focus' or 'chill' playlist.",
                    "Consider mindfulness: listen to one full song without skipping to practice presence."
                ],
                'mood_mismatch': [
                    "Your music choices might not match your emotional needs. Try mood-based playlists.",
                    "Low-energy music when stressed can be more soothing than high-energy tracks."
                ]
            },
            'physiological': {
                'high_hr': [
                    "Elevated heart rate detected. Try 4-7-8 breathing: inhale 4, hold 7, exhale 8.",
                    "Consider brief meditation or progressive muscle relaxation.",
                    "High heart rate might indicate stress. Take short breaks throughout the day."
                ],
                'optimal_hr': [
                    "Your heart rate indicates good cardiovascular health. Keep it up!"
                ]
            }
        }

    def analyze_context(self, data: PredictionInput) -> Dict:
        """Analyze user context for smarter recommendations"""
        context = {}

        # Time-based context
        current_hour = data.currentHour or datetime.now().hour
        if 22 <= current_hour or current_hour <= 6:
            context['time_period'] = 'night'
        elif 6 < current_hour <= 12:
            context['time_period'] = 'morning'
        elif 12 < current_hour <= 18:
            context['time_period'] = 'afternoon'
        else:
            context['time_period'] = 'evening'

        # Activity level context
        if data.stepsCount < 3000:
            context['activity_level'] = 'sedentary'
        elif data.stepsCount < 8000:
            context['activity_level'] = 'moderate'
        else:
            context['activity_level'] = 'active'

        # Sleep quality context
        if data.sleepHours < 6:
            context['sleep_quality'] = 'insufficient'
        elif data.sleepHours > 9:
            context['sleep_quality'] = 'excessive'
        else:
            context['sleep_quality'] = 'adequate'

        # Social media context
        if data.socialTime > 240:
            context['social_usage'] = 'excessive'
        elif data.socialTime > 120:
            context['social_usage'] = 'moderate'
        else:
            context['social_usage'] = 'low'

        return context

    def calculate_impact_score(self, recommendation_type: str, user_data: PredictionInput) -> float:
        """Calculate expected impact of recommendation based on user's current state"""
        impact_weights = {
            'sleep': 0.35, 'activity': 0.25, 'social_media': 0.20,
            'music': 0.10, 'physiological': 0.10
        }

        # Calculate deviation from optimal for each category
        sleep_deviation = abs(user_data.sleepHours - 7.5) / 7.5
        activity_deviation = max(0, (8000 - user_data.stepsCount) / 8000)
        social_deviation = max(0, (user_data.socialTime - 120) / 240)

        category_deviations = {
            'sleep': sleep_deviation,
            'activity': activity_deviation,
            'social_media': social_deviation,
            'music': min(user_data.songsSkipped / 25, 1),
            'physiological': abs(user_data.heartRate - 65) / 50
        }

        base_impact = impact_weights.get(recommendation_type, 0.1)
        deviation_multiplier = category_deviations.get(recommendation_type, 0.5)

        return min(base_impact * (1 + deviation_multiplier), 1.0)

    def generate_smart_recommendations(self, data: PredictionInput, predicted_emotion: str) -> List[SmartRecommendation]:
        """Generate contextually aware recommendations"""
        context = self.analyze_context(data)
        recommendations = []

        # Sleep recommendations
        if data.sleepHours < 6:
            rec = SmartRecommendation(
                category="Sleep Optimization",
                text=random.choice(self.recommendation_templates['sleep']['insufficient']),
                priority=1,
                impact_score=self.calculate_impact_score('sleep', data),
                time_to_implement="Tonight (30 min setup)"
            )
            recommendations.append(rec)
        elif data.sleepHours > 9:
            rec = SmartRecommendation(
                category="Sleep Regulation",
                text=random.choice(self.recommendation_templates['sleep']['excessive']),
                priority=3,
                impact_score=self.calculate_impact_score('sleep', data),
                time_to_implement="1-2 weeks (habit formation)"
            )
            recommendations.append(rec)

        # Activity recommendations
        if data.stepsCount < 5000:
            intensity = "Start small" if data.stepsCount < 2000 else "Gradually increase"
            rec = SmartRecommendation(
                category="Physical Activity",
                text=f"{intensity}: {random.choice(self.recommendation_templates['activity']['low'])}",
                priority=2,
                impact_score=self.calculate_impact_score('activity', data),
                time_to_implement="Immediate (10-15 min)"
            )
            recommendations.append(rec)

        # Social media recommendations
        if data.socialTime > 180:
            platform_specific = []
            if data.instagramTime and data.instagramTime > 90:
                platform_specific.append(f"Instagram usage is high ({data.instagramTime} min)")
            if data.xTime and data.xTime > 75:
                platform_specific.append(f"X/Twitter usage is elevated ({data.xTime} min)")

            text = random.choice(self.recommendation_templates['social_media']['excessive'])
            if platform_specific:
                text += f" Focus on: {', '.join(platform_specific)}."

            rec = SmartRecommendation(
                category="Digital Wellness",
                text=text,
                priority=2 if data.socialTime > 300 else 3,
                impact_score=self.calculate_impact_score('social_media', data),
                time_to_implement="Immediate (app settings)"
            )
            recommendations.append(rec)

        # Music and mood recommendations
        if data.songsSkipped > 15:
            rec = SmartRecommendation(
                category="Music & Mood",
                text=random.choice(self.recommendation_templates['music']['restless']),
                priority=4,
                impact_score=self.calculate_impact_score('music', data),
                time_to_implement="Next listening session"
            )
            recommendations.append(rec)

        # Physiological recommendations
        if data.heartRate > 80:
            rec = SmartRecommendation(
                category="Stress Management",
                text=random.choice(self.recommendation_templates['physiological']['high_hr']),
                priority=1 if data.heartRate > 90 else 2,
                impact_score=self.calculate_impact_score('physiological', data),
                time_to_implement="Immediate (2-5 min)"
            )
            recommendations.append(rec)

        # Emotion-specific recommendations
        if predicted_emotion in ['Stressed', 'Anxious']:
            rec = SmartRecommendation(
                category="Immediate Relief",
                text="Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
                priority=1,
                impact_score=0.7,
                time_to_implement="Right now (2-3 min)"
            )
            recommendations.append(rec)

        # Positive reinforcement for good habits
        positive_habits = []
        if data.sleepHours >= 7 and data.sleepHours <= 8.5:
            positive_habits.append("excellent sleep schedule")
        if data.stepsCount >= 8000:
            positive_habits.append("great physical activity")
        if data.socialTime <= 120:
            positive_habits.append("balanced social media usage")

        if positive_habits and not recommendations:
            rec = SmartRecommendation(
                category="Positive Reinforcement",
                text=f"You're maintaining {', '.join(positive_habits)}. Keep up the excellent work!",
                priority=5,
                impact_score=0.3,
                time_to_implement="Continue current habits"
            )
            recommendations.append(rec)

        # Sort by priority and impact
        recommendations.sort(key=lambda x: (x.priority, -x.impact_score))

        # Return top 3-4 most relevant recommendations
        return recommendations[:min(4, len(recommendations))]

class EnhancedWellbeingScorer:
    """Advanced wellbeing scoring with multiple dimensions"""

    def calculate_comprehensive_score(self, data: PredictionInput) -> tuple:
        """Calculate detailed wellbeing score with breakdown"""

        # Sleep score (0-100)
        sleep_optimal = 7.5
        sleep_deviation = abs(data.sleepHours - sleep_optimal)
        sleep_score = max(0, 100 - (sleep_deviation / sleep_optimal) * 60)

        # Activity score (0-100)
        steps_score = min(100, (data.stepsCount / 10000) * 100)
        calories_score = min(100, (data.caloriesBurnt / 800) * 100)
        activity_score = (steps_score + calories_score) / 2

        # Heart rate score (0-100) - optimal range 60-75
        hr_optimal_range = (60, 75)
        if hr_optimal_range[0] <= data.heartRate <= hr_optimal_range[1]:
            hr_score = 100
        else:
            hr_deviation = min(abs(data.heartRate - hr_optimal_range[0]),
                             abs(data.heartRate - hr_optimal_range[1]))
            hr_score = max(0, 100 - (hr_deviation / 30) * 100)

        # Music mood score (0-100)
        music_positivity = data.avg_valence * 100
        skip_penalty = min((data.songsSkipped / 30) * 50, 50)
        music_score = max(0, music_positivity - skip_penalty)

        # Social media score (0-100) - penalize excessive usage
        if data.socialTime <= 60:
            social_score = 100
        elif data.socialTime <= 180:
            social_score = 100 - ((data.socialTime - 60) / 120) * 40
        else:
            social_score = max(0, 60 - ((data.socialTime - 180) / 240) * 60)

        # Weighted composite score
        weights = {
            'sleep': 0.30,
            'activity': 0.25,
            'heart_rate': 0.15,
            'music_mood': 0.15,
            'social_balance': 0.15
        }

        composite_score = (
            sleep_score * weights['sleep'] +
            activity_score * weights['activity'] +
            hr_score * weights['heart_rate'] +
            music_score * weights['music_mood'] +
            social_score * weights['social_balance']
        )

        breakdown = {
            'sleep_quality': round(sleep_score, 1),
            'physical_activity': round(activity_score, 1),
            'physiological_health': round(hr_score, 1),
            'music_mood': round(music_score, 1),
            'digital_wellness': round(social_score, 1)
        }

        return int(composite_score), breakdown

def identify_risk_and_positive_factors(data: PredictionInput) -> tuple:
    """Identify risk factors and positive factors"""
    risk_factors = []
    positive_factors = []

    # Risk factors
    if data.sleepHours < 6:
        risk_factors.append(f"Insufficient sleep ({data.sleepHours:.1f} hours)")
    if data.stepsCount < 3000:
        risk_factors.append(f"Very low physical activity ({data.stepsCount:,} steps)")
    if data.socialTime > 300:
        risk_factors.append(f"Excessive social media usage ({data.socialTime} minutes)")
    if data.heartRate > 90:
        risk_factors.append(f"Elevated heart rate ({data.heartRate} bpm)")
    if data.songsSkipped > 25:
        risk_factors.append("High music restlessness (frequent skipping)")
    if data.avg_valence < 0.3:
        risk_factors.append("Preference for low-positivity music")

    # Positive factors
    if 7 <= data.sleepHours <= 8.5:
        positive_factors.append(f"Optimal sleep duration ({data.sleepHours:.1f} hours)")
    if data.stepsCount >= 8000:
        positive_factors.append(f"Excellent physical activity ({data.stepsCount:,} steps)")
    if data.socialTime <= 120:
        positive_factors.append("Balanced social media usage")
    if 60 <= data.heartRate <= 75:
        positive_factors.append("Healthy resting heart rate")
    if data.avg_valence >= 0.7:
        positive_factors.append("Positive music preferences")
    if data.caloriesBurnt >= 500:
        positive_factors.append(f"Good caloric burn ({data.caloriesBurnt} calories)")

    return risk_factors, positive_factors

def determine_next_checkin(predicted_emotion: str, risk_factors: List[str]) -> str:
    """Determine when user should check in next based on current state"""
    if predicted_emotion in ['Stressed', 'Anxious'] or len(risk_factors) >= 3:
        return "Check back in 4-6 hours"
    elif predicted_emotion in ['Restless', 'Neutral'] or len(risk_factors) >= 1:
        return "Check back tomorrow"
    else:
        return "Check back in 2-3 days"

# Initialize enhanced components
recommendation_engine = SmartRecommendationEngine()
wellbeing_scorer = EnhancedWellbeingScorer()

@app.post("/predict_enhanced", response_model=DetailedPrediction)
def predict_enhanced(input_data: PredictionInput):
    """Enhanced prediction endpoint with smart recommendations"""
    try:
        # Prepare model input
        model_input_dict = {key: getattr(input_data, key) for key in model_features}
        input_df = pd.DataFrame([model_input_dict])
        input_scaled = scaler.transform(input_df)

        # Get prediction and confidence
        prediction_proba = model.predict_proba(input_scaled)[0]
        predicted_class = model.predict(input_scaled)[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
        confidence_score = float(np.max(prediction_proba))

        # Calculate comprehensive wellbeing score
        wellbeing_score, wellbeing_breakdown = wellbeing_scorer.calculate_comprehensive_score(input_data)

        # Generate smart recommendations
        recommendations = recommendation_engine.generate_smart_recommendations(input_data, predicted_emotion)

        # Identify risk and positive factors
        risk_factors, positive_factors = identify_risk_and_positive_factors(input_data)

        # Determine next check-in
        next_checkin = determine_next_checkin(predicted_emotion, risk_factors)

        return DetailedPrediction(
            predicted_emotion=predicted_emotion,
            confidence_score=confidence_score,
            wellbeing_score=wellbeing_score,
            wellbeing_breakdown=wellbeing_breakdown,
            recommendations=recommendations,
            risk_factors=risk_factors,
            positive_factors=positive_factors,
            next_check_in=next_checkin
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_version": "v5_enhanced"}

# Run the Enhanced Server
print("🚀 Starting Enhanced Emotional Wellbeing API...")

# IMPORTANT: Reset your ngrok token on your dashboard as it was shared publicly
NGROK_AUTH_TOKEN = "2yzuP5u1KEfMovnvwlkKXEd4NR0_N1Z6t5wj9L8hAm6XNCmR"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
nest_asyncio.apply()

try:
    public_url = ngrok.connect(8000).public_url
    print(f"✅ ENHANCED API IS LIVE!")
    print(f"🌐 Public URL: {public_url}")
    print(f"📚 API Documentation: {public_url}/docs")
    print(f"🔍 Health Check: {public_url}/health")
    print(f"🎯 Enhanced Prediction: {public_url}/predict_enhanced")

    uvicorn.run(app, host='0.0.0.0', port=8000)

except Exception as e:
    print(f"❌ Failed to start server: {e}")
    print("Make sure port 8000 is available and ngrok token is valid")
