"""
CreditPath-AI FastAPI Backend
Loan Default Risk Prediction API with Recommendation Engine
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CreditPath-AI API",
    description="AI-Powered Loan Default Risk Prediction System",
    version="1.0.0"
)

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# SERVE FRONTEND FILES
# ==============================================================================

# Serve static files (CSS, JS, images)
static_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    logger.info(f"✓ Frontend static files mounted from {static_path}")
else:
    logger.warning(f"Frontend directory not found at {static_path}")

# Serve index.html for root path
@app.get("/", include_in_schema=False)
async def serve_root():
    """Serve frontend index.html"""
    index_path = os.path.join(static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "CreditPath-AI API - Frontend home page"}

# Serve individual HTML pages
@app.get("/predict", include_in_schema=False)
async def serve_predict():
    """Serve predict page"""
    page_path = os.path.join(static_path, "predict.html")
    if os.path.exists(page_path):
        return FileResponse(page_path, media_type="text/html")
    return {"error": "Predict page not found"}

@app.get("/batch", include_in_schema=False)
async def serve_batch():
    """Serve batch page"""
    page_path = os.path.join(static_path, "batch.html")
    if os.path.exists(page_path):
        return FileResponse(page_path, media_type="text/html")
    return {"error": "Batch page not found"}

# ==============================================================================
# LOAD MODEL ARTIFACTS
# ==============================================================================

try:
    # Load trained model
    model = pickle.load(open('creditpath_model.pkl', 'rb'))
    logger.info("✓ Model loaded successfully")
    
    # Load scaler
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    logger.info("✓ Scaler loaded successfully")
    
    # Load feature columns
    feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
    logger.info(f"✓ Feature columns loaded: {len(feature_columns)} features")
    
    # Load thresholds
    thresholds = pickle.load(open('thresholds.pkl', 'rb'))
    LOW_THRESHOLD = thresholds['low_threshold']
    HIGH_THRESHOLD = thresholds['high_threshold']
    logger.info(f"✓ Thresholds loaded: Low={LOW_THRESHOLD}, High={HIGH_THRESHOLD}")
    
except Exception as e:
    logger.error(f"Error loading model artifacts: {str(e)}")
    raise

# ==============================================================================
# PYDANTIC MODELS (Input Validation)
# ==============================================================================

class BorrowerInput(BaseModel):
    """Input schema for single borrower prediction"""
    
    person_age: int = Field(..., ge=18, le=100, description="Age of borrower")
    person_income: float = Field(..., gt=0, description="Annual income in rupees")
    person_emp_exp: int = Field(..., ge=0, le=50, description="Employment experience in years")
    person_home_ownership: str = Field(..., description="Home ownership status")
    loan_amnt: float = Field(..., gt=0, description="Loan amount requested")
    loan_int_rate: float = Field(..., ge=0, le=30, description="Loan interest rate")
    loan_intent: str = Field(..., description="Purpose of loan")
    credit_score: int = Field(..., ge=300, le=900, description="Credit score")
    cb_person_cred_hist_length: int = Field(..., ge=0, le=50, description="Credit history length in years")
    previous_loan_defaults_on_file: str = Field(..., description="Previous defaults: Yes or No")
    
    @validator('person_home_ownership')
    def validate_home_ownership(cls, v):
        valid_values = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
        if v.upper() not in valid_values:
            raise ValueError(f"Must be one of: {valid_values}")
        return v.upper()
    
    @validator('loan_intent')
    def validate_loan_intent(cls, v):
        valid_values = ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
        if v.upper() not in valid_values:
            raise ValueError(f"Must be one of: {valid_values}")
        return v.upper()
    
    @validator('previous_loan_defaults_on_file')
    def validate_defaults(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError("Must be 'Yes' or 'No'")
        return v.capitalize()

class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    
    default_probability: float
    risk_level: str
    threshold_range: str
    recommendation: dict
    borrower_summary: dict
    timestamp: str

# ==============================================================================
# RECOMMENDATION ENGINE
# ==============================================================================

def get_recommendation(probability: float) -> dict:
    """
    Generate action recommendation based on default probability
    Following banking industry best practices
    """
    if probability < LOW_THRESHOLD:
        return {
            'risk_level': 'Low Risk',
            'action': 'Send standard payment reminder',
            'priority': 'Normal',
            'method': 'Automated SMS/Email',
            'timeline': 'Standard schedule',
            'details': 'Borrower shows good repayment capability. Continue regular monitoring with automated reminders.',
            'next_steps': [
                'Send automated payment reminder 3 days before due date',
                'Continue standard monitoring',
                'Eligible for future loan considerations',
                'Maintain regular follow-up schedule'
            ]
        }
    elif probability < HIGH_THRESHOLD:
        return {
            'risk_level': 'Medium Risk',
            'action': 'Make personalized call to discuss the loan',
            'priority': 'Medium',
            'method': 'Personalized phone call',
            'timeline': 'Within 3-5 business days',
            'details': 'Borrower shows moderate risk. Personal engagement recommended to understand their situation and provide support.',
            'next_steps': [
                'Schedule personalized call with borrower',
                'Understand current financial situation',
                'Offer flexible payment options if needed',
                'Set up closer monitoring schedule',
                'Document conversation and commitments'
            ]
        }
    else:
        return {
            'risk_level': 'High Risk',
            'action': 'Prioritize collection efforts from borrower',
            'priority': 'Urgent',
            'method': 'Direct intervention by senior recovery agent',
            'timeline': 'Within 24-48 hours',
            'details': 'High default probability detected. Immediate action required. Consider loan restructuring or initiate recovery process.',
            'next_steps': [
                'Assign to senior recovery agent immediately',
                'Schedule urgent meeting with borrower',
                'Review collateral and guarantor details',
                'Discuss loan restructuring options',
                'Initiate recovery proceedings if necessary',
                'Escalate to management if no response'
            ]
        }

def classify_risk_level(probability: float) -> str:
    """Classify risk level based on thresholds"""
    if probability < LOW_THRESHOLD:
        return "Low Risk"
    elif probability < HIGH_THRESHOLD:
        return "Medium Risk"
    else:
        return "High Risk"

def get_threshold_range(probability: float) -> str:
    """Get the threshold range for given probability"""
    if probability < LOW_THRESHOLD:
        return f"0.0 - {LOW_THRESHOLD}"
    elif probability < HIGH_THRESHOLD:
        return f"{LOW_THRESHOLD} - {HIGH_THRESHOLD}"
    else:
        return f"{HIGH_THRESHOLD} - 1.0"

# ==============================================================================
# PREPROCESSING FUNCTIONS
# ==============================================================================

def preprocess_input(borrower_data: dict) -> pd.DataFrame:
    """
    Preprocess borrower input to match training data format
    """
    try:
        # Create DataFrame
        df = pd.DataFrame([borrower_data])
        
        # Calculate derived features
        df['LTI_Ratio'] = df['loan_amnt'] / df['person_income']
        df['loan_percent_income'] = df['loan_amnt'] / df['person_income']
        df['Credit_Stability_Index'] = df['credit_score'] / (df['cb_person_cred_hist_length'] + 1)
        
        # Add dummy columns for categorical variables (these will be created during encoding)
        categorical_cols = ['person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
        
        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Numerical features to scale
        numerical_features = ['person_age', 'person_income', 'person_emp_exp', 
                              'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                              'cb_person_cred_hist_length', 'credit_score',
                              'LTI_Ratio', 'Credit_Stability_Index']
        
        # Scale numerical features
        df[numerical_features] = scaler.transform(df[numerical_features])
        
        # Ensure all feature columns exist (add missing ones with 0)
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the columns used during training, in the same order
        df = df[feature_columns]
        
        return df
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/api/", tags=["API Info"])
def root():
    """Root endpoint - API information"""
    return {
        "message": "CreditPath-AI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict",
            "model_info": "/api/model_info",
            "frontend": "/"
        }
    }

@app.get("/api/health", tags=["API Info"])
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/model_info", tags=["API Info"])
def model_info():
    """Get model information and configuration"""
    return {
        "model_type": "LightGBM",
        "auc_score": 0.9907,
        "accuracy": 0.95,
        "features_count": len(feature_columns),
        "thresholds": {
            "low_risk": f"< {LOW_THRESHOLD}",
            "medium_risk": f"{LOW_THRESHOLD} - {HIGH_THRESHOLD}",
            "high_risk": f"> {HIGH_THRESHOLD}"
        },
        "version": "1.0.0"
    }

@app.post("/api/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_default(borrower: BorrowerInput):
    """
    Predict loan default risk for a single borrower
    Returns probability, risk level, and action recommendations
    """
    try:
        logger.info(f"Received prediction request for borrower with income: {borrower.person_income}")
        
        # Convert input to dict
        borrower_dict = borrower.dict()
        
        # Preprocess input
        processed_data = preprocess_input(borrower_dict)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[0][1]
        
        # Get recommendation
        recommendation = get_recommendation(prediction_proba)
        risk_level = classify_risk_level(prediction_proba)
        threshold_range = get_threshold_range(prediction_proba)
        
        # Create borrower summary
        borrower_summary = {
            "age": borrower.person_age,
            "annual_income": borrower.person_income,
            "loan_amount": borrower.loan_amnt,
            "credit_score": borrower.credit_score,
            "employment_experience": borrower.person_emp_exp,
            "credit_history_length": borrower.cb_person_cred_hist_length,
            "loan_to_income_ratio": round((borrower.loan_amnt / borrower.person_income) * 100, 2),
            "home_ownership": borrower.person_home_ownership,
            "loan_intent": borrower.loan_intent
        }
        
        # Log prediction
        logger.info(f"Prediction completed: Probability={prediction_proba:.4f}, Risk={risk_level}")
        
        # Return response
        return PredictionResponse(
            default_probability=round(float(prediction_proba), 4),
            risk_level=risk_level,
            threshold_range=threshold_range,
            recommendation=recommendation,
            borrower_summary=borrower_summary,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict_batch", tags=["Predictions"])
async def predict_batch(borrowers: List[BorrowerInput]):
    """
    Predict loan default risk for multiple borrowers
    Useful for batch processing
    """
    try:
        logger.info(f"Received batch prediction request for {len(borrowers)} borrowers")
        
        results = []
        for borrower in borrowers:
            # Reuse single prediction logic
            prediction = await predict_default(borrower)
            results.append(prediction.dict())
        
        logger.info(f"Batch prediction completed: {len(results)} predictions")
        
        return {
            "total_predictions": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# ==============================================================================
# ERROR HANDLERS
# ==============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.detail}")
    return {
        "error": True,
        "message": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unexpected error: {str(exc)}")
    return {
        "error": True,
        "message": "An unexpected error occurred. Please try again later.",
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }

# ==============================================================================
# STARTUP EVENT
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Runs when the application starts"""
    logger.info("=" * 80)
    logger.info("CreditPath-AI API Starting...")
    logger.info("=" * 80)
    logger.info(f"Model: LightGBM (AUC: 0.9907)")
    logger.info(f"Features: {len(feature_columns)}")
    logger.info(f"Thresholds: Low={LOW_THRESHOLD}, High={HIGH_THRESHOLD}")
    logger.info("Frontend: Available at /")
    logger.info("API Docs: Available at /api/docs")
    logger.info("API Ready to serve predictions!")
    logger.info("=" * 80)

# ==============================================================================
# RUN APPLICATION
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Heroku sets this)
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )