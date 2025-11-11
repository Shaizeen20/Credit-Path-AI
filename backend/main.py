"""
CreditPath-AI FastAPI Backend
Loan Default Risk Prediction API with Recommendation Engine + Authentication
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
from datetime import datetime, timedelta
import os
import sys
import logging
import jwt
import hashlib

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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# In-memory user database
users_db = {}

# ==============================================================================
# SETUP FRONTEND PATH
# ==============================================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(APP_DIR, "frontend")

logger.info(f"APP_DIR: {APP_DIR}")
logger.info(f"Static path: {static_path}")
logger.info(f"Static exists: {os.path.exists(static_path)}")

# ==============================================================================
# LOAD MODEL ARTIFACTS
# ==============================================================================

try:
    model = pickle.load(open('creditpath_model.pkl', 'rb'))
    logger.info("✓ Model loaded successfully")
    
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    logger.info("✓ Scaler loaded successfully")
    
    feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
    logger.info(f"✓ Feature columns loaded: {len(feature_columns)} features")
    
    thresholds = pickle.load(open('thresholds.pkl', 'rb'))
    LOW_THRESHOLD = thresholds['low_threshold']
    HIGH_THRESHOLD = thresholds['high_threshold']
    logger.info(f"✓ Thresholds loaded: Low={LOW_THRESHOLD}, High={HIGH_THRESHOLD}")
    
except Exception as e:
    logger.error(f"Error loading model artifacts: {str(e)}")
    raise

# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================

class SignupRequest(BaseModel):
    firstName: str = Field(..., min_length=2, max_length=50)
    lastName: str = Field(..., min_length=2, max_length=50)
    company: str = Field(..., min_length=2, max_length=100)
    email: str = Field(...)
    password: str = Field(..., min_length=8, max_length=100)

class LoginRequest(BaseModel):
    email: str = Field(...)
    password: str = Field(...)

class AuthResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    user: Optional[dict] = None

class BorrowerInput(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., gt=0)
    person_emp_exp: int = Field(..., ge=0, le=50)
    person_home_ownership: str = Field(...)
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., ge=0, le=30)
    loan_intent: str = Field(...)
    credit_score: int = Field(..., ge=300, le=900)
    cb_person_cred_hist_length: int = Field(..., ge=0, le=50)
    previous_loan_defaults_on_file: str = Field(...)
    
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
    default_probability: float
    risk_level: str
    threshold_range: str
    recommendation: dict
    borrower_summary: dict
    timestamp: str

# ==============================================================================
# AUTHENTICATION UTILITIES
# ==============================================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

def create_access_token(email: str, expires_delta: Optional[timedelta] = None) -> str:
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    expire = datetime.utcnow() + expires_delta
    to_encode = {"email": email, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ==============================================================================
# RECOMMENDATION ENGINE
# ==============================================================================

def get_recommendation(probability: float) -> dict:
    if probability < LOW_THRESHOLD:
        return {
            'risk_level': 'Low Risk',
            'action': 'Send standard payment reminder',
            'priority': 'Normal',
            'method': 'Automated SMS/Email',
            'timeline': 'Standard schedule',
            'details': 'Borrower shows good repayment capability.',
            'next_steps': ['Send automated payment reminder 3 days before due date', 'Continue standard monitoring']
        }
    elif probability < HIGH_THRESHOLD:
        return {
            'risk_level': 'Medium Risk',
            'action': 'Make personalized call to discuss the loan',
            'priority': 'Medium',
            'method': 'Personalized phone call',
            'timeline': 'Within 3-5 business days',
            'details': 'Borrower shows moderate risk.',
            'next_steps': ['Schedule personalized call with borrower', 'Understand current financial situation']
        }
    else:
        return {
            'risk_level': 'High Risk',
            'action': 'Prioritize collection efforts from borrower',
            'priority': 'Urgent',
            'method': 'Direct intervention by senior recovery agent',
            'timeline': 'Within 24-48 hours',
            'details': 'High default probability detected.',
            'next_steps': ['Assign to senior recovery agent immediately', 'Schedule urgent meeting with borrower']
        }

def classify_risk_level(probability: float) -> str:
    if probability < LOW_THRESHOLD:
        return "Low Risk"
    elif probability < HIGH_THRESHOLD:
        return "Medium Risk"
    else:
        return "High Risk"

def get_threshold_range(probability: float) -> str:
    if probability < LOW_THRESHOLD:
        return f"0.0 - {LOW_THRESHOLD}"
    elif probability < HIGH_THRESHOLD:
        return f"{LOW_THRESHOLD} - {HIGH_THRESHOLD}"
    else:
        return f"{HIGH_THRESHOLD} - 1.0"

# ==============================================================================
# PREPROCESSING
# ==============================================================================

def preprocess_input(borrower_data: dict) -> pd.DataFrame:
    try:
        df = pd.DataFrame([borrower_data])
        df['LTI_Ratio'] = df['loan_amnt'] / df['person_income']
        df['loan_percent_income'] = df['loan_amnt'] / df['person_income']
        df['Credit_Stability_Index'] = df['credit_score'] / (df['cb_person_cred_hist_length'] + 1)
        
        categorical_cols = ['person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        numerical_features = ['person_age', 'person_income', 'person_emp_exp', 
                              'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                              'cb_person_cred_hist_length', 'credit_score',
                              'LTI_Ratio', 'Credit_Stability_Index']
        
        df[numerical_features] = scaler.transform(df[numerical_features])
        
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_columns]
        return df
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

# ==============================================================================
# HTML ROUTES - SERVE PAGES
# ==============================================================================

@app.get("/", include_in_schema=False)
async def serve_root():
    """Serve login.html at root"""
    login_path = os.path.join(static_path, "login.html")
    logger.info(f"GET / -> {login_path} exists: {os.path.exists(login_path)}")
    return FileResponse(login_path, media_type="text/html")

@app.get("/home", include_in_schema=False)
async def serve_home():
    """Serve index.html at /home"""
    index_path = os.path.join(static_path, "index.html")
    logger.info(f"GET /home -> {index_path} exists: {os.path.exists(index_path)}")
    return FileResponse(index_path, media_type="text/html")

@app.get("/predict", include_in_schema=False)
async def serve_predict():
    """Serve predict.html"""
    page_path = os.path.join(static_path, "predict.html")
    logger.info(f"GET /predict -> {page_path} exists: {os.path.exists(page_path)}")
    return FileResponse(page_path, media_type="text/html")

@app.get("/batch", include_in_schema=False)
async def serve_batch():
    """Serve batch.html"""
    page_path = os.path.join(static_path, "batch.html")
    logger.info(f"GET /batch -> {page_path} exists: {os.path.exists(page_path)}")
    return FileResponse(page_path, media_type="text/html")

# ==============================================================================
# AUTHENTICATION ENDPOINTS
# ==============================================================================

@app.post("/api/signup", response_model=AuthResponse, tags=["Auth"])
async def signup(request: SignupRequest):
    if request.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = hash_password(request.password)
    user = {
        "email": request.email,
        "firstName": request.firstName,
        "lastName": request.lastName,
        "company": request.company,
        "password": hashed_password,
        "created_at": datetime.now().isoformat()
    }
    
    users_db[request.email] = user
    logger.info(f"✓ User registered: {request.email}")
    
    return AuthResponse(
        success=True,
        message="Account created successfully. Please login.",
        token=None,
        user={"email": user["email"], "firstName": user["firstName"], "lastName": user["lastName"], "company": user["company"]}
    )

@app.post("/api/login", response_model=AuthResponse, tags=["Auth"])
async def login(request: LoginRequest):
    if request.email not in users_db:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    user = users_db[request.email]
    
    if not verify_password(request.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token(request.email)
    logger.info(f"✓ User logged in: {request.email}")
    
    return AuthResponse(
        success=True,
        message="Login successful",
        token=access_token,
        user={"email": user["email"], "firstName": user["firstName"], "lastName": user["lastName"], "company": user["company"]}
    )

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/api/health", tags=["API"])
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/model_info", tags=["API"])
def model_info():
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
    try:
        borrower_dict = borrower.dict()
        processed_data = preprocess_input(borrower_dict)
        prediction_proba = model.predict_proba(processed_data)[0][1]
        
        recommendation = get_recommendation(prediction_proba)
        risk_level = classify_risk_level(prediction_proba)
        threshold_range = get_threshold_range(prediction_proba)
        
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
        
        logger.info(f"Prediction: {prediction_proba:.4f}, Risk: {risk_level}")
        
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
    try:
        results = []
        for borrower in borrowers:
            prediction = await predict_default(borrower)
            results.append(prediction.dict())
        
        return {
            "total_predictions": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# ==============================================================================
# STATIC FILES - MOUNT LAST
# ==============================================================================

if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    logger.info(f"✓ Static files mounted from {static_path}")
else:
    logger.error(f"✗ Static path not found: {static_path}")

# ==============================================================================
# STARTUP
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 80)
    logger.info("CreditPath-AI API Starting...")
    logger.info("=" * 80)
    logger.info(f"Model: LightGBM (AUC: 0.9907)")
    logger.info(f"Thresholds: Low={LOW_THRESHOLD}, High={HIGH_THRESHOLD}")
    logger.info("Auth: JWT enabled")
    logger.info("Frontend: Available at /")
    logger.info("=" * 80)

# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")