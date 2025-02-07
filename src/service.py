import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, JSON
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
OUTPUT_FOLDER = BASE_DIR / "models" 
CONFIG_FOLDER = BASE_DIR / "config"
MODEL_FOLDER = BASE_DIR / "models"

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "user123": "password123",
    "user456": "password456"
}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/models/admission/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})

            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return {"detail": "Invalid credentials"}, 401
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})

            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response

# Pydantic model to validate input data
class InputModel(BaseModel):
    GRE: float 
    TOEFL: float
    University: float
    SOP: float
    LOR: float
    CGPA: float
    Research: int
    
def load_config():
    """
    Load the model selection config (config.json).
    """
    with open(os.path.join(CONFIG_FOLDER, "config.json")) as f:
        config = json.load(f)
    return config

def load_scaler():
    """
    Load previously used scaler to scale input data
    """
    scaler_path = MODEL_FOLDER / "scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        return scaler
    else:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

# Get current model type from config
model_name = load_config().get("model_name")
model_name = model_name.lower()

# Get the model from the Model Store
admission_runner = bentoml.sklearn.get(f"admission_{model_name}:latest").to_runner()

# Get Scaler
scaler = load_scaler()

# Create a service API
rf_service = bentoml.Service("bianca_van_hemert_admission", runners=[admission_runner])

# Add the JWTAuthMiddleware to the service
rf_service.add_asgi_middleware(JWTAuthMiddleware)

# Create an API endpoint for the service
@rf_service.api(input=JSON(), output=JSON())
def login(credentials: dict) -> dict:
    try:
        username = credentials.get("username")
        password = credentials.get("password")

        if username in USERS and USERS[username] == password:
            token = create_jwt_token(username)
            return {"token": token}
        else:
            return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})
    except Exception as e:
        # Log the exception (optional)
        print(f"Login error: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    
# Create an API endpoint for the service
@rf_service.api(
    input=JSON(pydantic_model=InputModel),
    output=JSON(),
    route='v1/models/admission/predict'
)
async def classify(input_data: InputModel, ctx: bentoml.Context) -> dict:
    request = ctx.request
    user = request.state.user if hasattr(request.state, 'user') else None

    # Convert the input data to a numpy array
    input_series = np.array([input_data.GRE, input_data.TOEFL, input_data.University, input_data.SOP,
                             input_data.LOR, input_data.CGPA, input_data.Research])
    print(input_series)
    # Scale the input data
    input_series = scaler.transform(input_series.reshape(1, -1))
    print(input_series)
    
    result = await admission_runner.predict.async_run(input_series.reshape(1, -1))

    return {
        "prediction": result.tolist(),
        "user": user
    }

# Function to create a JWT token
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token