"""
FastAPI application for Polish Bankruptcy Prediction
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import mlflow

from ..features.financial_ratios import FinancialRatiosEngineer
from ..models.bankruptcy_classifier import BankruptcyClassifier
from .models import (BatchPredictionRequest, BatchPredictionResponse,
                     HealthResponse, PredictionRequest, PredictionResponse)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Polish Bankruptcy Prediction API",
    description="MLOps API for predicting company bankruptcies using financial data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model caching
MODEL_CACHE = {}
FEATURE_ENGINEER = None
MODEL_REGISTRY_NAME = "BankruptcyXgboost"


def get_feature_engineer():
    """Get or initialize feature engineer"""
    global FEATURE_ENGINEER
    if FEATURE_ENGINEER is None:
        FEATURE_ENGINEER = FinancialRatiosEngineer()
    return FEATURE_ENGINEER


def load_model(model_name: str = None, version: str = "latest") -> Any:
    """Load model from MLflow registry"""
    try:
        if model_name is None:
            model_name = MODEL_REGISTRY_NAME

        cache_key = f"{model_name}_{version}"

        if cache_key in MODEL_CACHE:
            logger.info(f"Using cached model: {cache_key}")
            return MODEL_CACHE[cache_key]

        # Set MLflow tracking URI
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(mlflow_uri)

        # Load model from registry
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Loading model from: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)

        # Cache the model
        MODEL_CACHE[cache_key] = model
        logger.info(f"Model {cache_key} loaded and cached successfully")

        return model

    except Exception as e:
        logger.error(f"Error loading model {model_name}:{version}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Polish Bankruptcy Prediction API...")

    try:
        # Pre-load the default model
        load_model()

        # Initialize feature engineer
        get_feature_engineer()

        logger.info("API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        model_loaded = len(MODEL_CACHE) > 0

        # Check MLflow connection
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow_healthy = True
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.get_experiment("0")  # Try to get default experiment
        except:
            mlflow_healthy = False

        status = "healthy" if model_loaded and mlflow_healthy else "degraded"

        return HealthResponse(
            status=status,
            model_loaded=model_loaded,
            mlflow_connection=mlflow_healthy,
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            mlflow_connection=False,
            timestamp=datetime.utcnow(),
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_bankruptcy(request: PredictionRequest):
    """
    Predict bankruptcy probability for a single company
    """
    try:
        logger.info("Processing prediction request")

        # Load model
        model = load_model()

        # Convert request to DataFrame
        features_dict = request.dict()
        features_df = pd.DataFrame([features_dict])

        # Apply feature engineering
        feature_engineer = get_feature_engineer()
        features_engineered = feature_engineer.transform(features_df)

        # Remove target column if present
        if "bankruptcy" in features_engineered.columns:
            features_engineered = features_engineered.drop("bankruptcy", axis=1)

        # Make prediction
        prediction = model.predict(features_engineered)[0]
        probability = model.predict_proba(features_engineered)[0]

        # Get probability for bankruptcy class (class 1)
        bankruptcy_prob = probability[1] if len(probability) > 1 else probability[0]

        # Determine risk level
        if bankruptcy_prob < 0.3:
            risk_level = "low"
        elif bankruptcy_prob < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Get model info
        model_version = f"{MODEL_REGISTRY_NAME}_latest"

        response = PredictionResponse(
            prediction=int(prediction),
            probability=float(bankruptcy_prob),
            risk_level=risk_level,
            model_version=model_version,
            timestamp=datetime.utcnow(),
            features_used=len(features_engineered.columns),
        )

        logger.info(f"Prediction completed: {response.dict()}")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_bankruptcy(request: BatchPredictionRequest):
    """
    Predict bankruptcy probability for multiple companies
    """
    try:
        logger.info(
            f"Processing batch prediction request for {len(request.samples)} samples"
        )

        # Load model
        model = load_model()

        # Convert requests to DataFrame
        features_list = [sample.dict() for sample in request.samples]
        features_df = pd.DataFrame(features_list)

        # Apply feature engineering
        feature_engineer = get_feature_engineer()
        features_engineered = feature_engineer.transform(features_df)

        # Remove target column if present
        if "bankruptcy" in features_engineered.columns:
            features_engineered = features_engineered.drop("bankruptcy", axis=1)

        # Make predictions
        predictions = model.predict(features_engineered)
        probabilities = model.predict_proba(features_engineered)

        # Process results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            bankruptcy_prob = prob[1] if len(prob) > 1 else prob[0]

            # Determine risk level
            if bankruptcy_prob < 0.3:
                risk_level = "low"
            elif bankruptcy_prob < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"

            results.append(
                PredictionResponse(
                    prediction=int(pred),
                    probability=float(bankruptcy_prob),
                    risk_level=risk_level,
                    model_version=f"{MODEL_REGISTRY_NAME}_latest",
                    timestamp=datetime.utcnow(),
                    features_used=len(features_engineered.columns),
                )
            )

        response = BatchPredictionResponse(
            predictions=results,
            total_samples=len(results),
            processing_time_ms=0,  # Could add timing if needed
            timestamp=datetime.utcnow(),
        )

        logger.info(f"Batch prediction completed for {len(results)} samples")
        return response

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """List available models from MLflow registry"""
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(mlflow_uri)

        client = mlflow.MlflowClient()
        models = client.search_registered_models()

        model_info = []
        for model in models:
            latest_versions = client.get_latest_versions(
                model.name, stages=["None", "Staging", "Production"]
            )

            for version in latest_versions:
                model_info.append(
                    {
                        "name": model.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "description": model.description,
                        "creation_timestamp": version.creation_timestamp,
                    }
                )

        return {"models": model_info}

    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Polish Bankruptcy Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
