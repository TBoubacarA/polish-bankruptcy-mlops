"""
Unit tests for FastAPI endpoints
"""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.models import PredictionRequest, PredictionResponse


class TestAPI:
    """Test cases for API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def sample_prediction_request(self):
        """Create sample prediction request"""
        return {
            "attr1": 0.56,
            "attr2": 0.12,
            "attr3": 0.89,
            "attr4": 0.23,
            "attr5": 1.45,
            "attr6": 0.67,
            "attr7": 0.34,
            "attr8": 0.78,
            "attr9": 0.45,
            "attr10": 0.91,
            "attr11": 0.56,
            "attr12": 0.34,
            "attr13": 0.78,
            "attr14": 0.23,
            "attr15": 0.45,
            "attr16": 0.67,
            "attr17": 0.89,
            "attr18": 0.12,
            "attr19": 0.56,
            "attr20": 0.34,
            "attr21": 0.78,
            "attr22": 0.45,
            "attr23": 0.67,
            "attr24": 0.23,
            "attr25": 0.89,
            "attr26": 0.56,
            "attr27": 0.34,
            "attr28": 0.78,
            "attr29": 0.45,
            "attr30": 0.67,
            "attr31": 0.23,
            "attr32": 0.89,
            "attr33": 0.56,
            "attr34": 0.34,
            "attr35": 0.78,
            "attr36": 0.45,
            "attr37": 0.67,
            "attr38": 0.23,
            "attr39": 0.89,
            "attr40": 0.56,
            "attr41": 0.34,
            "attr42": 0.78,
            "attr43": 0.45,
            "attr44": 0.67,
            "attr45": 0.23,
            "attr46": 0.89,
            "attr47": 0.56,
            "attr48": 0.34,
            "attr49": 0.78,
            "attr50": 0.45,
            "attr51": 0.67,
            "attr52": 0.23,
            "attr53": 0.89,
            "attr54": 0.56,
            "attr55": 0.34,
            "attr56": 0.78,
            "attr57": 0.45,
            "attr58": 0.67,
            "attr59": 0.23,
            "attr60": 0.89,
            "attr61": 0.56,
            "attr62": 0.34,
            "attr63": 0.78,
            "attr64": 0.45,
            "years_before_bankruptcy": 2,
            "company_id": "TEST_COMPANY_001",
        }

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Polish Bankruptcy Prediction API"

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "mlflow_connection" in data
        assert "timestamp" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    @patch("src.api.main.load_model")
    @patch("src.api.main.get_feature_engineer")
    def test_predict_endpoint_success(
        self, mock_feature_engineer, mock_load_model, client, sample_prediction_request
    ):
        """Test successful prediction endpoint"""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        mock_load_model.return_value = mock_model

        # Mock feature engineer
        mock_engineer = Mock()
        mock_engineer.transform.return_value = pd.DataFrame(
            [list(sample_prediction_request.values())[:-1]]
        )
        mock_feature_engineer.return_value = mock_engineer

        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
        assert data["risk_level"] in ["low", "medium", "high"]

    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction endpoint with invalid data"""
        invalid_request = {"attr1": "invalid", "years_before_bankruptcy": 2}

        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_attributes(self, client):
        """Test prediction endpoint with missing attributes"""
        incomplete_request = {"attr1": 0.5, "years_before_bankruptcy": 2}

        response = client.post("/predict", json=incomplete_request)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_years(self, client, sample_prediction_request):
        """Test prediction endpoint with invalid years_before_bankruptcy"""
        sample_prediction_request["years_before_bankruptcy"] = 10  # Invalid value

        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 422  # Validation error

    @patch("src.api.main.load_model")
    @patch("src.api.main.get_feature_engineer")
    def test_batch_predict_endpoint(
        self, mock_feature_engineer, mock_load_model, client, sample_prediction_request
    ):
        """Test batch prediction endpoint"""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_load_model.return_value = mock_model

        # Mock feature engineer
        mock_engineer = Mock()
        mock_engineer.transform.return_value = pd.DataFrame(
            [[0.5] * 64 + [2], [0.6] * 64 + [3]]
        )
        mock_feature_engineer.return_value = mock_engineer

        batch_request = {
            "samples": [sample_prediction_request, sample_prediction_request.copy()]
        }

        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "total_samples" in data
        assert "timestamp" in data
        assert len(data["predictions"]) == 2
        assert data["total_samples"] == 2

    def test_batch_predict_empty_batch(self, client):
        """Test batch prediction with empty batch"""
        empty_batch = {"samples": []}

        response = client.post("/predict/batch", json=empty_batch)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_too_large(self, client, sample_prediction_request):
        """Test batch prediction with too many samples"""
        large_batch = {"samples": [sample_prediction_request] * 1001}  # Exceeds limit

        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 422  # Validation error

    @patch("src.api.main.mlflow")
    def test_models_endpoint_success(self, mock_mlflow, client):
        """Test models endpoint success"""
        # Mock MLflow client
        mock_client = Mock()
        mock_model = Mock()
        mock_model.name = "BankruptcyXgboost"
        mock_model.description = "Test model"

        mock_version = Mock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.creation_timestamp = 1640995200000

        mock_client.search_registered_models.return_value = [mock_model]
        mock_client.get_latest_versions.return_value = [mock_version]
        mock_mlflow.MlflowClient.return_value = mock_client

        response = client.get("/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0

    @patch("src.api.main.mlflow")
    def test_models_endpoint_error(self, mock_mlflow, client):
        """Test models endpoint with MLflow error"""
        # Mock MLflow error
        mock_mlflow.MlflowClient.side_effect = Exception("MLflow connection error")

        response = client.get("/models")
        assert response.status_code == 500


class TestPydanticModels:
    """Test Pydantic model validation"""

    def test_prediction_request_validation(self):
        """Test PredictionRequest validation"""
        valid_data = {f"attr{i}": 0.5 for i in range(1, 65)}
        valid_data["years_before_bankruptcy"] = 2

        # Should not raise exception
        request = PredictionRequest(**valid_data)
        assert request.years_before_bankruptcy == 2

    def test_prediction_request_invalid_years(self):
        """Test PredictionRequest with invalid years"""
        data = {f"attr{i}": 0.5 for i in range(1, 65)}
        data["years_before_bankruptcy"] = 0  # Invalid

        with pytest.raises(ValueError):
            PredictionRequest(**data)

    def test_prediction_request_extreme_values(self):
        """Test PredictionRequest with extreme values"""
        data = {f"attr{i}": 0.5 for i in range(1, 65)}
        data["attr1"] = float("inf")  # Should be converted
        data["years_before_bankruptcy"] = 2

        request = PredictionRequest(**data)
        assert request.attr1 == 1e10  # Converted from infinity

    def test_prediction_response_validation(self):
        """Test PredictionResponse validation"""
        response_data = {
            "prediction": 1,
            "probability": 0.75,
            "risk_level": "high",
            "model_version": "v1.0",
            "timestamp": datetime.now(),
            "features_used": 64,
        }

        response = PredictionResponse(**response_data)
        assert response.prediction == 1
        assert response.probability == 0.75
        assert response.risk_level == "high"

    def test_prediction_response_invalid_probability(self):
        """Test PredictionResponse with invalid probability"""
        response_data = {
            "prediction": 1,
            "probability": 1.5,  # Invalid
            "risk_level": "high",
            "model_version": "v1.0",
            "timestamp": datetime.now(),
            "features_used": 64,
        }

        with pytest.raises(ValueError):
            PredictionResponse(**response_data)

    def test_prediction_response_invalid_prediction(self):
        """Test PredictionResponse with invalid prediction"""
        response_data = {
            "prediction": 2,  # Invalid
            "probability": 0.75,
            "risk_level": "high",
            "model_version": "v1.0",
            "timestamp": datetime.now(),
            "features_used": 64,
        }

        with pytest.raises(ValueError):
            PredictionResponse(**response_data)
