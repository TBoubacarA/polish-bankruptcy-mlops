"""
Unit tests for bankruptcy classifier models
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models.bankruptcy_classifier import (BankruptcyClassifier,
                                              EnsembleBankruptcyClassifier)


class TestBankruptcyClassifier:
    """Test cases for BankruptcyClassifier"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, n_redundant=0, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"attr{i+1}" for i in range(20)])
        y_series = pd.Series(y)
        return X_df, y_series

    @pytest.fixture
    def validation_data(self):
        """Create validation dataset"""
        X, y = make_classification(
            n_samples=200, n_features=20, n_classes=2, n_redundant=0, random_state=123
        )
        X_df = pd.DataFrame(X, columns=[f"attr{i+1}" for i in range(20)])
        y_series = pd.Series(y)
        return X_df, y_series

    def test_xgboost_initialization(self):
        """Test XGBoost classifier initialization"""
        classifier = BankruptcyClassifier(model_type="xgboost")
        assert classifier.model_type == "xgboost"
        assert hasattr(classifier, "model")

    def test_lightgbm_initialization(self):
        """Test LightGBM classifier initialization"""
        classifier = BankruptcyClassifier(model_type="lightgbm")
        assert classifier.model_type == "lightgbm"
        assert hasattr(classifier, "model")

    def test_random_forest_initialization(self):
        """Test Random Forest classifier initialization"""
        classifier = BankruptcyClassifier(model_type="random_forest")
        assert classifier.model_type == "random_forest"
        assert hasattr(classifier, "model")

    def test_unsupported_model_type(self):
        """Test error handling for unsupported model type"""
        with pytest.raises(ValueError):
            BankruptcyClassifier(model_type="unsupported_model")

    def test_fit_without_validation(self, sample_data):
        """Test model fitting without validation data"""
        X, y = sample_data
        classifier = BankruptcyClassifier(model_type="random_forest")

        # Should not raise any exception
        classifier.fit(X, y)
        assert classifier.feature_names is not None

    def test_fit_with_validation(self, sample_data, validation_data):
        """Test model fitting with validation data"""
        X_train, y_train = sample_data
        X_val, y_val = validation_data

        classifier = BankruptcyClassifier(model_type="xgboost")
        classifier.fit(X_train, y_train, X_val, y_val)

        assert classifier.feature_names is not None
        assert len(classifier.feature_names) == X_train.shape[1]

    def test_predict(self, sample_data):
        """Test prediction functionality"""
        X, y = sample_data
        classifier = BankruptcyClassifier(model_type="random_forest")
        classifier.fit(X, y)

        predictions = classifier.predict(X[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_proba(self, sample_data):
        """Test probability prediction"""
        X, y = sample_data
        classifier = BankruptcyClassifier(model_type="random_forest")
        classifier.fit(X, y)

        probabilities = classifier.predict_proba(X[:10])
        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_evaluate(self, sample_data):
        """Test model evaluation"""
        X, y = sample_data
        classifier = BankruptcyClassifier(model_type="random_forest")
        classifier.fit(X[:800], y[:800])

        metrics = classifier.evaluate(X[800:], y[800:])

        expected_metrics = ["roc_auc", "precision", "recall", "f1", "accuracy"]
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

    def test_get_feature_importance(self, sample_data):
        """Test feature importance extraction"""
        X, y = sample_data
        classifier = BankruptcyClassifier(model_type="random_forest")
        classifier.fit(X, y)

        importance = classifier.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(isinstance(imp, float) for imp, val in importance.items())

    def test_cross_validate(self, sample_data):
        """Test cross-validation"""
        X, y = sample_data
        classifier = BankruptcyClassifier(model_type="random_forest")

        cv_results = classifier.cross_validate(X, y, cv=3)

        assert "cv_mean" in cv_results
        assert "cv_std" in cv_results
        assert 0 <= cv_results["cv_mean"] <= 1
        assert cv_results["cv_std"] >= 0


class TestEnsembleBankruptcyClassifier:
    """Test cases for EnsembleBankruptcyClassifier"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, n_redundant=0, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"attr{i+1}" for i in range(20)])
        y_series = pd.Series(y)
        return X_df, y_series

    @pytest.fixture
    def validation_data(self):
        """Create validation dataset"""
        X, y = make_classification(
            n_samples=200, n_features=20, n_classes=2, n_redundant=0, random_state=123
        )
        X_df = pd.DataFrame(X, columns=[f"attr{i+1}" for i in range(20)])
        y_series = pd.Series(y)
        return X_df, y_series

    def test_ensemble_initialization(self):
        """Test ensemble classifier initialization"""
        ensemble = EnsembleBankruptcyClassifier()
        assert hasattr(ensemble, "models")
        assert hasattr(ensemble, "model_weights")
        assert hasattr(ensemble, "ensemble_model")

    @patch("src.models.bankruptcy_classifier.BankruptcyClassifier")
    def test_fit(self, mock_classifier_class, sample_data, validation_data):
        """Test ensemble fitting"""
        X_train, y_train = sample_data
        X_val, y_val = validation_data

        # Mock individual classifiers
        mock_classifier = Mock()
        mock_classifier.fit.return_value = mock_classifier
        mock_classifier.predict_proba.return_value = np.random.rand(len(X_train), 2)
        mock_classifier_class.return_value = mock_classifier

        ensemble = EnsembleBankruptcyClassifier()
        ensemble.fit(X_train, y_train, X_val, y_val)

        # Check that individual models were fitted
        assert len(ensemble.models) > 0

    def test_evaluate_integration(self, sample_data):
        """Integration test for ensemble evaluation"""
        X, y = sample_data

        # Use smaller dataset for faster testing
        X_small = X[:200]
        y_small = y[:200]

        ensemble = EnsembleBankruptcyClassifier()
        ensemble.fit(X_small[:150], y_small[:150], X_small[150:], y_small[150:])

        # Test evaluation
        results = ensemble.evaluate(X_small[150:], y_small[150:])

        # Check that we get results for individual models and ensemble
        assert "ensemble" in results
        expected_metrics = ["roc_auc", "precision", "recall", "f1", "accuracy"]

        for model_name, metrics in results.items():
            for metric in expected_metrics:
                assert metric in metrics
                assert 0 <= metrics[metric] <= 1


class TestModelIntegration:
    """Integration tests for model components"""

    def test_model_pipeline_integration(self):
        """Test complete model pipeline"""
        # Create synthetic data
        X, y = make_classification(
            n_samples=500, n_features=10, n_classes=2, n_redundant=0, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"attr{i+1}" for i in range(10)])
        y_series = pd.Series(y)

        # Split data
        train_idx = int(0.7 * len(X_df))
        X_train, y_train = X_df[:train_idx], y_series[:train_idx]
        X_test, y_test = X_df[train_idx:], y_series[train_idx:]

        # Train model
        classifier = BankruptcyClassifier(model_type="random_forest")
        classifier.fit(X_train, y_train)

        # Make predictions
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)

        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)

        # Assertions
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert metrics["roc_auc"] > 0.5  # Should be better than random

    def test_model_persistence_attributes(self, sample_data):
        """Test that model attributes are properly set after training"""
        X, y = sample_data
        classifier = BankruptcyClassifier(model_type="xgboost")

        # Before fitting
        assert classifier.feature_names is None

        # After fitting
        classifier.fit(X, y)
        assert classifier.feature_names is not None
        assert len(classifier.feature_names) == X.shape[1]
        assert classifier.model is not None
