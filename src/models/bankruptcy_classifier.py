#!/usr/bin/env python3
"""
Modèles de classification pour la prédiction de faillite
"""

import logging
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import mlflow

logger = logging.getLogger(__name__)


class BankruptcyClassifier(BaseEstimator, ClassifierMixin):
    """Classificateur de faillite avec support MLflow"""

    def __init__(self, model_type="xgboost", **kwargs):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.metrics = {}
        self.shap_explainer = None
        self.model_params = kwargs

        # Initialiser le modèle selon le type
        self._init_model()

    def _init_model(self):
        """Initialise le modèle selon le type"""

        if self.model_type == "xgboost":
            default_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
                "eval_metric": "auc",
                "objective": "binary:logistic",
            }
            default_params.update(self.model_params)
            self.model = xgb.XGBClassifier(**default_params)

        elif self.model_type == "lightgbm":
            default_params = {
                "num_leaves": 50,
                "learning_rate": 0.05,
                "n_estimators": 300,
                "max_depth": 10,
                "min_child_samples": 30,
                "random_state": 42,
                "verbosity": -1,
                "objective": "binary",
            }
            default_params.update(self.model_params)
            self.model = lgb.LGBMClassifier(**default_params)

        elif self.model_type == "random_forest":
            default_params = {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced",
            }
            default_params.update(self.model_params)
            self.model = RandomForestClassifier(**default_params)

        elif self.model_type == "logistic":
            default_params = {
                "random_state": 42,
                "max_iter": 1000,
                "class_weight": "balanced",
            }
            default_params.update(self.model_params)
            self.model = LogisticRegression(**default_params)

        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")

    def fit(self, X, y, X_val=None, y_val=None):
        """Entraîne le modèle"""

        self.feature_names = list(X.columns) if hasattr(X, "columns") else None

        # Entraînement spécifique selon le type
        if self.model_type == "xgboost" and X_val is not None:
            # Early stopping pour XGBoost
            self.model.fit(
                X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False
            )
        elif self.model_type == "lightgbm" and X_val is not None:
            # Early stopping pour LightGBM
            self.model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
        else:
            self.model.fit(X, y)

        return self

    def predict(self, X):
        """Prédictions binaires"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Probabilités de prédiction"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Évalue les performances du modèle"""

        # Prédictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        # Métriques de classification
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "accuracy": (y_pred == y_test).mean(),
        }

        # Métriques business
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update(
            {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            }
        )

        # Gini coefficient
        metrics["gini"] = 2 * metrics["roc_auc"] - 1

        self.metrics = metrics
        return metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """Retourne l'importance des features"""

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            return {}

        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}

    def explain_prediction(self, X, max_display=10):
        """Explication SHAP des prédictions"""

        if self.shap_explainer is None:
            # Créer l'explainer selon le type de modèle
            if self.model_type in ["xgboost", "lightgbm", "random_forest"]:
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                self.shap_explainer = shap.LinearExplainer(self.model, X)

        # Calculer les valeurs SHAP
        shap_values = self.shap_explainer.shap_values(X)

        # Pour les modèles binaires, prendre la classe positive
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return {
            "shap_values": shap_values,
            "base_value": self.shap_explainer.expected_value,
            "feature_names": self.feature_names,
        }

    def cross_validate(self, X, y, cv=5) -> Dict[str, Any]:
        """Validation croisée"""

        cv_scores = cross_val_score(
            self.model,
            X,
            y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring="roc_auc",
        )

        return {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }

    def save_model(self, filepath: str):
        """Sauvegarde le modèle"""
        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
            },
            filepath,
        )

    @classmethod
    def load_model(cls, filepath: str):
        """Charge un modèle sauvegardé"""
        data = joblib.load(filepath)

        classifier = cls(model_type=data["model_type"])
        classifier.model = data["model"]
        classifier.feature_names = data["feature_names"]
        classifier.metrics = data.get("metrics", {})

        return classifier


class EnsembleBankruptcyClassifier:
    """Ensemble de modèles pour la prédiction de faillite"""

    def __init__(self, models_config=None):
        if models_config is None:
            models_config = {
                "xgboost": {"weight": 0.4},
                "lightgbm": {"weight": 0.4},
                "random_forest": {"weight": 0.2},
            }

        self.models_config = models_config
        self.models = {}
        self.ensemble_model = None
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Entraîne l'ensemble de modèles"""

        self.feature_names = (
            list(X_train.columns) if hasattr(X_train, "columns") else None
        )

        # Entraîner chaque modèle individuellement
        base_estimators = []

        for model_type, config in self.models_config.items():
            logger.info(f"Entraînement du modèle {model_type}")

            model = BankruptcyClassifier(model_type=model_type)
            model.fit(X_train, y_train, X_val, y_val)

            self.models[model_type] = model

            # Ajouter au voting classifier
            weight = config.get("weight", 1.0)
            base_estimators.append((model_type, model.model))

        # Créer le voting classifier
        weights = [self.models_config[name]["weight"] for name, _ in base_estimators]

        self.ensemble_model = VotingClassifier(
            estimators=base_estimators, voting="soft", weights=weights
        )

        self.ensemble_model.fit(X_train, y_train)

        return self

    def predict(self, X):
        """Prédictions de l'ensemble"""
        return self.ensemble_model.predict(X)

    def predict_proba(self, X):
        """Probabilités de l'ensemble"""
        return self.ensemble_model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Évalue l'ensemble et les modèles individuels"""

        results = {}

        # Évaluer l'ensemble
        y_pred_ensemble = self.predict(X_test)
        y_proba_ensemble = self.predict_proba(X_test)[:, 1]

        results["ensemble"] = {
            "roc_auc": roc_auc_score(y_test, y_proba_ensemble),
            "precision": precision_score(y_test, y_pred_ensemble),
            "recall": recall_score(y_test, y_pred_ensemble),
            "f1_score": f1_score(y_test, y_pred_ensemble),
        }

        # Évaluer chaque modèle individuel
        for model_name, model in self.models.items():
            results[model_name] = model.evaluate(X_test, y_test)

        return results
