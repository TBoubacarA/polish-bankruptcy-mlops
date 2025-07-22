#!/usr/bin/env python3
"""
Pipeline d'entraînement pour la prédiction de faillite
"""

import logging
import os
import sys
from pathlib import Path

import click
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlflow

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from features.financial_ratios import FinancialRatiosEngineer
from models.bankruptcy_classifier import (BankruptcyClassifier,
                                          EnsembleBankruptcyClassifier)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-path",
    default="data/processed/bankruptcy_combined.csv",
    help="Chemin vers les données",
)
@click.option(
    "--config-path",
    default="config/model_config.yaml",
    help="Chemin vers la configuration",
)
@click.option("--model-type", default="xgboost", help="Type de modèle à entraîner")
@click.option(
    "--experiment-name",
    default="polish_bankruptcy_prediction",
    help="Nom de l'expérience MLflow",
)
@click.option("--run-ensemble", is_flag=True, help="Entraîner un ensemble de modèles")
def train_model(data_path, config_path, model_type, experiment_name, run_ensemble):
    """Pipeline d'entraînement des modèles de faillite"""

    # Configuration MLflow
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(experiment_name)

    # Charger la configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Charger les données
    logger.info(f"Chargement des données depuis {data_path}")
    data = pd.read_csv(data_path)

    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Taux de faillite: {data['bankruptcy'].mean():.3f}")

    # Feature engineering
    logger.info("Feature engineering...")
    feature_engineer = FinancialRatiosEngineer()
    data_engineered = feature_engineer.transform(data)

    # Séparer features et target
    target_col = "bankruptcy"
    feature_cols = [col for col in data_engineered.columns if col != target_col]

    X = data_engineered[feature_cols]
    y = data_engineered[target_col]

    logger.info(f"Nombre de features: {len(feature_cols)}")

    # Split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    if run_ensemble:
        # Entraîner un ensemble de modèles
        with mlflow.start_run(
            run_name=f"ensemble_bankruptcy_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
        ):
            # Log des paramètres
            mlflow.log_params(
                {
                    "model_type": "ensemble",
                    "data_shape": data.shape,
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "test_size": len(X_test),
                    "bankruptcy_rate": data["bankruptcy"].mean(),
                }
            )

            # Entraîner l'ensemble
            ensemble = EnsembleBankruptcyClassifier()
            ensemble.fit(X_train, y_train, X_val, y_val)

            # Évaluer
            results = ensemble.evaluate(X_test, y_test)

            # Log des métriques
            for model_name, metrics in results.items():
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)

            # Sauvegarder le modèle
            mlflow.sklearn.log_model(
                ensemble.ensemble_model,
                "ensemble_model",
                registered_model_name="BankruptcyEnsemble",
            )

            logger.info("Ensemble entraîné et sauvegardé")
            logger.info(f"AUC Ensemble: {results['ensemble']['roc_auc']:.4f}")

    else:
        # Entraîner un modèle simple
        with mlflow.start_run(
            run_name=f"{model_type}_bankruptcy_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
        ):
            # Paramètres du modèle
            model_params = config["models"].get(model_type, {}).get("parameters", {})

            # Log des paramètres
            mlflow.log_params(
                {
                    "model_type": model_type,
                    "data_shape": data.shape,
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "test_size": len(X_test),
                    "bankruptcy_rate": data["bankruptcy"].mean(),
                    **model_params,
                }
            )

            # Créer et entraîner le modèle
            model = BankruptcyClassifier(model_type=model_type, **model_params)
            model.fit(X_train, y_train, X_val, y_val)

            # Évaluation
            metrics = model.evaluate(X_test, y_test)

            # Log des métriques
            mlflow.log_metrics(metrics)

            # Feature importance
            importance = model.get_feature_importance()

            # Log top 10 features
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]
            for i, (feature, imp) in enumerate(top_features):
                mlflow.log_metric(f"feature_importance_{i+1}_{feature[:20]}", imp)

            # Validation croisée
            cv_results = model.cross_validate(X_train, y_train)
            mlflow.log_metrics(
                {
                    "cv_auc_mean": cv_results["cv_mean"],
                    "cv_auc_std": cv_results["cv_std"],
                }
            )

            # Sauvegarder le modèle
            if model_type == "xgboost":
                mlflow.xgboost.log_model(
                    model.model,
                    model_type,
                    registered_model_name=f"Bankruptcy{model_type.title()}",
                )
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(
                    model.model,
                    model_type,
                    registered_model_name=f"Bankruptcy{model_type.title()}",
                )
            else:
                mlflow.sklearn.log_model(
                    model.model,
                    model_type,
                    registered_model_name=f"Bankruptcy{model_type.title()}",
                )

            logger.info(f"Modèle {model_type} entraîné et sauvegardé")
            logger.info(f"AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")


if __name__ == "__main__":
    train_model()
