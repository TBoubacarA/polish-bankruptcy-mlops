# 🏦 Polish Bankruptcy Prediction MLOps

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8.1-orange.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-containerized-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **End-to-end MLOps system for predicting Polish company bankruptcies using machine learning and temporal financial data.**

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [📊 Dataset](#-dataset)
- [🤖 Models](#-models)
- [🔧 API Usage](#-api-usage)
- [🧪 Testing](#-testing)
- [📈 MLflow Tracking](#-mlflow-tracking)
- [🐳 Docker Deployment](#-docker-deployment)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)

## 🎯 Overview

This project implements a comprehensive **MLOps pipeline** for predicting company bankruptcies using Polish financial data spanning 5 years. It combines traditional machine learning with modern MLOps practices to deliver a production-ready system suitable for financial institutions and risk assessment applications.

### Business Context
- **Objective**: Predict company bankruptcy 1-5 years in advance
- **Use Cases**: Credit risk assessment, investment decisions, early warning systems
- **Target Users**: Banks, financial institutions, investors, risk analysts

### Technical Highlights
- **End-to-end MLOps pipeline** with experiment tracking
- **Multiple ML models** (XGBoost, LightGBM, Random Forest, Ensembles)
- **RESTful API** for real-time predictions
- **Containerized deployment** with Docker Compose
- **Production monitoring** and model interpretability

## ✨ Features

### 🔬 Machine Learning
- **Advanced feature engineering** with financial ratios and composite scores
- **Multiple model types** with automatic hyperparameter tuning
- **Ensemble methods** (Voting, Stacking) for improved accuracy
- **Cross-validation** and robust evaluation metrics
- **Model interpretability** with SHAP values

### 🛠️ MLOps Infrastructure
- **Experiment tracking** with MLflow (model registry, metrics, artifacts)
- **Automated pipelines** for training and evaluation
- **Containerized services** for scalable deployment
- **API endpoints** for real-time inference
- **Data quality monitoring** with Great Expectations

### 📊 Monitoring & Observability
- **Model performance tracking** with key business metrics
- **Data drift detection** capabilities
- **Feature importance analysis** and model explainability
- **Health checks** and service monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Training       │    │  Serving        │
│                 │    │  Pipeline       │    │  Layer          │
│ • Raw data      │───▶│ • Feature eng.  │───▶│ • FastAPI       │
│ • Processed     │    │ • Model training│    │ • Predictions   │
│ • Validation    │    │ • Evaluation    │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Storage &      │    │   MLflow        │    │  Infrastructure │
│  Database       │    │   Tracking      │    │                 │
│                 │    │                 │    │ • Docker        │
│ • PostgreSQL    │    │ • Experiments   │    │ • Compose       │
│ • Artifacts     │    │ • Model Registry│    │ • Networking    │
│ • Metadata      │    │ • Metrics       │    │ • Volumes       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Git

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/polish-bankruptcy-mlops.git
cd polish-bankruptcy-mlops
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Prepare Data
```bash
# Download and process Polish bankruptcy dataset
python scripts/prepare_data.py
```

### 4. Train Models
```bash
# Train individual model
python scripts/train_pipeline.py --model-type xgboost

# Train ensemble of models
python scripts/train_pipeline.py --run-ensemble
```

### 5. Launch Services
```bash
# Start all services (PostgreSQL, MLflow, API)
docker-compose up -d

# View MLflow UI
# Navigate to http://localhost:5000
```

## 📊 Dataset

### Polish Companies Bankruptcy Dataset
- **Size**: 43,405 samples across 5 temporal datasets
- **Features**: 64 financial attributes (financial ratios)
- **Target**: Binary classification (0=Healthy, 1=Bankrupt)
- **Temporal Range**: 1-5 years before bankruptcy event

### Data Distribution
| Years Before Bankruptcy | Samples | Bankruptcy Rate |
|-------------------------|---------|----------------|
| 1 year                  | 7,027   | 8.9%          |
| 2 years                 | 10,173  | 5.5%          |
| 3 years                 | 10,503  | 4.4%          |
| 4 years                 | 9,792   | 3.9%          |
| 5 years                 | 5,910   | 2.9%          |

### Feature Engineering
- **Financial ratios**: Liquidity, profitability, leverage metrics
- **Composite scores**: Altman Z-Score approximation
- **Temporal features**: Years before bankruptcy indicator
- **Derived metrics**: Custom risk scores and financial health indicators

## 🤖 Models

### Individual Models
| Model | Type | Key Parameters | Typical AUC |
|-------|------|----------------|-------------|
| **XGBoost** | Gradient Boosting | max_depth=6, learning_rate=0.1 | 0.974 |
| **LightGBM** | Gradient Boosting | num_leaves=50, learning_rate=0.05 | 0.971 |
| **Random Forest** | Ensemble | n_estimators=200, max_depth=15 | 0.965 |
| **Logistic Regression** | Linear | C=1.0, penalty='l2' | 0.895 |

### Ensemble Methods
- **Voting Classifier**: Weighted combination (XGBoost: 40%, LightGBM: 40%, RF: 20%)
- **Stacking**: Meta-learner with logistic regression and 5-fold CV
- **Performance**: Ensemble AUC ~0.966, balanced precision/recall

### Evaluation Metrics
- **Primary**: ROC-AUC (handles class imbalance)
- **Business**: Precision, Recall, F1-Score
- **Financial**: False Positive Rate (Type I error), False Negative Rate (Type II error)
- **Interpretability**: SHAP feature importance

## 🔧 API Usage

### Start API Server
```bash
# Using Docker (recommended)
docker-compose up api

# Or directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Prediction Endpoint
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "attr1": 0.56,
    "attr2": 0.12,
    "attr3": 0.89,
    ...
    "years_before_bankruptcy": 2
  }'

# Response
{
  "prediction": 0,
  "probability": 0.15,
  "risk_level": "low",
  "model_version": "BankruptcyXgboost_v2",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Health Check
```bash
curl http://localhost:8000/health
# {"status": "healthy", "model_loaded": true}
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests

# With coverage
pytest --cov=src/ --cov-report=html
```

## 📈 MLflow Tracking

### Access MLflow UI
```bash
# Local development
mlflow ui --port 5000

# Or via Docker
docker-compose up mlflow
# Navigate to http://localhost:5000
```

### Experiment Tracking
- **Automatic logging**: Parameters, metrics, artifacts
- **Model registry**: Versioned model storage
- **Comparison tools**: Run comparison and model selection
- **Deployment**: Model staging and production promotion

### Key Metrics Tracked
- Model performance (AUC, Precision, Recall)
- Feature importance rankings
- Cross-validation scores
- Training time and resource usage
- Hyperparameter values

## 🐳 Docker Deployment

### Services Architecture
```yaml
services:
  postgres:     # Database for MLflow backend
  pgadmin:      # Database administration
  mlflow:       # Experiment tracking server
  api:          # FastAPI prediction service
```

### Production Deployment
```bash
# Start all services
docker-compose up -d

# Scale API service
docker-compose up -d --scale api=3

# View logs
docker-compose logs -f api

# Health checks
curl http://localhost:8000/health
curl http://localhost:5000/health
```

### Environment Variables
```bash
# Required environment variables
DATABASE_URL=postgresql://user:pass@host:port/db
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=postgresql://...
MLFLOW_DEFAULT_ARTIFACT_ROOT=./artifacts
```

## 📚 Documentation

### Project Structure
```
polish-bankruptcy-mlops/
├── src/                    # Source code
│   ├── models/            # ML models and training
│   ├── features/          # Feature engineering
│   ├── api/               # FastAPI application
│   ├── data/              # Data processing utilities
│   ├── monitoring/        # Model monitoring
│   └── utils/             # Helper functions
├── data/                  # Data storage
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned and processed data
├── config/               # Configuration files
├── scripts/              # Training and utility scripts
├── tests/                # Test suite
├── infrastructure/       # Docker and deployment
└── docs/                 # Documentation
```

### Key Components
- **`BankruptcyClassifier`**: Main model class supporting multiple algorithms
- **`EnsembleBankruptcyClassifier`**: Ensemble methods implementation
- **`FinancialRatiosEngineer`**: Feature engineering pipeline
- **Training Pipeline**: Automated model training with MLflow integration

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
```

### Pull Request Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- **Black** code formatting
- **isort** import sorting
- **flake8** linting
- **pytest** for testing
- **Type hints** encouraged
- **Docstrings** for all public functions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ If this project helps you, please consider giving it a star!**