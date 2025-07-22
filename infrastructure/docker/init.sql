-- Initialization script for PostgreSQL database
-- This script sets up the MLflow database and basic bankruptcy prediction tables

-- Create MLflow database
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;

-- Connect to MLflow database and create extensions if needed
\c mlflow;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create bankruptcy database for application data
\c postgres;
CREATE DATABASE bankruptcy_db;
GRANT ALL PRIVILEGES ON DATABASE bankruptcy_db TO postgres;

-- Connect to bankruptcy database and create tables
\c bankruptcy_db;

-- Create companies table for storing company information
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    registration_id VARCHAR(100) UNIQUE,
    industry VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table for storing model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id),
    model_version VARCHAR(100) NOT NULL,
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    probability FLOAT NOT NULL CHECK (probability >= 0 AND probability <= 1),
    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'medium', 'high')),
    features JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(100),
    years_before_bankruptcy INTEGER
);

-- Create indexes for better performance
CREATE INDEX idx_predictions_company_id ON predictions(company_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
CREATE INDEX idx_predictions_model_version ON predictions(model_version);
CREATE INDEX idx_predictions_risk_level ON predictions(risk_level);

-- Create monitoring table for tracking model performance
CREATE TABLE IF NOT EXISTS model_monitoring (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    evaluation_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for monitoring queries
CREATE INDEX idx_monitoring_model_date ON model_monitoring(model_name, evaluation_date);

-- Insert sample data (optional)
INSERT INTO companies (name, registration_id, industry) VALUES
    ('Sample Company A', 'REG001', 'Manufacturing'),
    ('Sample Company B', 'REG002', 'Services'),
    ('Sample Company C', 'REG003', 'Retail');

-- Create user for application (with limited privileges)
CREATE USER bankruptcy_app WITH ENCRYPTED PASSWORD 'bankruptcy_app_password';
GRANT CONNECT ON DATABASE bankruptcy_db TO bankruptcy_app;
GRANT USAGE ON SCHEMA public TO bankruptcy_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO bankruptcy_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO bankruptcy_app;