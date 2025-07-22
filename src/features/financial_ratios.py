"""
Feature engineering pour les ratios financiers
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FinancialRatiosEngineer(BaseEstimator, TransformerMixin):
    """Transformateur pour créer des ratios financiers avancés"""

    def __init__(self):
        self.feature_names = None

    def fit(self, X, y=None):
        """Fit du transformateur"""
        return self

    def transform(self, X):
        """Transformation des features"""

        df = X.copy()

        # Ratios dérivés (si les colonnes de base existent)
        self._create_liquidity_ratios(df)
        self._create_profitability_ratios(df)
        self._create_leverage_ratios(df)
        self._create_efficiency_ratios(df)
        self._create_risk_scores(df)

        # Nettoyage des valeurs infinies/NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())

        return df

    def _create_liquidity_ratios(self, df):
        """Créer des ratios de liquidité avancés"""

        # Score de liquidité combiné
        liquidity_cols = [
            col
            for col in df.columns
            if "current" in col.lower() or "quick" in col.lower()
        ]
        if len(liquidity_cols) >= 2:
            df["liquidity_score"] = df[liquidity_cols].mean(axis=1)

    def _create_profitability_ratios(self, df):
        """Créer des ratios de rentabilité avancés"""

        # Score de rentabilité
        profitability_cols = [
            col
            for col in df.columns
            if any(term in col.lower() for term in ["roa", "roe", "margin"])
        ]
        if len(profitability_cols) >= 2:
            df["profitability_score"] = df[profitability_cols].mean(axis=1)

    def _create_leverage_ratios(self, df):
        """Créer des ratios de levier avancés"""

        # Score de risque de levier
        leverage_cols = [col for col in df.columns if "debt" in col.lower()]
        if len(leverage_cols) >= 1:
            df["leverage_risk"] = df[leverage_cols].mean(axis=1)

    def _create_efficiency_ratios(self, df):
        """Créer des ratios d'efficacité"""

        # Score d'efficacité
        efficiency_cols = [col for col in df.columns if "turnover" in col.lower()]
        if len(efficiency_cols) >= 1:
            df["efficiency_score"] = df[efficiency_cols].mean(axis=1)

    def _create_risk_scores(self, df):
        """Créer des scores de risque composites"""

        # Z-Score d'Altman approximatif (si possible)
        required_cols = [
            "working_capital_total_assets",
            "retained_earnings_total_assets",
            "ebit_total_assets",
            "market_value_equity_total_liabilities",
            "sales_total_assets",
        ]

        available_cols = [col for col in required_cols if col in df.columns]

        if len(available_cols) >= 3:
            # Score de risque approximatif basé sur les colonnes disponibles
            df["altman_z_approx"] = (
                df.get("working_capital_total_assets", 0) * 1.2
                + df.get("retained_earnings_total_assets", 0) * 1.4
                + df.get("ebit_total_assets", 0) * 3.3
                + df.get("market_value_equity_total_liabilities", 0) * 0.6
                + df.get("sales_total_assets", 0) * 1.0
            )
