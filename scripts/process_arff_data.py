#!/usr/bin/env python3
"""
Script pour traiter les données ARFF du Polish Bankruptcy Dataset
"""

import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.io import arff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-dir", default="data/raw", help="Répertoire contenant les fichiers ARFF"
)
@click.option("--output-dir", default="data/processed", help="Répertoire de sortie")
def process_arff_files(input_dir, output_dir):
    """Traite tous les fichiers ARFF et les convertit en CSV"""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Traitement des fichiers ARFF depuis {input_path}")

    # Fichiers ARFF attendus
    arff_files = ["1year.arff", "2year.arff", "3year.arff", "4year.arff", "5year.arff"]

    combined_data = []

    for arff_file in arff_files:
        file_path = input_path / arff_file

        if not file_path.exists():
            logger.warning(f"Fichier manquant: {file_path}")
            continue

        logger.info(f"Traitement de {arff_file}")

        try:
            # Charger le fichier ARFF
            data, meta = arff.loadarff(str(file_path))

            # Convertir en DataFrame
            df = pd.DataFrame(data)

            # Décoder les bytes en string pour les colonnes string
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].str.decode("utf-8")
                    except:
                        pass

            # Ajouter la colonne année
            year = arff_file.replace("year.arff", "")
            df["years_before_bankruptcy"] = int(year)

            # Nettoyer les noms de colonnes
            df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

            # Convertir la colonne target
            if "class" in df.columns:
                df["bankruptcy"] = (df["class"] == "1").astype(int)
                df = df.drop("class", axis=1)

            logger.info(f"  - {len(df)} échantillons, {len(df.columns)} features")
            logger.info(f"  - Taux de faillite: {df['bankruptcy'].mean():.3f}")

            # Sauvegarder individuellement
            individual_output = output_path / f"bankruptcy_{year}year.csv"
            df.to_csv(individual_output, index=False)

            combined_data.append(df)

        except Exception as e:
            logger.error(f"Erreur lors du traitement de {arff_file}: {e}")
            continue

    # Combiner toutes les données
    if combined_data:
        logger.info("Combinaison de toutes les années...")

        combined_df = pd.concat(combined_data, ignore_index=True)

        # Statistiques globales
        logger.info(f"Dataset combiné:")
        logger.info(f"  - Total échantillons: {len(combined_df)}")
        logger.info(f"  - Total features: {len(combined_df.columns)}")
        logger.info(
            f"  - Taux de faillite global: {combined_df['bankruptcy'].mean():.3f}"
        )
        logger.info(f"  - Répartition par année:")

        year_stats = (
            combined_df.groupby("years_before_bankruptcy")
            .agg({"bankruptcy": ["count", "mean"]})
            .round(3)
        )
        print(year_stats)

        # Sauvegarder le dataset combiné
        combined_output = output_path / "bankruptcy_combined.csv"
        combined_df.to_csv(combined_output, index=False)

        logger.info(f"Dataset combiné sauvegardé: {combined_output}")

        # Créer les splits train/val/test
        create_data_splits(combined_df, output_path)

    else:
        logger.error("Aucune donnée traitée avec succès")


def create_data_splits(df, output_path):
    """Crée les splits train/validation/test"""

    logger.info("Création des splits train/val/test...")

    # Stratification par années et target
    from sklearn.model_selection import train_test_split

    # Split initial : 80% train+val, 20% test
    train_val, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df[["bankruptcy", "years_before_bankruptcy"]],
        random_state=42,
    )

    # Split train/val : 75% train, 25% val (du 80% initial)
    train, val = train_test_split(
        train_val,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 du total
        stratify=train_val[["bankruptcy", "years_before_bankruptcy"]],
        random_state=42,
    )

    # Sauvegarder les splits
    train.to_csv(output_path / "train.csv", index=False)
    val.to_csv(output_path / "val.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)

    logger.info(f"Splits créés:")
    logger.info(f"  - Train: {len(train)} ({len(train)/len(df):.1%})")
    logger.info(f"  - Val: {len(val)} ({len(val)/len(df):.1%})")
    logger.info(f"  - Test: {len(test)} ({len(test)/len(df):.1%})")

    # Vérifier la distribution
    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        bankruptcy_rate = split_df["bankruptcy"].mean()
        logger.info(f"  - {split_name} bankruptcy rate: {bankruptcy_rate:.3f}")


if __name__ == "__main__":
    process_arff_files()
