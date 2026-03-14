from pathlib import Path

import pandas as pd
import streamlit as st


COLUMN_DESCRIPTIONS = {
    "id": "Identifiant unique du patient.",
    "gender": "Genre du patient.",
    "age": "Âge du patient.",
    "hypertension": "Indicateur d'hypertension (0 = non, 1 = oui).",
    "heart_disease": "Maladie cardiaque (0 = non, 1 = oui).",
    "ever_married": "Le patient a-t-il déjà été marié ?",
    "work_type": "Type d'emploi du patient.",
    "Residence_type": "Zone de résidence (Rural ou Urban).",
    "avg_glucose_level": "Niveau moyen de glucose dans le sang.",
    "bmi": "Indice de masse corporelle.",
    "smoking_status": "Statut tabagique du patient.",
    "stroke": "Variable cible (0 = pas d'AVC, 1 = AVC).",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "healthcare-dataset-stroke-data.csv"
    )
    df = pd.read_csv(dataset_path, na_values=["N/A"])
    return df


def get_column_descriptions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Colonne": list(COLUMN_DESCRIPTIONS.keys()),
            "Description": list(COLUMN_DESCRIPTIONS.values()),
        }
    )
