from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ModelName = Literal["Logistic Regression", "Random Forest"]


class BiasAuditResult(TypedDict):
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    sensitive: np.ndarray
    acc: float
    prec: float
    rec: float


class CachedModelOutputs(TypedDict):
    y_test: np.ndarray
    sensitive: np.ndarray
    y_proba_by_model: dict[ModelName, np.ndarray]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def _build_model(model_name: ModelName):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1200, random_state=42)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=42)
    raise ValueError(f"Unknown model name: {model_name}")


@st.cache_data(show_spinner=False)
def run_bias_audit_model(
    input_df: pd.DataFrame,
    model_name: ModelName,
    sensitive_attr: str,
) -> BiasAuditResult:
    if sensitive_attr not in input_df.columns:
        raise KeyError(f"Missing sensitive attribute column: {sensitive_attr}")

    model_df = input_df.dropna(subset=["stroke", sensitive_attr]).copy()
    X = model_df.drop(columns=["stroke", "id"], errors="ignore")
    y = model_df["stroke"].astype(int)

    preprocessor = _build_preprocessor(X)
    model = _build_model(model_name)

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    return {
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "sensitive": X_test[sensitive_attr].to_numpy(),
        "acc": float(accuracy_score(y_test, y_pred)),
        "prec": float(precision_score(y_test, y_pred, zero_division=0)),
        "rec": float(recall_score(y_test, y_pred, zero_division=0)),
    }


@st.cache_data(show_spinner=False)
def get_cached_model_outputs(
    input_df: pd.DataFrame,
    sensitive_attr: str,
    test_size: float,
) -> CachedModelOutputs:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in the open interval (0, 1)")
    if sensitive_attr not in input_df.columns:
        raise KeyError(f"Missing sensitive attribute column: {sensitive_attr}")

    X = input_df.drop(columns=["stroke", "id"])
    y = input_df["stroke"].astype(int)
    preprocessor = _build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    model_names: list[ModelName] = [
        "Logistic Regression",
        "Random Forest",
    ]
    y_proba_by_model: dict[ModelName, np.ndarray] = {}

    for model_name in model_names:
        model = _build_model(model_name)
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        pipeline.fit(X_train, y_train)
        y_proba_by_model[model_name] = pipeline.predict_proba(X_test)[:, 1]

    return {
        "y_test": y_test.to_numpy(),
        "sensitive": X_test[sensitive_attr].to_numpy(),
        "y_proba_by_model": y_proba_by_model,
    }
