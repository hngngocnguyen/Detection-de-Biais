from __future__ import annotations

import numpy as np


def _positive_rate(y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred == 1))


def demographic_parity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attribute: np.ndarray,
) -> float:
    """Écart entre le taux positif max et min par groupe sensible."""
    _ = y_true
    y_pred = np.asarray(y_pred)
    sensitive_attribute = np.asarray(sensitive_attribute)

    unique_groups = np.unique(sensitive_attribute)
    rates = []
    for group in unique_groups:
        mask = sensitive_attribute == group
        if np.sum(mask) == 0:
            continue
        rates.append(_positive_rate(y_pred[mask]))

    if not rates:
        return float("nan")

    return float(np.max(rates) - np.min(rates))


def disparate_impact_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attribute: np.ndarray,
    unprivileged_value,
    privileged_value,
) -> float:
    """Ratio des taux positifs non privilégié / privilégié."""
    _ = y_true
    y_pred = np.asarray(y_pred)
    sensitive_attribute = np.asarray(sensitive_attribute)

    unpriv_mask = sensitive_attribute == unprivileged_value
    priv_mask = sensitive_attribute == privileged_value

    if np.sum(unpriv_mask) == 0 or np.sum(priv_mask) == 0:
        return float("nan")

    unpriv_rate = _positive_rate(y_pred[unpriv_mask])
    priv_rate = _positive_rate(y_pred[priv_mask])

    if priv_rate == 0:
        return float("nan")

    return float(unpriv_rate / priv_rate)


def positive_rate_by_group(
    y_pred: np.ndarray, sensitive_attribute: np.ndarray
) -> dict:
    y_pred = np.asarray(y_pred)
    sensitive_attribute = np.asarray(sensitive_attribute)

    results = {}
    for group in np.unique(sensitive_attribute):
        mask = sensitive_attribute == group
        if np.sum(mask) == 0:
            continue
        results[group] = _positive_rate(y_pred[mask])
    return results
