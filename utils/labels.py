SENSITIVE_ATTR_OPTIONS = ["gender", "Residence_type"]

SENSITIVE_ATTR_LABELS = {
    "gender": "Genre",
    "Residence_type": "Zone de résidence",
}

GROUP_VALUE_LABELS = {
    "Female": "Femmes",
    "Male": "Hommes",
    "Other": "Autre",
    "Rural": "Zone rurale",
    "Urban": "Zone urbaine",
}

METRIC_LABELS = {
    "Accuracy": "Accuracy",
    "Balanced_Accuracy": "Accuracy équilibrée",
    "Precision": "Précision",
    "Recall": "Rappel",
    "Specificity": "Spécificité",
    "F1": "Score F1",
    "F2": "Score F2",
    "PR_AUC": "Aire précision-rappel",
    "AUC": "Aire ROC",
    "Brier": "Score de Brier",
    "DPD": "Écart de parité",
    "DI": "Ratio d'impact",
}


def label_sensitive_attr(value: str) -> str:
    return SENSITIVE_ATTR_LABELS.get(value, value)


def label_group_value(value: str) -> str:
    return GROUP_VALUE_LABELS.get(value, value)


def label_metric(value: str) -> str:
    return METRIC_LABELS.get(value, value)
