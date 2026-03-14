import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_curve,
)

from utils.colors import METRIC_COLOR_MAP, VIVID_SEQUENCE
from utils.data import load_data
from utils.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
)
from utils.labels import (
    SENSITIVE_ATTR_OPTIONS,
    label_group_value,
    label_metric,
    label_sensitive_attr,
)
from utils.modeling import get_cached_model_outputs
from utils.ui import apply_page_style, hero, story_block

METRIC_ORDER_GLOBAL = [
    "Balanced_Accuracy",
    "Recall",
    "Specificity",
    "F2",
    "PR_AUC",
]


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = float(cm[0, 0])
    fp = float(cm[0, 1])
    denom = tn + fp
    if denom == 0:
        return float("nan")
    return tn / denom


def _compute_metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    sensitive_vals: np.ndarray,
    unprivileged_value: str,
    privileged_value: str,
) -> dict[str, float]:
    if len(np.unique(y_true)) > 1:
        fpr_vals, tpr_vals, _ = roc_curve(y_true, y_proba)
        auc_value = auc(fpr_vals, tpr_vals)
    else:
        auc_value = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": _specificity(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "PR_AUC": average_precision_score(y_true, y_proba),
        "AUC": auc_value,
        "Brier": brier_score_loss(y_true, y_proba),
        "DPD": demographic_parity_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attribute=sensitive_vals,
        ),
        "DI": disparate_impact_ratio(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attribute=sensitive_vals,
            unprivileged_value=unprivileged_value,
            privileged_value=privileged_value,
        ),
    }


@st.cache_data(show_spinner=False)
def _bootstrap_ci_samples(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sensitive_vals: np.ndarray,
    threshold: float,
    unprivileged_value: str,
    privileged_value: str,
    n_boot: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_true_b = y_true[idx]
        y_proba_b = y_proba[idx]
        sensitive_b = sensitive_vals[idx]

        y_pred_b = (y_proba_b >= threshold).astype(int)
        row = _compute_metric_row(
            y_true=y_true_b,
            y_pred=y_pred_b,
            y_proba=y_proba_b,
            sensitive_vals=sensitive_b,
            unprivileged_value=unprivileged_value,
            privileged_value=privileged_value,
        )
        rows.append(row)

    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Modélisation", page_icon="🤖", layout="wide"
)
apply_page_style()

hero(
    "🤖 Modélisation",
    "Comparer précision prédictive et équité inter-groupes "
    "pour choisir un modèle responsable.",
)
story_block(
    "Deux modèles sont entraînés : Logistic Regression et "
    "Random Forest. Le tableau combine performance globale "
    "et métriques d'équité pour éviter une lecture uniquement "
    "centrée sur l'accuracy."
)

df = load_data().copy()

st.markdown("### Filtres du modèle")
model_names = ["Logistic Regression", "Random Forest"]

controls_col1, controls_col2, controls_col3, controls_col4 = st.columns(4)
with controls_col1:
    sensitive_attr = st.selectbox(
        "Attribut sensible pour l'analyse",
        SENSITIVE_ATTR_OPTIONS,
        format_func=label_sensitive_attr,
    )
with controls_col2:
    test_size = st.slider(
        "Taille du jeu de test",
        min_value=0.2,
        max_value=0.4,
        value=0.25,
        step=0.05,
    )

y_cache = get_cached_model_outputs(df, sensitive_attr, test_size)
y_test = y_cache["y_test"]
sensitive_test = y_cache["sensitive"]
probabilities = y_cache["y_proba_by_model"]

# Set a recall-oriented default threshold once (F2 optimization).
if "decision_threshold_autoset" not in st.session_state:
    y_proba_default = probabilities["Logistic Regression"]
    thr_candidates = np.round(np.arange(0.1, 0.91, 0.01), 2)
    f2_scores = []
    for thr in thr_candidates:
        y_pred_thr = (y_proba_default >= thr).astype(int)
        f2_scores.append(
            fbeta_score(
                y_test,
                y_pred_thr,
                beta=2,
                zero_division=0,
            )
        )
    best_idx = int(np.argmax(f2_scores))
    st.session_state.decision_threshold = float(thr_candidates[best_idx])
    st.session_state.decision_threshold_autoset = True
elif "decision_threshold" not in st.session_state:
    st.session_state.decision_threshold = 0.5

# Apply recommended threshold on the next rerun, before the slider exists.
if "decision_pending_threshold" in st.session_state:
    st.session_state.decision_threshold = float(
        st.session_state.pop("decision_pending_threshold")
    )

with controls_col3:
    decision_threshold = st.slider(
        "Seuil de décision (classe 1)",
        min_value=0.1,
        max_value=0.9,
        step=0.01,
        key="decision_threshold",
    )
with controls_col4:
    selected_model = st.selectbox(
        "Modèle affiché (matrices et analyses avancées)",
        model_names,
    )

model_go_min_recall = 0.35
model_go_min_pr_auc = 0.20
model_go_max_abs_dpd = 0.05
model_di_min = 0.80
model_di_max = 1.25
model_prudence_min_recall = 0.20
model_prudence_max_abs_dpd = 0.10

overall_rows = []
group_rows = []
predictions = {}

for model_name in model_names:
    y_proba = probabilities[model_name]
    y_pred = (y_proba >= decision_threshold).astype(int)

    predictions[model_name] = y_pred

    unpriv = "Female" if sensitive_attr == "gender" else "Rural"
    priv = "Male" if sensitive_attr == "gender" else "Urban"
    metric_row = _compute_metric_row(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        sensitive_vals=sensitive_test,
        unprivileged_value=unpriv,
        privileged_value=priv,
    )
    overall_rows.append(
        {
            "Modèle": model_name,
            **metric_row,
        }
    )

    for grp in sorted(pd.Series(sensitive_test).dropna().unique()):
        mask = sensitive_test == grp
        if mask.sum() == 0:
            continue
        group_rows.append(
            {
                "Modèle": model_name,
                "Groupe": label_group_value(grp),
                "Accuracy": accuracy_score(
                    y_test[mask], y_pred[mask]
                ),
                "Balanced_Accuracy": balanced_accuracy_score(
                    y_test[mask], y_pred[mask]
                ),
                "Precision": precision_score(
                    y_test[mask], y_pred[mask], zero_division=0
                ),
                "Recall": recall_score(
                    y_test[mask], y_pred[mask], zero_division=0
                ),
                "Specificity": _specificity(y_test[mask], y_pred[mask]),
                "Taille": int(mask.sum()),
            }
        )

overall_df = pd.DataFrame(overall_rows)
group_df = pd.DataFrame(group_rows)

best_row = overall_df.sort_values("F2", ascending=False).iloc[0]
best_model = str(best_row["Modèle"])
best_recall = float(best_row["Recall"])
best_pr_auc = float(best_row["PR_AUC"])
best_abs_dpd = abs(float(best_row["DPD"]))
best_di = float(best_row["DI"])

st.markdown("### Synthèse des résultats")
summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
summary_col1.metric("Modèle en tête", best_model)
summary_col2.metric("Score F2", f"{float(best_row['F2']):.3f}")
summary_col3.metric("Rappel", f"{best_recall:.3f}")
summary_col4.metric("|Écart de parité|", f"{best_abs_dpd:.3f}")

st.divider()

st.markdown("### Performances globales")
overall_display_df = overall_df.rename(columns=label_metric)
st.dataframe(
    overall_display_df.style.format(
        {
            "Accuracy": "{:.3f}",
            "Accuracy équilibrée": "{:.3f}",
            "Précision": "{:.3f}",
            "Rappel": "{:.3f}",
            "Spécificité": "{:.3f}",
            "Score F1": "{:.3f}",
            "Score F2": "{:.3f}",
            "Aire précision-rappel": "{:.3f}",
            "Aire ROC": "{:.3f}",
            "Score de Brier": "{:.3f}",
            "Écart de parité": "{:.3f}",
            "Ratio d'impact": "{:.3f}",
        }
    ),
    use_container_width=True,
)

perf_plot_df = overall_df.melt(
    id_vars="Modèle",
    value_vars=[
        "Balanced_Accuracy",
        "Recall",
        "Specificity",
        "F2",
        "PR_AUC",
    ],
)
perf_plot_df["Métrique"] = perf_plot_df["variable"].map(label_metric)

fig_perf = px.bar(
    perf_plot_df,
    x="Modèle",
    y="value",
    color="Métrique",
    barmode="group",
    title="Comparaison des performances globales",
    color_discrete_map=METRIC_COLOR_MAP,
    category_orders={
        "Métrique": [label_metric(name) for name in METRIC_ORDER_GLOBAL]
    },
    labels={"value": "Valeur"},
)
st.plotly_chart(fig_perf, use_container_width=True)

st.markdown("### Métriques d'équité (fairness)")
fairness_df = overall_df[["Modèle", "DPD", "DI"]].copy()
st.dataframe(
    fairness_df.rename(columns=label_metric).style.format(
        {
            "Écart de parité": "{:.3f}",
            "Ratio d'impact": "{:.3f}",
        }
    ),
    use_container_width=True,
)

fairness_plot_df = fairness_df.melt(
    id_vars="Modèle",
    value_vars=["DPD", "DI"],
)
fairness_plot_df["Métrique"] = fairness_plot_df["variable"].map(
    label_metric
)

fig_fairness = px.bar(
    fairness_plot_df,
    x="Modèle",
    y="value",
    color="Métrique",
    barmode="group",
    title="Comparaison des métriques d'équité",
    color_discrete_map=METRIC_COLOR_MAP,
    labels={"value": "Valeur"},
)
st.plotly_chart(fig_fairness, use_container_width=True)

st.markdown("### Indicateur décisionnel")
if (
    best_recall >= model_go_min_recall
    and best_pr_auc >= model_go_min_pr_auc
    and best_abs_dpd <= model_go_max_abs_dpd
    and model_di_min <= best_di <= model_di_max
):
    st.success(
        "Indicateur: Favorable. Le meilleur modèle combine un "
        "niveau de détection correct et une équité acceptable."
    )
elif (
    best_recall >= model_prudence_min_recall
    and best_abs_dpd <= model_prudence_max_abs_dpd
):
    st.warning(
        "Indicateur: À surveiller. Le modèle est utile pour pilotage, "
        "mais des améliorations restent nécessaires avant usage sensible."
    )
else:
    st.error(
        "Indicateur: Non favorable. Les performances ou l'équité ne "
        "sont pas suffisantes pour une recommandation opérationnelle."
    )

st.caption(
    f"Modèle recommandé actuellement: {best_model} | "
    f"Rappel={best_recall:.3f}, aire précision-rappel={best_pr_auc:.3f}, "
    f"|Écart de parité|={best_abs_dpd:.3f}, ratio d'impact={best_di:.3f} "
    f"au seuil {decision_threshold:.2f}. "
    "Seuils fixés dans le code. Aide à la décision, non prescriptive."
)

st.divider()

st.markdown(
    "### Comparaison par groupe sensible : "
    f"{label_sensitive_attr(sensitive_attr)}"
)
st.dataframe(
    group_df.rename(columns=label_metric).style.format(
        {
            "Accuracy": "{:.3f}",
            "Accuracy équilibrée": "{:.3f}",
            "Précision": "{:.3f}",
            "Rappel": "{:.3f}",
            "Spécificité": "{:.3f}",
        }
    ),
    use_container_width=True,
)

group_plot_df = group_df.melt(
    id_vars=["Modèle", "Groupe", "Taille"],
    value_vars=[
        "Balanced_Accuracy",
        "Recall",
        "Specificity",
    ],
)
group_plot_df["Métrique"] = group_plot_df["variable"].map(label_metric)

fig_group = px.bar(
    group_plot_df,
    x="Groupe",
    y="value",
    color="Modèle",
    facet_col="Métrique",
    barmode="group",
    title="Performances par groupe sensible",
    color_discrete_sequence=VIVID_SEQUENCE,
    labels={"value": "Valeur"},
)
st.plotly_chart(fig_group, use_container_width=True)

st.markdown("### Matrices de confusion par groupe")
y_pred_selected = predictions[selected_model]
y_proba_selected = probabilities[selected_model]

unique_groups = sorted(pd.Series(sensitive_test).dropna().unique())
for grp in unique_groups:
    mask = sensitive_test == grp
    cm = confusion_matrix(y_test[mask], y_pred_selected[mask], labels=[0, 1])
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=["Prédit 0", "Prédit 1"],
            y=["Réel 0", "Réel 1"],
            colorscale="Teal",
            text=cm,
            texttemplate="%{text}",
            hovertemplate="Valeur: %{text}<extra></extra>",
        )
    )
    fig_cm.update_layout(
        title=f"{selected_model} — Groupe {label_group_value(grp)}"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

st.markdown("### Analyse avancée (seuil, ROC, calibration)")

thresholds = np.round(np.arange(0.0, 1.01, 0.01), 2)
curve_rows = []
unpriv = "Female" if sensitive_attr == "gender" else "Rural"
priv = "Male" if sensitive_attr == "gender" else "Urban"

for thr in thresholds:
    y_pred_thr = (y_proba_selected >= thr).astype(int)
    row_thr = _compute_metric_row(
        y_true=y_test,
        y_pred=y_pred_thr,
        y_proba=y_proba_selected,
        sensitive_vals=sensitive_test,
        unprivileged_value=unpriv,
        privileged_value=priv,
    )
    curve_rows.append(
        {
            "Seuil": float(thr),
            "Precision": row_thr["Precision"],
            "Recall": row_thr["Recall"],
            "F1": row_thr["F1"],
            "F2": row_thr["F2"],
            "Balanced_Accuracy": row_thr["Balanced_Accuracy"],
            "DPD": row_thr["DPD"],
        }
    )
threshold_df = pd.DataFrame(curve_rows)

st.markdown("#### Recommandation automatique du seuil")
strategy = st.selectbox(
    "Stratégie d'optimisation",
    [
        "Maximiser F1",
        "Priorité rappel (précision minimale)",
        "Priorité précision (rappel minimal)",
        "Minimiser le biais (|écart de parité|) sous contrainte",
    ],
)

recommended_row = None
if strategy == "Maximiser F1":
    recommended_row = threshold_df.loc[threshold_df["F1"].idxmax()]
elif strategy == "Priorité rappel (précision minimale)":
    min_precision = st.slider(
        "Précision minimale",
        min_value=0.1,
        max_value=0.95,
        value=0.40,
        step=0.01,
    )
    candidates = threshold_df[threshold_df["Precision"] >= min_precision]
    if len(candidates) == 0:
        recommended_row = threshold_df.loc[threshold_df["Recall"].idxmax()]
    else:
        recommended_row = candidates.loc[candidates["Recall"].idxmax()]
elif strategy == "Priorité précision (rappel minimal)":
    min_recall = st.slider(
        "Rappel minimal",
        min_value=0.1,
        max_value=0.95,
        value=0.40,
        step=0.01,
    )
    candidates = threshold_df[threshold_df["Recall"] >= min_recall]
    if len(candidates) == 0:
        recommended_row = threshold_df.loc[
            threshold_df["Precision"].idxmax()
        ]
    else:
        recommended_row = candidates.loc[candidates["Precision"].idxmax()]
else:
    min_bal_acc = st.slider(
        "Accuracy équilibrée minimale",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.01,
    )
    candidates = threshold_df[
        threshold_df["Balanced_Accuracy"] >= min_bal_acc
    ].copy()
    if len(candidates) == 0:
        candidates = threshold_df.copy()
    candidates["abs_dpd"] = candidates["DPD"].abs()
    recommended_row = candidates.loc[candidates["abs_dpd"].idxmin()]

st.info(
    "Seuil recommandé = "
    f"{float(recommended_row['Seuil']):.2f} | "
    f"Score F1={float(recommended_row['F1']):.3f} | "
    f"Rappel={float(recommended_row['Recall']):.3f} | "
    f"Précision={float(recommended_row['Precision']):.3f} | "
    f"|Écart de parité|={abs(float(recommended_row['DPD'])):.3f}"
)

if st.button("Appliquer le seuil recommandé", use_container_width=False):
    st.session_state.decision_pending_threshold = float(
        recommended_row["Seuil"]
    )
    st.rerun()

threshold_plot_df = threshold_df.melt(
    id_vars="Seuil",
    value_vars=["Recall", "Precision", "F1", "F2"],
)
threshold_plot_df["Métrique"] = threshold_plot_df["variable"].map(
    label_metric
)

fig_threshold = px.line(
    threshold_plot_df,
    x="Seuil",
    y="value",
    color="Métrique",
    title=f"{selected_model} - effet du seuil sur la détection",
    color_discrete_map=METRIC_COLOR_MAP,
    category_orders={
        "Métrique": [
            label_metric(name)
            for name in ["Recall", "Precision", "F1", "F2"]
        ]
    },
    labels={"value": "Valeur"},
)
fig_threshold.add_vline(
    x=decision_threshold,
    line_dash="dash",
    line_color="#102a43",
)
st.plotly_chart(fig_threshold, use_container_width=True)

st.markdown("#### Intervalles de confiance (IC 95%)")
boot_df = _bootstrap_ci_samples(
    y_true=y_test,
    y_proba=y_proba_selected,
    sensitive_vals=sensitive_test,
    threshold=decision_threshold,
    unprivileged_value=unpriv,
    privileged_value=priv,
    n_boot=200,
)
ci_metrics = [
    "Accuracy",
    "Balanced_Accuracy",
    "Recall",
    "Specificity",
    "F2",
    "PR_AUC",
    "DPD",
    "DI",
]
ci_rows = []
for metric_name in ci_metrics:
    vals = boot_df[metric_name].dropna().to_numpy()
    if len(vals) == 0:
        continue
    ci_rows.append(
        {
            "Métrique": label_metric(metric_name),
            "Moyenne": float(np.mean(vals)),
            "IC95 bas": float(np.quantile(vals, 0.025)),
            "IC95 haut": float(np.quantile(vals, 0.975)),
        }
    )

ci_df = pd.DataFrame(ci_rows)
st.dataframe(
    ci_df.style.format(
        {
            "Moyenne": "{:.3f}",
            "IC95 bas": "{:.3f}",
            "IC95 haut": "{:.3f}",
        }
    ),
    use_container_width=True,
)

if len(np.unique(y_test)) > 1:
    fpr, tpr, _ = roc_curve(y_test, y_proba_selected)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            name=f"ROC (AUC={roc_auc:.3f})",
            line=dict(color="#1f9d8b", width=3),
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Aléatoire",
            line=dict(color="#486581", dash="dash"),
        )
    )
    fig_roc.update_layout(
        title=f"{selected_model} - courbe ROC",
        xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

calib_df = pd.DataFrame(
    {
        "y_true": y_test,
        "y_proba": y_proba_selected,
    }
)
calib_df["bin"] = pd.cut(
    calib_df["y_proba"],
    bins=np.linspace(0.0, 1.0, 11),
    include_lowest=True,
)
calibration_table = (
    calib_df.groupby("bin", observed=False)
    .agg(
        predicted_mean=("y_proba", "mean"),
        observed_rate=("y_true", "mean"),
        count=("y_true", "count"),
    )
    .reset_index(drop=True)
    .dropna()
)

fig_cal = go.Figure()
fig_cal.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name="Calibration parfaite",
        line=dict(color="#486581", dash="dash"),
    )
)
fig_cal.add_trace(
    go.Scatter(
        x=calibration_table["predicted_mean"],
        y=calibration_table["observed_rate"],
        name=selected_model,
        mode="lines+markers",
        marker=dict(size=8, color="#d6456d"),
        line=dict(color="#d6456d", width=2),
        text=calibration_table["count"].astype(str),
        hovertemplate=(
            "Proba moyenne: %{x:.2f}<br>"
            "Taux observé: %{y:.2f}<br>"
            "N: %{text}<extra></extra>"
        ),
    )
)
fig_cal.update_layout(
    title=f"{selected_model} - courbe de calibration",
    xaxis_title="Probabilité prédite",
    yaxis_title="Fréquence observée",
    xaxis_range=[0, 1],
    yaxis_range=[0, 1],
)
st.plotly_chart(fig_cal, use_container_width=True)

st.caption(
    f"Score de Brier ({selected_model}) : "
    f"{brier_score_loss(y_test, y_proba_selected):.4f} "
    "(plus proche de 0 = meilleure calibration)."
)
