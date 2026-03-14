import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from utils.colors import METRIC_COLOR_MAP, VIVID_SEQUENCE
from utils.data import load_data
from utils.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    positive_rate_by_group,
)
from utils.labels import (
    SENSITIVE_ATTR_OPTIONS,
    label_group_value,
    label_metric,
    label_sensitive_attr,
)
from utils.modeling import run_bias_audit_model
from utils.ui import apply_page_style, hero, story_block


BIAS_METRIC_ORDER = ["Recall", "Precision", "F1", "F2"]


def _compute_bias_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    unprivileged_value: str,
    privileged_value: str,
) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "DPD": demographic_parity_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attribute=sensitive,
        ),
        "DI": disparate_impact_ratio(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attribute=sensitive,
            unprivileged_value=unprivileged_value,
            privileged_value=privileged_value,
        ),
    }


@st.cache_data(show_spinner=False)
def _bootstrap_bias_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sensitive: np.ndarray,
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
        sensitive_b = sensitive[idx]
        y_pred_b = (y_proba_b >= threshold).astype(int)

        rows.append(
            _compute_bias_metrics(
                y_true=y_true_b,
                y_pred=y_pred_b,
                sensitive=sensitive_b,
                unprivileged_value=unprivileged_value,
                privileged_value=privileged_value,
            )
        )

    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Détection de biais", page_icon="⚠️", layout="wide"
)
apply_page_style()

hero(
    "⚠️ Détection de Biais",
    "Mesurer l'équité entre groupes sensibles pour anticiper "
    "les risques de décisions injustes.",
)

story_block(
    "Attributs sensibles analysés : genre et zone de résidence. "
    "Un écart important entre groupes peut signifier que certains "
    "patients sont moins souvent identifiés comme à risque."
)

story_block(
    "Lecture conseillée : commencez par la synthèse des écarts, "
    "puis observez les taux positifs par groupe et l'effet du seuil "
    "avant d'interpréter l'indicateur décisionnel."
)

df = load_data().copy()

st.markdown("### Réglages d'audit")
controls_col1, controls_col2, controls_col3 = st.columns(3)
with controls_col1:
    sensitive_attr = st.selectbox(
        "Attribut sensible",
        SENSITIVE_ATTR_OPTIONS,
        format_func=label_sensitive_attr,
    )
with controls_col2:
    model_name = st.selectbox(
        "Modèle utilisé pour l'audit",
        ["Logistic Regression", "Random Forest"],
    )

# Apply recommended threshold on the next rerun, before the slider exists.
if "bias_pending_threshold" in st.session_state:
    st.session_state.bias_decision_threshold = float(
        st.session_state.pop("bias_pending_threshold")
    )

if "bias_decision_threshold" not in st.session_state:
    st.session_state.bias_decision_threshold = 0.5

with controls_col3:
    bias_decision_threshold = st.slider(
        "Seuil de décision pour l'audit",
        min_value=0.1,
        max_value=0.9,
        step=0.01,
        key="bias_decision_threshold",
    )

if sensitive_attr == "gender":
    unprivileged_value = "Female"
    privileged_value = "Male"
else:
    unprivileged_value = "Rural"
    privileged_value = "Urban"

st.markdown("### Biais analysé")
st.markdown(
    f"""
- Attribut sensible étudié : **{label_sensitive_attr(sensitive_attr)}**.
- Enjeu : un modèle peut attribuer moins souvent une prédiction positive à un groupe, même à niveau de risque comparable.
- Risque concret : certains patients peuvent être moins bien orientés vers un suivi préventif, ce qui crée une inégalité de prise en charge.
"""
)

bias_go_min_recall = 0.35
bias_go_max_abs_dpd = 0.05
bias_di_min = 0.80
bias_di_max = 1.25
bias_prudence_min_recall = 0.20
bias_prudence_max_abs_dpd = 0.10

audit = run_bias_audit_model(df, model_name, sensitive_attr)
y_true = audit["y_true"]
y_proba = audit["y_proba"]
y_pred = (y_proba >= bias_decision_threshold).astype(int)
sensitive = audit["sensitive"]

metrics = _compute_bias_metrics(
    y_true=y_true,
    y_pred=y_pred,
    unprivileged_value=unprivileged_value,
    privileged_value=privileged_value,
    sensitive=sensitive,
)
dpd = metrics["DPD"]
di_ratio = metrics["DI"]

st.markdown("### Lecture rapide")
quick1, quick2, quick3, quick4 = st.columns(4)
quick1.metric("Modèle audité", model_name)
quick2.metric("Rappel", f"{metrics['Recall']:.3f}")
quick3.metric("|Écart de parité|", f"{abs(float(dpd)):.3f}")
quick4.metric("Ratio d'impact", f"{di_ratio:.3f}")

st.markdown("### Mesures clés d'équité")

m1, m2 = st.columns(2)
m1.metric("Écart de parité démographique", f"{dpd:.4f}")
m2.metric("Ratio d'impact disproportionné", f"{di_ratio:.4f}")

p1, p2, p3 = st.columns(3)
p1.metric(
    "Taux de bonne classification",
    f"{metrics['Accuracy']:.3f}",
)
p2.metric(
    "Précision sur test",
    f"{metrics['Precision']:.3f}",
)
p3.metric(
    "Rappel sur test",
    f"{metrics['Recall']:.3f}",
)

abs_dpd = abs(float(metrics["DPD"]))
di_ok = bias_di_min <= float(metrics["DI"]) <= bias_di_max
recall_val = float(metrics["Recall"])

st.markdown("### Recommandation automatique du seuil")
thresholds = np.round(np.arange(0.0, 1.01, 0.01), 2)
threshold_rows = []
for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    m_thr = _compute_bias_metrics(
        y_true=y_true,
        y_pred=y_pred_thr,
        sensitive=sensitive,
        unprivileged_value=unprivileged_value,
        privileged_value=privileged_value,
    )
    threshold_rows.append(
        {
            "Seuil": float(thr),
            "Precision": m_thr["Precision"],
            "Recall": m_thr["Recall"],
            "F1": m_thr["F1"],
            "F2": m_thr["F2"],
            "DPD": m_thr["DPD"],
        }
    )
threshold_df = pd.DataFrame(threshold_rows)

strategy = st.selectbox(
    "Stratégie",
    [
        "Maximiser F1",
        "Priorité rappel (précision minimale)",
        "Minimiser le biais (|écart de parité|) sous contrainte",
    ],
)

if strategy == "Maximiser F1":
    recommended_row = threshold_df.loc[threshold_df["F1"].idxmax()]
elif strategy == "Priorité rappel (précision minimale)":
    min_precision = st.slider(
        "Précision minimale",
        min_value=0.1,
        max_value=0.95,
        value=0.40,
        step=0.01,
        key="bias_min_precision",
    )
    candidates = threshold_df[threshold_df["Precision"] >= min_precision]
    if len(candidates) == 0:
        recommended_row = threshold_df.loc[threshold_df["Recall"].idxmax()]
    else:
        recommended_row = candidates.loc[candidates["Recall"].idxmax()]
else:
    min_f2 = st.slider(
        "F2 minimal",
        min_value=0.1,
        max_value=0.95,
        value=0.40,
        step=0.01,
        key="bias_min_f2",
    )
    candidates = threshold_df[threshold_df["F2"] >= min_f2].copy()
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
    st.session_state.bias_pending_threshold = float(
        recommended_row["Seuil"]
    )
    st.rerun()

threshold_plot_df = threshold_df.melt(
    id_vars="Seuil",
    value_vars=BIAS_METRIC_ORDER,
)
threshold_plot_df["Métrique"] = threshold_plot_df["variable"].map(
    label_metric
)

fig_threshold = px.line(
    threshold_plot_df,
    x="Seuil",
    y="value",
    color="Métrique",
    title=f"{model_name} - effet du seuil sur la détection",
    color_discrete_map=METRIC_COLOR_MAP,
    category_orders={
        "Métrique": [label_metric(name) for name in BIAS_METRIC_ORDER]
    },
    labels={"value": "Valeur", "Seuil": "Seuil"},
)
fig_threshold.add_vline(
    x=bias_decision_threshold,
    line_dash="dash",
    line_color="#102a43",
)
st.plotly_chart(fig_threshold, use_container_width=True)

st.markdown("### Intervalles de confiance (IC 95%)")
boot_df = _bootstrap_bias_ci(
    y_true=y_true,
    y_proba=y_proba,
    sensitive=sensitive,
    threshold=bias_decision_threshold,
    unprivileged_value=unprivileged_value,
    privileged_value=privileged_value,
    n_boot=200,
)
ci_rows = []
for metric_name in ["Accuracy", "Precision", "Recall", "F2", "DPD", "DI"]:
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

rate_by_group = positive_rate_by_group(
    y_pred=y_pred, sensitive_attribute=sensitive
)
rate_df = pd.DataFrame(
    {
        sensitive_attr: list(rate_by_group.keys()),
        "positive_rate": [v * 100 for v in rate_by_group.values()],
    }
).sort_values("positive_rate", ascending=False)
rate_df["Libellé groupe"] = rate_df[sensitive_attr].map(label_group_value)

st.markdown("### Visualisation des résultats")

fig = px.bar(
    rate_df,
    x="Libellé groupe",
    y="positive_rate",
    color="Libellé groupe",
    labels={"positive_rate": "Taux positif (%)"},
    title=(
        "Comparaison des taux positifs par "
        f"{label_sensitive_attr(sensitive_attr).lower()}"
    ),
    color_discrete_sequence=VIVID_SEQUENCE,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Indicateur décisionnel")
if (
    recall_val >= bias_go_min_recall
    and abs_dpd <= bias_go_max_abs_dpd
    and di_ok
):
    st.success(
        "Indicateur: Favorable. Le niveau de détection est correct "
        "et les écarts de fairness restent contenus."
    )
elif (
    recall_val >= bias_prudence_min_recall
    and abs_dpd <= bias_prudence_max_abs_dpd
):
    st.warning(
        "Indicateur: À surveiller. Le modèle reste exploitable pour "
        "analyse, mais le rappel ou l'équité doivent être améliorés."
    )
else:
    st.error(
        "Indicateur: Non favorable. Les performances de détection ou "
        "les écarts d'équité sont trop risqués à ce stade."
    )

st.caption(
    "Règles utilisées: FAVORABLE si rappel >= 0.35, "
    "|écart de parité| <= 0.05 et ratio d'impact dans [0.80, 1.25]; "
    "À SURVEILLER si rappel >= 0.20 et |écart de parité| <= 0.10; "
    "NON FAVORABLE sinon (rappel < 0.20 ou |écart de parité| > 0.10, "
    "ou ratio d'impact hors [0.80, 1.25] dans le cas favorable). "
    "Aide à la décision, non prescriptive."
)

st.markdown("### Interprétation")

if pd.isna(di_ratio):
    fairness_statement = (
        "Le ratio d'impact n'est pas calculable car l'un des groupes "
        "n'a pas de taux de référence exploitable."
    )
elif 0.8 <= di_ratio <= 1.25:
    fairness_statement = (
        "Le ratio d'impact est proche de 1, ce qui suggère une "
        "disparité limitée entre groupes."
    )
else:
    fairness_statement = (
        "Le ratio d'impact s'écarte de l'intervalle [0.8, 1.25], "
        "ce qui indique un risque de biais significatif."
    )

least_favored = (
    label_group_value(rate_df.iloc[-1][sensitive_attr])
    if not rate_df.empty
    else "N/A"
)

st.markdown(
    f"""
L'écart de parité démographique observé est de **{dpd:.4f}** pour l'attribut **{label_sensitive_attr(sensitive_attr).lower()}**.  
{fairness_statement}  
Le groupe le plus défavorisé est **{least_favored}**, car son taux positif est le plus faible dans l'audit.  
En pratique, cela peut retarder l'identification des patients à risque dans ce groupe ; pour réduire ce biais, il est recommandé de suivre régulièrement les métriques d'équité, de rééquilibrer les données d'entraînement et d'ajuster le seuil de décision.
"""
)
