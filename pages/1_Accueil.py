import plotly.express as px
import streamlit as st

from utils.colors import STROKE_COLOR_MAP
from utils.data import get_column_descriptions, load_data
from utils.ui import apply_page_style, hero, story_block

st.set_page_config(
    page_title="Accueil - Analyse AVC", page_icon="🏠", layout="wide"
)
apply_page_style()

hero(
    "🏠 Accueil",
    "Comprendre le dataset Stroke Prediction, ses enjeux "
    "métiers et les risques d'iniquité associés.",
)

story_block(
    "Ce jeu de données contient des variables médicales et "
    "socio-démographiques, avec pour objectif la prédiction du "
    "risque d'AVC via la variable cible stroke."
)
story_block(
    "En santé, une erreur d'évaluation n'a pas le même impact "
    "pour tous : une sous-estimation répétée pour un groupe peut "
    "retarder le dépistage et la prévention."
)
story_block(
    "L'application combine exploration visuelle, quantification "
    "du biais (fairness) et modélisation comparative pour "
    "éclairer les décisions."
)

df = load_data()

n_rows, n_cols = df.shape
missing_rate = (df.isna().sum().sum() / (n_rows * n_cols)) * 100
stroke_rate = df["stroke"].mean() * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Nombre total de lignes", f"{n_rows:,}")
k2.metric("Nombre de colonnes", f"{n_cols}")
k3.metric("Taux de valeurs manquantes", f"{missing_rate:.2f}%")
k4.metric("Distribution cible (AVC=1)", f"{stroke_rate:.2f}%")
st.markdown(
    "<div class='kpi-caption'>Lecture rapide de la qualité "
    "et de l'équilibre du dataset.</div>",
    unsafe_allow_html=True,
)

st.markdown("### Distribution de la variable cible")
target_counts = (
    df["stroke"]
    .value_counts()
    .rename_axis("stroke")
    .reset_index(name="count")
)
fig_target = px.bar(
    target_counts,
    x="stroke",
    y="count",
    color="stroke",
    color_discrete_map=STROKE_COLOR_MAP,
    title="Nombre de cas par classe cible",
)
fig_target.update_layout(showlegend=False)
st.plotly_chart(fig_target, use_container_width=True)

st.markdown("### Aperçu interactif des données")
st.dataframe(df, use_container_width=True, height=320)

st.markdown("### Description des colonnes")
st.dataframe(
    get_column_descriptions(), use_container_width=True, hide_index=True
)
