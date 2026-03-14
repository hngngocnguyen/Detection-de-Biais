import streamlit as st

from utils.data import get_column_descriptions, load_data
from utils.ui import apply_page_style, hero, story_block

st.set_page_config(
    page_title="Analyse AVC - Détection de Biais",
    page_icon="🧠",
    layout="wide",
)

apply_page_style()

hero(
    "Analyse du risque d'AVC et équité des modèles",
    "Explorer les données, détecter les biais et comparer les performances de manière responsable.",
)

story_block(
    "Cette page présente l'objectif du projet, son cadre méthodologique "
    "et les principes éthiques retenus pour l'analyse."
)

st.markdown(
    """
### La question centrale
Comment concevoir un modèle de prédiction du risque d'AVC qui soit utile, interprétable et le plus équitable possible entre les groupes de patients ?

### Contexte et problématique
L'AVC représente un enjeu majeur de santé publique. Les données cliniques disponibles permettent d'estimer un niveau de risque, mais la qualité de cette estimation dépend directement des choix de modélisation et de la qualité du jeu de données.

Dans un contexte d'IA en santé, la performance globale ne suffit pas. Un modèle peut sembler efficace en moyenne et produire des résultats moins justes pour certains profils de patients. L'identification de ces écarts est donc une étape centrale de l'analyse.

Cette démarche vise à combiner trois objectifs : comprendre les données, comparer des modèles de prédiction et analyser l'équité des résultats avant toute interprétation décisionnelle.
"""
)

st.info(
    "Dataset : "
    "[Stroke Prediction Dataset (Kaggle)]"
    "(https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)"
)

st.caption(
    "Jeu de données de santé publique comportant des variables démographiques, "
    "médicales et comportementales liées au risque d'AVC."
)

st.divider()

df = load_data()

row_count, col_count = df.shape
missing_rate = (df.isna().sum().sum() / (row_count * col_count)) * 100

if "stroke" in df.columns:
    target_counts = df["stroke"].value_counts(dropna=False)
    stroke_yes = int(target_counts.get(1, 0))
    stroke_no = int(target_counts.get(0, 0))
    stroke_rate = (stroke_yes / row_count) * 100
    stroke_metric_value = f"{stroke_rate:.1f}%"
    stroke_caption = (
        f"Répartition cible : AVC=0 ({stroke_no}) | AVC=1 ({stroke_yes})"
    )
else:
    stroke_metric_value = "N/A"
    stroke_caption = "Variable cible 'stroke' absente du dataset."

st.markdown("### Métriques clés")
kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Nombre total de lignes", f"{row_count:,}".replace(",", " "))
kpi_1.markdown(
    "<div class='kpi-caption'>Taille totale de l'échantillon de patients.</div>",
    unsafe_allow_html=True,
)
kpi_2.metric("Nombre de colonnes", str(col_count))
kpi_2.markdown(
    "<div class='kpi-caption'>Nombre de variables disponibles pour l'analyse.</div>",
    unsafe_allow_html=True,
)
kpi_3.metric("Taux de valeurs manquantes", f"{missing_rate:.2f}%")
kpi_3.markdown(
    "<div class='kpi-caption'>Part des cellules non renseignées dans le dataset.</div>",
    unsafe_allow_html=True,
)
kpi_4.metric("Distribution de la cible (AVC=1)", stroke_metric_value)
kpi_4.markdown(
    "<div class='kpi-caption'>Proportion des cas positifs pour la variable cible.</div>",
    unsafe_allow_html=True,
)
st.caption(stroke_caption)

st.markdown("### Aperçu des données")
st.dataframe(df, use_container_width=True, height=320)

st.markdown("### Description des colonnes")
st.dataframe(get_column_descriptions(), use_container_width=True, hide_index=True)

st.markdown(
    """
### Parcours conseillé
Commencez par la page Exploration, poursuivez avec Détection de biais, puis terminez par Modélisation pour arbitrer entre performance et équité.
"""
)

st.divider()

st.markdown("## ℹ️ À propos")
st.markdown(
    "Ce projet pédagogique illustre un cas d'usage d'IA en santé. "
    "L'objectif n'est pas de produire un dispositif médical, mais de "
    "montrer une démarche rigoureuse d'analyse des données, de détection "
    "de biais et d'aide à la décision."
)

with st.expander("🧪 Méthodologie, RGPD et éthique", expanded=True):
    st.markdown(
        """
- Minimisation des données : seules les variables utiles à la prédiction sont traitées.
- Transparence : les performances et facteurs influents sont affichés pour auditabilité.
- Vigilance biais : le modèle est analysé par sous-groupes pour limiter les discriminations indirectes.
- Usage responsable : cet outil assiste l'analyse humaine, il ne se substitue pas à un avis médical.
"""
    )

st.info(
    "📫 Contact : hongngoc.nguyen@edu.nexa.fr\n\n"
    "Auteur : Hong Ngoc NGUYEN"
)
