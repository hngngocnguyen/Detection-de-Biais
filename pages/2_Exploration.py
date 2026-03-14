import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.colors import STROKE_COLOR_MAP, VIVID_SEQUENCE
from utils.data import load_data
from utils.labels import (
    SENSITIVE_ATTR_OPTIONS,
    label_group_value,
    label_sensitive_attr,
)
from utils.ui import apply_page_style, hero, story_block

st.set_page_config(
    page_title="Exploration des données", page_icon="📊", layout="wide"
)
apply_page_style()

hero(
    "📊 Exploration des Données",
    "Pilotez les filtres, comparez les groupes sensibles et "
    "identifiez les signaux critiques du risque d'AVC.",
)
story_block(
    "Ajustez les filtres dans la barre latérale pour observer "
    "l'évolution des KPIs et des distributions."
)

df = load_data().copy()

st.sidebar.header("Filtres interactifs")
age_min, age_max = int(df["age"].min()), int(df["age"].max())
selected_age = st.sidebar.slider(
    "Plage d'âge", age_min, age_max, (age_min, age_max)
)
_gender_opts = sorted(df["gender"].dropna().unique())
selected_gender = st.sidebar.multiselect(
    "Genre",
    options=_gender_opts,
    default=_gender_opts,
    format_func=label_group_value,
)
_res_opts = sorted(df["Residence_type"].dropna().unique())
selected_residence = st.sidebar.multiselect(
    "Zone de résidence",
    options=_res_opts,
    default=_res_opts,
    format_func=label_group_value,
)
_stroke_opts = sorted(df["stroke"].dropna().unique())
selected_stroke = st.sidebar.multiselect(
    "Classe cible stroke", options=_stroke_opts, default=_stroke_opts
)

filtered_df = df[
    (df["age"].between(selected_age[0], selected_age[1]))
    & (df["gender"].isin(selected_gender))
    & (df["Residence_type"].isin(selected_residence))
    & (df["stroke"].isin(selected_stroke))
]

total_rows = len(df)
rows = len(filtered_df)
stroke_rate = filtered_df["stroke"].mean() * 100 if rows else 0
mean_age = filtered_df["age"].mean() if rows else 0
median_glucose = filtered_df["avg_glucose_level"].median() if rows else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Nombre total de lignes",
    f"{total_rows:,}",
    delta=f"{rows:,} après filtres",
)
c2.metric("Taux AVC (stroke=1)", f"{stroke_rate:.2f}%")
c3.metric("Âge moyen", f"{mean_age:.1f}")
c4.metric("Médiane glucose", f"{median_glucose:.1f}")

st.markdown("### Distribution de la variable cible")
if rows:
    target_counts = (
        filtered_df["stroke"]
        .value_counts()
        .rename_axis("stroke")
        .reset_index(name="count")
    )
    fig1 = px.bar(
        target_counts,
        x="stroke",
        y="count",
        color="stroke",
        color_discrete_map=STROKE_COLOR_MAP,
        title="Distribution de stroke",
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("Aucune ligne ne correspond aux filtres actuels.")

st.markdown("### Comparaison entre groupes sensibles")
sensitive_attr = st.radio(
    "Attribut sensible à comparer",
    options=SENSITIVE_ATTR_OPTIONS,
    format_func=label_sensitive_attr,
    horizontal=True,
)
if rows:
    group_rate = (
        filtered_df
        .groupby(sensitive_attr, as_index=False)["stroke"]
        .mean()
        .sort_values("stroke", ascending=False)
    )
    group_rate["stroke"] = group_rate["stroke"] * 100
    group_rate["Libellé groupe"] = group_rate[sensitive_attr].map(
        label_group_value
    )
    fig2 = px.bar(
        group_rate,
        x="Libellé groupe",
        y="stroke",
        color="Libellé groupe",
        title=(
            "Taux d'AVC par "
            f"{label_sensitive_attr(sensitive_attr).lower()}"
        ),
        labels={"stroke": "Taux d'AVC (%)"},
        color_discrete_sequence=VIVID_SEQUENCE,
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Aucune comparaison possible : ajustez les filtres.")

st.markdown("### Vue complémentaire")
chart_choice = st.selectbox(
    "Choisissez une visualisation",
    [
        "Nuage de points",
        "Boîte à moustaches",
        "Heatmap des corrélations",
        "Diagramme circulaire",
    ],
)

if rows:
    if chart_choice == "Nuage de points":
        scatter_df = filtered_df.assign(
            gender_label=filtered_df["gender"].map(label_group_value)
        )
        fig3 = px.scatter(
            scatter_df,
            x="age",
            y="avg_glucose_level",
            color="stroke",
            symbol="gender_label",
            title="Relation âge vs glucose",
            opacity=0.72,
            color_discrete_map=STROKE_COLOR_MAP,
        )
        st.plotly_chart(fig3, use_container_width=True)

    elif chart_choice == "Boîte à moustaches":
        box_df = filtered_df.assign(
            sensitive_label=filtered_df[sensitive_attr].map(
                label_group_value
            )
        )
        fig3 = px.box(
            box_df,
            x="sensitive_label",
            y="avg_glucose_level",
            color="sensitive_label",
            title=(
                "Distribution du glucose par "
                f"{label_sensitive_attr(sensitive_attr).lower()}"
            ),
            color_discrete_sequence=VIVID_SEQUENCE,
        )
        st.plotly_chart(fig3, use_container_width=True)

    elif chart_choice == "Heatmap des corrélations":
        corr_cols = [
            "age", "hypertension", "heart_disease",
            "avg_glucose_level", "bmi", "stroke",
        ]
        corr = filtered_df[corr_cols].corr(numeric_only=True)
        fig3 = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
            )
        )
        fig3.update_layout(title="Matrice de corrélation")
        st.plotly_chart(fig3, use_container_width=True)

    else:
        pie_data = filtered_df[sensitive_attr].value_counts().reset_index()
        pie_data.columns = [sensitive_attr, "count"]
        pie_data["Libellé groupe"] = pie_data[sensitive_attr].map(
            label_group_value
        )
        fig3 = px.pie(
            pie_data,
            values="count",
            names="Libellé groupe",
            title=(
                "Proportions par "
                f"{label_sensitive_attr(sensitive_attr).lower()}"
            ),
            color_discrete_sequence=VIVID_SEQUENCE,
        )
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("Aucune visualisation complémentaire disponible sans données filtrées.")
