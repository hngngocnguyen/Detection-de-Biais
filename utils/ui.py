from pathlib import Path

import streamlit as st


def _find_logo_path() -> Path | None:
    project_root = Path(__file__).resolve().parent.parent
    candidate_paths = [
        project_root / "assets" / "output-onlinepngtools.png",
        project_root / "assets" / "Logos.png",
        project_root / "assets" / "logo.png",
        project_root / "assets" / "logo.svg",
        project_root / "assets" / "logo.jpg",
        project_root / "assets" / "logo.jpeg",
        project_root / "logo.png",
        project_root / "logo.svg",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    assets_dir = project_root / "assets"
    if assets_dir.exists():
        for ext in ("png", "svg", "jpg", "jpeg", "webp"):
            for candidate in assets_dir.glob(f"*logo*.{ext}"):
                return candidate
            for candidate in assets_dir.glob(f"*.{ext}"):
                return candidate
    return None


def _render_sidebar_brand() -> None:
    logo_path = _find_logo_path()
    if logo_path is not None:
        _, centered_col, _ = st.sidebar.columns([0.5, 5.0, 0.5])
        with centered_col:
            st.image(str(logo_path), use_container_width=True)
    else:
        st.sidebar.markdown(
            """
            <div style='padding-top:0.15rem; text-align:center;'>
                <div style='font-size:1.15rem; font-weight:700; line-height:1.05;'>
                    CarePulse
                </div>
                <div style='font-size:1.05rem; font-weight:600; color:#1f9d8b; line-height:1.05;'>
                    Analytics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.sidebar.markdown(
        "<p style='text-align:center; margin:0;'>Prédiction du risque d'AVC et équité</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()


def _render_sidebar_nav() -> None:
    _render_sidebar_brand()
    st.sidebar.markdown("### Navigation")
    st.sidebar.page_link("app.py", label="Accueil", icon="🏠")
    st.sidebar.page_link(
        "pages/2_Exploration.py",
        label="Exploration",
        icon="📊",
    )
    st.sidebar.page_link(
        "pages/3_Détection_Biais.py",
        label="Détection de biais",
        icon="⚠️",
    )
    st.sidebar.page_link(
        "pages/4_Modélisation.py",
        label="Modélisation",
        icon="🤖",
    )


def apply_page_style() -> None:
    _render_sidebar_nav()
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:\
wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

        :root {
            --ink: #102a43;
            --bg-soft: #f4f8fb;
            --mint: #1f9d8b;
            --sun: #f0b429;
            --rose: #d6456d;
        }

        .stApp {
            background:
                radial-gradient(
                    1200px 450px at -10% -20%,
                    #d7f3ec 0%, transparent 60%),
                radial-gradient(
                    1000px 420px at 110% -10%,
                    #fff4d6 0%, transparent 55%),
                linear-gradient(180deg, #fcfdff 0%, var(--bg-soft) 100%);
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            padding-top: 0.8rem;
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif !important;
            letter-spacing: 0.2px;
            color: var(--ink);
        }

        h3 {
            margin-top: 0.7rem;
            margin-bottom: 0.45rem;
        }

        .hero-card {
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            background: linear-gradient(130deg, #102a43 0%, #1f9d8b 100%);
            color: #ffffff;
            box-shadow: 0 12px 30px rgba(16, 42, 67, 0.18);
            margin-bottom: 1rem;
        }

        .story-card {
            border-left: 6px solid var(--mint);
            background: #ffffff;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 24px rgba(16, 42, 67, 0.08);
            margin-bottom: 0.8rem;
        }

        .kpi-caption {
            font-size: 0.9rem;
            color: #486581;
            margin-top: -0.35rem;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 12px;
            padding: 0.45rem 0.55rem;
            box-shadow: 0 8px 20px rgba(16, 42, 67, 0.08);
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid #d9e2ec;
            background: #ffffff;
            color: var(--ink);
            font-weight: 600;
        }

        .stSelectbox > div[data-baseweb="select"] > div,
        .stMultiSelect > div[data-baseweb="select"] > div {
            border-radius: 12px;
            border-color: #d9e2ec;
            background: rgba(255, 255, 255, 0.92);
        }

        .stPlotlyChart {
            border-radius: 14px;
            background: #ffffff;
            padding: 0.35rem;
            box-shadow: 0 8px 20px rgba(16, 42, 67, 0.08);
        }

        [data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h2 style="margin-bottom:0.4rem; color:#ffffff;">{title}</h2>
            <p style="margin:0; opacity:0.95;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def story_block(text: str) -> None:
    st.markdown(
        f"<div class='story-card'>{text}</div>",
        unsafe_allow_html=True,
    )
