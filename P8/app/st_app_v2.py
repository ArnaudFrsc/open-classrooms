"""
Dashboard de scoring crédit — Home Credit
Script Streamlit unifié : upload du fichier brut, appel API /predict/explain,
cache des résultats par client dans session_state.
"""

import io
import re

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────
# Colonnes attendues par l'API
# ─────────────────────────────────────────────

API_COLS = [
    'SK_ID_CURR', 'DAYS_EMPLOYED', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'FLAG_DOCUMENT_6',
    'NAME_EDUCATION_TYPE', 'EMPLOYED_TO_AGE_RATIO', 'PHONE_CHANGE_YEARS', 'CREDIT_GOODS_RATIO',
    'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'OCCUPATION_TYPE_nan',
    'AMT_GOODS_PRICE', 'AMT_CREDIT', 'FLAG_OWN_CAR', 'RECENT_PHONE_CHANGE',
    'NAME_INCOME_TYPE_State_servant', 'EMERGENCYSTATE_MODE_No', 'EMERGENCYSTATE_MODE_nan',
    'HOUSETYPE_MODE_block_of_flats', 'OCCUPATION_TYPE_Core_staff', 'HOUSETYPE_MODE_nan',
    'WALLSMATERIAL_MODE_nan', 'NAME_INCOME_TYPE_Commercial_associate', 'HOUR_APPR_PROCESS_START',
    'NAME_FAMILY_STATUS_Married', 'OCCUPATION_TYPE_Managers', 'CODE_GENDER',
    'REGION_POPULATION_RELATIVE', 'NAME_CONTRACT_TYPE_Revolving_loans', 'WALLSMATERIAL_MODE_Panel',
    'OCCUPATION_TYPE_Accountants', 'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
    'REG_CITY_NOT_LIVE_CITY', 'FONDKAPREMONT_MODE_nan', 'AMT_ANNUITY', 'ANNUITY_INCOME_RATIO',
    'AMT_REQ_CREDIT_BUREAU_YEAR', 'OCCUPATION_TYPE_Low_skill_Laborers', 'ORGANIZATION_TYPE_School',
    'FONDKAPREMONT_MODE_reg_oper_account', 'NAME_FAMILY_STATUS_Single___not_married',
    'CNT_FAM_MEMBERS', 'DAYS_ID_PUBLISH', 'FLAG_PHONE', 'ORGANIZATION_TYPE_Medicine',
    'OCCUPATION_TYPE_High_skill_tech_staff', 'FLAG_DOCUMENT_8', 'OCCUPATION_TYPE_Medicine_staff',
    'NAME_FAMILY_STATUS_Civil_marriage', 'REG_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE_Military',
    'ORGANIZATION_TYPE_Government', 'ANNUITY_CREDIT_RATIO', 'AMT_REQ_CREDIT_BUREAU_MON',
    'NAME_HOUSING_TYPE_House___apartment', 'REGISTRATION_YEARS', 'ORGANIZATION_TYPE_Other',
    'FLAG_DOCUMENT_16', 'NAME_HOUSING_TYPE_With_parents', 'ORGANIZATION_TYPE_Kindergarten',
    'ORGANIZATION_TYPE_Security_Ministries', 'OCCUPATION_TYPE_Drivers', 'ORGANIZATION_TYPE_Police',
    'FLAG_DOCUMENT_13', 'DOCUMENT_COUNT', 'NAME_FAMILY_STATUS_Widow', 'FLAG_DOCUMENT_18',
    'RECENT_ID_CHANGE', 'CNT_CHILDREN', 'OCCUPATION_TYPE_Laborers',
    'NAME_HOUSING_TYPE_Rented_apartment', 'ORGANIZATION_TYPE_Bank',
    'WALLSMATERIAL_MODE_Stone__brick', 'FLAG_DOCUMENT_14', 'ORGANIZATION_TYPE_Transport__type_3',
    'ORGANIZATION_TYPE_Self_employed', 'ORGANIZATION_TYPE_Industry__type_9',
    'ORGANIZATION_TYPE_University', 'FONDKAPREMONT_MODE_org_spec_account', 'FLAG_EMAIL',
    'FLAG_DOCUMENT_3', 'ORGANIZATION_TYPE_Construction', 'OBS_30_CNT_SOCIAL_CIRCLE',
    'OCCUPATION_TYPE_Private_service_staff', 'WALLSMATERIAL_MODE_Monolithic',
    'ORGANIZATION_TYPE_Services', 'ORGANIZATION_TYPE_Trade__type_6',
    'NAME_INCOME_TYPE_Unemployed', 'WALLSMATERIAL_MODE_Block',
    'WEEKDAY_APPR_PROCESS_START_SATURDAY', 'ORGANIZATION_TYPE_Restaurant',
    'ORGANIZATION_TYPE_Business_Entity_Type_2', 'FLAG_DOCUMENT_15',
    'ORGANIZATION_TYPE_Transport__type_2', 'FONDKAPREMONT_MODE_reg_oper_spec_account',
    'NAME_HOUSING_TYPE_Office_apartment', 'ORGANIZATION_TYPE_Electricity',
    'WEEKDAY_APPR_PROCESS_START_TUESDAY', 'FLAG_OWN_REALTY',
]

# ─────────────────────────────────────────────
# Configuration API
# ─────────────────────────────────────────────

API_URL = "https://open-classrooms.onrender.com"
DEFAULT_MODEL = "lgb"
DEFAULT_THRESHOLD = 0.434
DEFAULT_N_TOP_SHAP = 10

# ─────────────────────────────────────────────
# Dictionnaire des noms de features
# ─────────────────────────────────────────────

FEATURE_LABELS: dict[str, str] = {
    "SK_ID_CURR": "ID Client",
    "CODE_GENDER": "Genre",
    "FLAG_OWN_CAR": "Possède une voiture",
    "FLAG_OWN_REALTY": "Possède un bien immobilier",
    "CNT_CHILDREN": "Nombre d'enfants",
    "NAME_CONTRACT_TYPE": "Type de contrat",
    "NAME_INCOME_TYPE": "Type de revenus",
    "NAME_EDUCATION_TYPE": "Niveau d'éducation",
    "NAME_FAMILY_STATUS": "Situation familiale",
    "NAME_HOUSING_TYPE": "Type de logement",
    "OCCUPATION_TYPE": "Profession",
    "ORGANIZATION_TYPE": "Type d'organisation employeur",
    "HOUR_APPR_PROCESS_START": "Heure de la demande",
    "REG_CITY_NOT_LIVE_CITY": "Ville enregistrée ≠ ville de résidence",
    "REG_CITY_NOT_WORK_CITY": "Ville enregistrée ≠ ville de travail",
    "EMERGENCYSTATE_MODE": "État d'urgence du logement",
    "FONDKAPREMONT_MODE": "Mode rénovation Fondkapremont",
    "HOUSETYPE_MODE": "Type de maison (mode)",
    "WALLSMATERIAL_MODE": "Matériau des murs (mode)",
    "AMT_INCOME_TOTAL": "Revenus annuels totaux",
    "AMT_CREDIT": "Montant du crédit demandé",
    "AMT_ANNUITY": "Montant de l'annuité",
    "AMT_GOODS_PRICE": "Prix du bien financé",
    "AMT_REQ_CREDIT_BUREAU_MON": "Demandes bureau crédit (dernier mois)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Demandes bureau crédit (dernière année)",
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "DAYS_BIRTH": "Âge (jours depuis naissance)",
    "DAYS_EMPLOYED": "Ancienneté emploi (jours)",
    "DAYS_ID_PUBLISH": "Jours depuis renouvellement pièce d'identité",
    "FLAG_DOCUMENT_3": "Document 3 fourni",
    "FLAG_DOCUMENT_6": "Document 6 fourni",
    "FLAG_DOCUMENT_8": "Document 8 fourni",
    "FLAG_DOCUMENT_13": "Document 13 fourni",
    "FLAG_DOCUMENT_14": "Document 14 fourni",
    "FLAG_DOCUMENT_15": "Document 15 fourni",
    "FLAG_DOCUMENT_16": "Document 16 fourni",
    "FLAG_DOCUMENT_18": "Document 18 fourni",
    "FLAG_EMAIL": "Email fourni",
    "FLAG_PHONE": "Téléphone fixe fourni",
    "REGION_POPULATION_RELATIVE": "Population relative de la région",
    "REGION_RATING_CLIENT": "Note de la région client (1–3)",
    "REGION_RATING_CLIENT_W_CITY": "Note de la région client (avec ville)",
    "CNT_FAM_MEMBERS": "Nombre de membres de la famille",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Contacts observés défaut 30j (entourage)",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Contacts en défaut 30j (entourage)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Contacts en défaut 60j (entourage)",
    "CREDIT_INCOME_RATIO": "Ratio crédit / revenus",
    "ANNUITY_INCOME_RATIO": "Ratio annuité / revenus",
    "ANNUITY_CREDIT_RATIO": "Ratio annuité / crédit",
    "CREDIT_GOODS_RATIO": "Ratio crédit / prix du bien",
    "EMPLOYED_TO_AGE_RATIO": "Ratio ancienneté emploi / âge",
    "PHONE_CHANGE_YEARS": "Changement téléphone (années)",
    "RECENT_PHONE_CHANGE": "Changement récent de téléphone",
    "RECENT_ID_CHANGE": "Changement récent d'identité",
    "REGISTRATION_YEARS": "Années depuis enregistrement",
    "DOCUMENT_COUNT": "Nombre de documents fournis",
}


def get_label(col: str) -> str:
    return FEATURE_LABELS.get(col, col.replace("_", " ").title())


# ─────────────────────────────────────────────
# Config page
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Scoring Crédit · Home Credit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS custom — dark banking aesthetic
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0d0f14;
    color: #e8e6e0;
}
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    color: #f0ede6;
    margin-bottom: 0;
    line-height: 1.1;
}
.main-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #6b7280;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 2rem;
}
.verdict-card {
    border-radius: 4px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border-left: 4px solid;
}
.verdict-accept {
    background: linear-gradient(135deg, #0a1f14 0%, #0d2b1a 100%);
    border-color: #22c55e;
}
.verdict-reject {
    background: linear-gradient(135deg, #1f0a0a 0%, #2b0d0d 100%);
    border-color: #ef4444;
}
.verdict-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 0;
    line-height: 1;
}
.verdict-accept .verdict-label { color: #22c55e; }
.verdict-reject .verdict-label { color: #ef4444; }
.verdict-sub {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-top: 6px;
}
.metric-box {
    background: #161920;
    border: 1px solid #252830;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #f0ede6;
    line-height: 1;
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b7280;
    margin-top: 6px;
}
.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b7280;
    border-bottom: 1px solid #252830;
    padding-bottom: 8px;
    margin-bottom: 1rem;
    margin-top: 2rem;
}
section[data-testid="stSidebar"] {
    background: #0a0c10;
    border-right: 1px solid #1e2028;
}
section[data-testid="stSidebar"] .main-title {
    font-size: 1.4rem;
}
[data-testid="stFileUploader"] {
    background: #161920;
    border: 1px dashed #2d3040;
    border-radius: 4px;
    padding: 1rem;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2d3040; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly defaults
# ─────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#9ca3af", size=11),
)
DEFAULT_MARGIN = dict(l=10, r=10, t=30, b=10)
DEFAULT_XAXIS = dict(gridcolor="#1e2028", zerolinecolor="#2d3040")
DEFAULT_YAXIS = dict(gridcolor="#1e2028", zerolinecolor="#2d3040")

COLOR_ACCEPT = "#22c55e"
COLOR_REJECT = "#ef4444"
COLOR_NEUTRAL = "#3b82f6"
COLOR_HIGHLIGHT = "#f59e0b"

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────

if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}  # {SK_ID_CURR: {proba, predicted_label, shap_values: {feat: val}}}

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

if "file_id" not in st.session_state:
    st.session_state.file_id = None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aligne le DataFrame sur les colonnes attendues par l'API."""
    df.columns = df.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
    return df.reindex(columns=API_COLS)


def call_api_explain(row_df: pd.DataFrame, model: str, threshold: float, n_top: int) -> dict | None:
    """
    Envoie un CSV d'une seule ligne à /predict/explain et retourne le résultat parsé.
    Retourne None en cas d'erreur.
    """
    csv_buffer = io.StringIO()
    row_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    url = f"{API_URL.rstrip('/')}/predict/explain"
    params = {
        "model": model,
        "threshold": threshold,
        "return_proba": "true",
        "n_top": n_top,
    }

    try:
        resp = requests.post(
            url,
            params=params,
            files={"file": ("client.csv", csv_bytes, "text/csv")},
            timeout=300,
        )
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur réseau : {e}")
        return None

    if resp.status_code != 200:
        st.error(f"❌ Erreur API ({resp.status_code}) : {resp.text[:500]}")
        return None

    # Parse CSV de retour
    try:
        result_df = pd.read_csv(io.BytesIO(resp.content))
    except Exception as e:
        st.error(f"❌ Impossible de lire la réponse API : {e}")
        return None

    if result_df.empty:
        st.error("❌ L'API a retourné un résultat vide.")
        return None

    row = result_df.iloc[0]
    shap_cols = [c for c in result_df.columns if c.startswith("shap_")]
    shap_values = {c: float(row[c]) for c in shap_cols}

    return {
        "predicted_label": int(row["predicted_label"]),
        "proba": float(row["proba"]),
        "shap_values": shap_values,
    }


def feature_name(shap_col: str) -> str:
    return re.sub(r"^shap_", "", shap_col)


def feature_label(shap_col: str) -> str:
    return get_label(feature_name(shap_col))


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="main-title">🏦 Scoring<br>Crédit</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Home Credit · Analyse client</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Charger le fichier de données clients",
        type=["csv", "xlsx"],
        help="Fichier CSV ou Excel contenant au minimum la colonne SK_ID_CURR",
    )

    if uploaded:
        # Détection de changement de fichier → reset du cache
        new_file_id = f"{uploaded.name}_{uploaded.size}"
        if new_file_id != st.session_state.file_id:
            st.session_state.file_id = new_file_id
            st.session_state.results_cache = {}
            if uploaded.name.endswith(".csv"):
                st.session_state.df_raw = pd.read_csv(uploaded)
            else:
                st.session_state.df_raw = pd.read_excel(uploaded)

    df_raw = st.session_state.df_raw

    if df_raw is not None:
        if "SK_ID_CURR" not in df_raw.columns:
            st.error("❌ Colonne SK_ID_CURR absente du fichier.")
            st.stop()

        st.markdown("---")
        st.markdown('<p class="metric-label">Fichier chargé</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{len(df_raw):,}</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">clients</p>', unsafe_allow_html=True)

        # Résumé du cache
        n_cached = len(st.session_state.results_cache)
        if n_cached > 0:
            n_acc = sum(1 for v in st.session_state.results_cache.values() if v["predicted_label"] == 0)
            n_rej = n_cached - n_acc
            st.markdown(f"""
            <div style="margin-top:0.5rem; font-size:0.7rem; color:#6b7280;">
                {n_cached} client(s) analysé(s)<br>
                ✅ Acceptés : <span style="color:{COLOR_ACCEPT}">{n_acc}</span> ·
                ❌ Rejetés : <span style="color:{COLOR_REJECT}">{n_rej}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Sélection client
        client_ids = sorted(df_raw["SK_ID_CURR"].unique().tolist())
        selected_id = st.selectbox(
            "🔍 Sélectionner un client",
            options=client_ids,
            format_func=lambda x: f"#{int(x)}",
        )

        # Paramètres API
        st.markdown("---")
        st.markdown('<p class="metric-label">Paramètres API</p>', unsafe_allow_html=True)
        model = st.selectbox("Modèle", ["lgb", "xgb"], index=0)
        threshold = st.slider("Seuil de décision", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
        n_top_shap = st.slider("Nombre de features SHAP", 3, 20, DEFAULT_N_TOP_SHAP)

        # Bouton d'analyse
        already_cached = selected_id in st.session_state.results_cache
        btn_label = "✅ Déjà analysé — Réafficher" if already_cached else "🚀 Lancer l'analyse"
        analyze = st.button(btn_label, use_container_width=True, type="primary")

        st.markdown(f"""
        <div style="margin-top:1rem; font-size:0.7rem; color:#6b7280; letter-spacing:0.1em;">
            SEUIL : <span style="color:#f59e0b">{threshold}</span> ·
            MODÈLE : <span style="color:#f59e0b">{model.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main — état initial
# ─────────────────────────────────────────────

if df_raw is None:
    st.markdown('<h1 class="main-title">Analyse de<br><em>scoring crédit</em></h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Chargez un fichier de données dans la barre latérale</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:3rem; color:#4b5563; font-size:0.8rem; line-height:2;">
        Chargez un fichier CSV ou Excel contenant vos données clients.<br>
        Le fichier doit contenir au minimum la colonne <code>SK_ID_CURR</code>.<br><br>
        Pour chaque client sélectionné, le dashboard appellera l'API<br>
        <code>/predict/explain</code> pour obtenir la prédiction et les SHAP values.<br><br>
        Les résultats sont mis en cache — pas de double appel API.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not analyze:
    # Pas de clic → si un client est déjà dans le cache, on l'affiche
    if selected_id not in st.session_state.results_cache:
        st.markdown('<h1 class="main-title">Analyse de<br><em>scoring crédit</em></h1>', unsafe_allow_html=True)
        st.markdown('<p class="main-subtitle">Sélectionnez un client et cliquez sur « Lancer l\'analyse »</p>', unsafe_allow_html=True)
        st.stop()

# ─────────────────────────────────────────────
# Appel API (ou lecture cache)
# ─────────────────────────────────────────────

if selected_id not in st.session_state.results_cache:
    with st.spinner("⏳ Appel API en cours — calcul SHAP…"):
        # Préparer le CSV d'une ligne aligné
        client_row = df_raw[df_raw["SK_ID_CURR"] == selected_id].copy()
        client_row_aligned = align_columns(client_row)

        result = call_api_explain(client_row_aligned, model, threshold, n_top_shap)
        if result is None:
            st.stop()

        st.session_state.results_cache[selected_id] = result
        st.rerun()  # Rerun pour mettre à jour la sidebar (compteurs)

# Lire depuis le cache
cached = st.session_state.results_cache[selected_id]
label = cached["predicted_label"]
proba = cached["proba"]
shap_values = cached["shap_values"]
is_accepted = label == 0

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown(f'<h1 class="main-title">Client <em>#{int(selected_id)}</em></h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Analyse individuelle · Scoring crédit</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Row 1 — Verdict + Jauge
# ─────────────────────────────────────────────

col_verdict, col_metrics = st.columns([1.4, 1], gap="large")

with col_verdict:
    verdict_class = "verdict-accept" if is_accepted else "verdict-reject"
    verdict_text = "✅ ACCEPTÉ" if is_accepted else "❌ REJETÉ"
    verdict_desc = "Risque faible — dossier favorable" if is_accepted else "Risque élevé — dossier défavorable"

    st.markdown(f"""
    <div class="verdict-card {verdict_class}">
        <p class="verdict-label">{verdict_text}</p>
        <p class="verdict-sub">{verdict_desc}</p>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    proba_pct = round(proba * 100, 1)
    risk_color = COLOR_REJECT if proba >= threshold else COLOR_ACCEPT

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba_pct,
        number={"suffix": "%", "font": {"size": 28, "family": "DM Serif Display", "color": "#f0ede6"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4b5563", "tickfont": {"size": 9}},
            "bar": {"color": risk_color, "thickness": 0.25},
            "bgcolor": "#161920",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 100], "color": "#0a1f14"},
                {"range": [threshold * 100, 100], "color": "#1f0a0a"},
            ],
            "threshold": {
                "line": {"color": COLOR_HIGHLIGHT, "width": 2},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
        title={"text": "PROBABILITÉ DE DÉFAUT", "font": {"size": 9, "color": "#6b7280", "family": "DM Mono"}},
    ))
    fig_gauge.update_layout(**PLOTLY_LAYOUT, height=200)
    fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────────────────────────
# Row 2 — SHAP waterfall
# ─────────────────────────────────────────────

if shap_values:
    st.markdown('<p class="section-title">📊 Analyse locale SHAP — contribution des features</p>', unsafe_allow_html=True)

    # Libellés lisibles
    shap_display = {feature_label(c): v for c, v in shap_values.items()}
    shap_sorted = dict(sorted(shap_display.items(), key=lambda x: x[1]))

    features = list(shap_sorted.keys())
    values = list(shap_sorted.values())
    colors = [COLOR_REJECT if v > 0 else COLOR_ACCEPT for v in values]

    fig_shap = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>SHAP : %{x:.4f}<extra></extra>",
    ))
    fig_shap.add_vline(x=0, line_color="#4b5563", line_width=1)
    fig_shap.update_layout(
        **PLOTLY_LAYOUT,
        margin=DEFAULT_MARGIN,
        xaxis=DEFAULT_XAXIS,
        yaxis=dict(gridcolor="#1e2028", zerolinecolor="#2d3040", tickfont=dict(size=10)),
        height=420,
        title=dict(
            text="<span style='font-size:10px;color:#6b7280'>Rouge = pousse vers DÉFAUT · Vert = pousse vers NON-DÉFAUT</span>",
            x=0, font=dict(size=10),
        ),
        xaxis_title="Valeur SHAP",
    )
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("Aucune valeur SHAP retournée par l'API.")

# ─────────────────────────────────────────────
# Row 3 — Distribution des probabilités (tous clients analysés)
# ─────────────────────────────────────────────

cache = st.session_state.results_cache
if len(cache) > 1:
    st.markdown('<p class="section-title">📈 Position du client parmi les clients analysés</p>', unsafe_allow_html=True)

    all_probas = pd.DataFrame([
        {"SK_ID_CURR": k, "proba": v["proba"], "predicted_label": v["predicted_label"]}
        for k, v in cache.items()
    ])

    df_acc = all_probas[all_probas["predicted_label"] == 0]
    df_rej = all_probas[all_probas["predicted_label"] == 1]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_acc["proba"], name="Acceptés", marker_color=COLOR_ACCEPT,
        opacity=0.55, nbinsx=30,
        hovertemplate="Proba : %{x:.2f}<br>Nb : %{y}<extra>Acceptés</extra>",
    ))
    fig_hist.add_trace(go.Histogram(
        x=df_rej["proba"], name="Rejetés", marker_color=COLOR_REJECT,
        opacity=0.55, nbinsx=30,
        hovertemplate="Proba : %{x:.2f}<br>Nb : %{y}<extra>Rejetés</extra>",
    ))
    fig_hist.add_vline(
        x=proba, line_color=COLOR_HIGHLIGHT, line_width=2, line_dash="dash",
        annotation_text=f"  Client #{int(selected_id)}",
        annotation_font=dict(color=COLOR_HIGHLIGHT, size=11),
        annotation_position="top right",
    )
    fig_hist.add_vline(
        x=threshold, line_color="#6b7280", line_width=1, line_dash="dot",
        annotation_text=f"  Seuil {threshold}",
        annotation_font=dict(color="#6b7280", size=10),
        annotation_position="top left",
    )
    fig_hist.update_layout(
        **PLOTLY_LAYOUT, margin=DEFAULT_MARGIN,
        xaxis=DEFAULT_XAXIS, yaxis=DEFAULT_YAXIS,
        barmode="overlay", height=420,
        title=dict(text="Distribution des probabilités de défaut", font=dict(size=12, color="#9ca3af")),
        xaxis_title="Probabilité de défaut", yaxis_title="Nombre de clients",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#9ca3af"), x=0.75, y=0.95),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
elif len(cache) == 1:
    st.markdown("""
    <div style="margin-top:1rem; padding:1rem; background:#161920; border:1px solid #252830;
         border-radius:4px; font-size:0.8rem; color:#6b7280;">
        📈 Le graphique de distribution apparaîtra dès que 2 clients ou plus seront analysés.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Row 4 — Tableau récap SHAP
# ─────────────────────────────────────────────

if shap_values:
    st.markdown('<p class="section-title">📋 Détail des valeurs SHAP du client</p>', unsafe_allow_html=True)

    recap = pd.DataFrame({
        "Feature": [f"{feature_label(c)}  [{feature_name(c)}]" for c in shap_values.keys()],
        "Valeur SHAP": [round(v, 4) for v in shap_values.values()],
        "Impact": ["⬆ Défaut" if v > 0 else "⬇ Non-défaut" for v in shap_values.values()],
    }).sort_values("Valeur SHAP", key=abs, ascending=False).reset_index(drop=True)

    st.dataframe(
        recap,
        use_container_width=True,
        hide_index=True,
        column_config={"Valeur SHAP": st.column_config.NumberColumn(format="%.4f")},
    )

# ─────────────────────────────────────────────
# Bouton pour analyser tous les clients d'un coup
# ─────────────────────────────────────────────

st.markdown("---")
remaining = [cid for cid in client_ids if cid not in st.session_state.results_cache]

if remaining:
    if st.button(f"🔄 Analyser tous les clients restants ({len(remaining)})", use_container_width=True):
        progress_bar = st.progress(0, text="Analyse en cours…")
        for i, cid in enumerate(remaining):
            client_row = df_raw[df_raw["SK_ID_CURR"] == cid].copy()
            client_row_aligned = align_columns(client_row)
            result = call_api_explain(client_row_aligned, model, threshold, n_top_shap)
            if result is not None:
                st.session_state.results_cache[cid] = result
            progress_bar.progress((i + 1) / len(remaining), text=f"Client #{int(cid)} — {i + 1}/{len(remaining)}")
        st.rerun()
else:
    st.markdown("""
    <div style="font-size:0.8rem; color:#22c55e; text-align:center;">
        ✅ Tous les clients ont été analysés.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("""
<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #1e2028;
     font-size:0.65rem; color:#374151; letter-spacing:0.1em; text-align:center;">
    HOME CREDIT SCORING DASHBOARD · SHAP ANALYSIS · LIGHTGBM / XGBOOST
</div>
""", unsafe_allow_html=True)
