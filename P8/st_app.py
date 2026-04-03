"""
Dashboard de prédiction crédit — Home Credit
Charge un fichier _explained.csv (ou predictions.csv avec colonnes shap_*)
et affiche pour chaque client :
  - Décision (Accepté / Rejeté) + probabilité
  - Top-10 SHAP values (waterfall-style)
  - Distribution des probabilités (position du client)
  - Scatter plot sur 2 features clés (position du client)
"""

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# Dictionnaire des noms de features — Home Credit
# Source : HomeCredit_columns_description.csv (Kaggle)
# Couvre application_train/test + features engineerées courantes
# ─────────────────────────────────────────────

FEATURE_LABELS: dict[str, str] = {
    # ── Identité & démographie ──
    "SK_ID_CURR":                        "ID Client",
    "CODE_GENDER":                       "Genre",
    "FLAG_OWN_CAR":                      "Possède une voiture",
    "FLAG_OWN_REALTY":                   "Possède un bien immobilier",
    "CNT_CHILDREN":                      "Nombre d'enfants",
    "NAME_CONTRACT_TYPE":                "Type de contrat",
    "NAME_TYPE_SUITE":                   "Accompagnateur lors de la demande",
    "NAME_INCOME_TYPE":                  "Type de revenus",
    "NAME_EDUCATION_TYPE":               "Niveau d'éducation",
    "NAME_FAMILY_STATUS":                "Situation familiale",
    "NAME_HOUSING_TYPE":                 "Type de logement",
    "OCCUPATION_TYPE":                   "Profession",
    "ORGANIZATION_TYPE":                 "Type d'organisation employeur",
    "WEEKDAY_APPR_PROCESS_START":        "Jour de la semaine (demande)",
    "HOUR_APPR_PROCESS_START":           "Heure de la demande",
    "LIVE_CITY_NOT_WORK_CITY":           "Ne travaille pas dans sa ville de résidence",
    "REG_CITY_NOT_LIVE_CITY":            "Ville enregistrée ≠ ville de résidence",
    "REG_CITY_NOT_WORK_CITY":            "Ville enregistrée ≠ ville de travail",
    "LIVE_REGION_NOT_WORK_REGION":       "Région résidence ≠ région travail",
    "REG_REGION_NOT_LIVE_REGION":        "Région enregistrée ≠ région de résidence",
    "REG_REGION_NOT_WORK_REGION":        "Région enregistrée ≠ région de travail",
    "EMERGENCYSTATE_MODE":               "État d'urgence du logement",
    "FONDKAPREMONT_MODE":                "Mode rénovation Fondkapremont",
    "HOUSETYPE_MODE":                    "Type de maison (mode)",
    "WALLSMATERIAL_MODE":                "Matériau des murs (mode)",

    # ── Finances & revenus ──
    "AMT_INCOME_TOTAL":                  "Revenus annuels totaux",
    "AMT_CREDIT":                        "Montant du crédit demandé",
    "AMT_ANNUITY":                       "Montant de l'annuité",
    "AMT_GOODS_PRICE":                   "Prix du bien financé",
    "AMT_REQ_CREDIT_BUREAU_HOUR":        "Demandes bureau crédit (dernière heure)",
    "AMT_REQ_CREDIT_BUREAU_DAY":         "Demandes bureau crédit (dernier jour)",
    "AMT_REQ_CREDIT_BUREAU_WEEK":        "Demandes bureau crédit (dernière semaine)",
    "AMT_REQ_CREDIT_BUREAU_MON":         "Demandes bureau crédit (dernier mois)",
    "AMT_REQ_CREDIT_BUREAU_QRT":         "Demandes bureau crédit (dernier trimestre)",
    "AMT_REQ_CREDIT_BUREAU_YEAR":        "Demandes bureau crédit (dernière année)",

    # ── Scores externes ──
    "EXT_SOURCE_1":                      "Score externe 1",
    "EXT_SOURCE_2":                      "Score externe 2",
    "EXT_SOURCE_3":                      "Score externe 3",

    # ── Durées / âges (en jours, valeurs négatives = passé) ──
    "DAYS_BIRTH":                        "Âge (jours depuis naissance)",
    "DAYS_EMPLOYED":                     "Ancienneté emploi (jours)",
    "DAYS_REGISTRATION":                 "Jours depuis changement d'enregistrement",
    "DAYS_ID_PUBLISH":                   "Jours depuis renouvellement pièce d'identité",
    "DAYS_LAST_PHONE_CHANGE":            "Jours depuis changement de téléphone",

    # ── Flags documents fournis ──
    "FLAG_DOCUMENT_2":                   "Document 2 fourni",
    "FLAG_DOCUMENT_3":                   "Document 3 fourni",
    "FLAG_DOCUMENT_4":                   "Document 4 fourni",
    "FLAG_DOCUMENT_5":                   "Document 5 fourni",
    "FLAG_DOCUMENT_6":                   "Document 6 fourni",
    "FLAG_DOCUMENT_7":                   "Document 7 fourni",
    "FLAG_DOCUMENT_8":                   "Document 8 fourni",
    "FLAG_DOCUMENT_9":                   "Document 9 fourni",
    "FLAG_DOCUMENT_10":                  "Document 10 fourni",
    "FLAG_DOCUMENT_11":                  "Document 11 fourni",
    "FLAG_DOCUMENT_12":                  "Document 12 fourni",
    "FLAG_DOCUMENT_13":                  "Document 13 fourni",
    "FLAG_DOCUMENT_14":                  "Document 14 fourni",
    "FLAG_DOCUMENT_15":                  "Document 15 fourni",
    "FLAG_DOCUMENT_16":                  "Document 16 fourni",
    "FLAG_DOCUMENT_17":                  "Document 17 fourni",
    "FLAG_DOCUMENT_18":                  "Document 18 fourni",
    "FLAG_DOCUMENT_19":                  "Document 19 fourni",
    "FLAG_DOCUMENT_20":                  "Document 20 fourni",
    "FLAG_DOCUMENT_21":                  "Document 21 fourni",
    "FLAG_EMAIL":                        "Email fourni",
    "FLAG_EMP_PHONE":                    "Téléphone employeur fourni",
    "FLAG_MOBIL":                        "Téléphone mobile fourni",
    "FLAG_PHONE":                        "Téléphone fixe fourni",
    "FLAG_WORK_PHONE":                   "Téléphone professionnel fourni",
    "FLAG_CONT_MOBILE":                  "Téléphone mobile joignable",

    # ── Caractéristiques du logement / région ──
    "REGION_POPULATION_RELATIVE":        "Population relative de la région",
    "REGION_RATING_CLIENT":              "Note de la région client (1–3)",
    "REGION_RATING_CLIENT_W_CITY":       "Note de la région client (avec ville)",
    "TOTALAREA_MODE":                    "Surface totale du logement (mode)",
    "YEARS_BUILD_AVG":                   "Ancienneté de construction (moyenne)",
    "YEARS_BEGINEXPLUATATION_AVG":       "Début d'exploitation du logement (moy.)",
    "FLOORSMAX_AVG":                     "Nombre d'étages max (moyenne)",
    "FLOORSMIN_AVG":                     "Nombre d'étages min (moyenne)",
    "LIVINGAREA_AVG":                    "Surface habitable (moyenne)",
    "LIVINGAPARTMENTS_AVG":              "Appartements habitables (moyenne)",
    "NONLIVINGAREA_AVG":                 "Surface non habitable (moyenne)",
    "BASEMENTAREA_AVG":                  "Surface sous-sol (moyenne)",
    "COMMONAREA_AVG":                    "Surface commune (moyenne)",
    "ELEVATORS_AVG":                     "Nombre d'ascenseurs (moyenne)",
    "ENTRANCES_AVG":                     "Nombre d'entrées (moyenne)",
    "LANDAREA_AVG":                      "Surface terrain (moyenne)",
    "APARTMENTS_AVG":                    "Nombre d'appartements (moyenne)",

    # ── Contacts & entourage ──
    "CNT_FAM_MEMBERS":                   "Nombre de membres de la famille",
    "OBS_30_CNT_SOCIAL_CIRCLE":          "Contacts observés défaut 30j (entourage)",
    "DEF_30_CNT_SOCIAL_CIRCLE":          "Contacts en défaut 30j (entourage)",
    "OBS_60_CNT_SOCIAL_CIRCLE":          "Contacts observés défaut 60j (entourage)",
    "DEF_60_CNT_SOCIAL_CIRCLE":          "Contacts en défaut 60j (entourage)",

    # ── Features engineerées courantes ──
    "CREDIT_INCOME_RATIO":               "Ratio crédit / revenus",
    "ANNUITY_INCOME_RATIO":              "Ratio annuité / revenus",
    "CREDIT_ANNUITY_RATIO":              "Ratio crédit / annuité",
    "CREDIT_GOODS_RATIO":                "Ratio crédit / prix du bien",
    "INCOME_PER_PERSON":                 "Revenus par membre de la famille",
    "PAYMENT_RATE":                      "Taux de remboursement (annuité/crédit)",
    "AGE_YEARS":                         "Âge (années)",
    "EMPLOYED_YEARS":                    "Ancienneté emploi (années)",
    "EXT_SOURCES_MEAN":                  "Moyenne des scores externes",
    "EXT_SOURCES_STD":                   "Écart-type des scores externes",
    "EXT_SOURCES_MIN":                   "Minimum des scores externes",
    "EXT_SOURCES_MAX":                   "Maximum des scores externes",
    "EXT_SOURCE_1_EXT_SOURCE_2":         "Score ext. 1 × Score ext. 2",
    "EXT_SOURCE_1_EXT_SOURCE_3":         "Score ext. 1 × Score ext. 3",
    "EXT_SOURCE_2_EXT_SOURCE_3":         "Score ext. 2 × Score ext. 3",
    "EXT_SOURCE_1_EXT_SOURCE_2_EXT_SOURCE_3": "Score ext. 1 × 2 × 3",
    "DAYS_EMPLOYED_PERC":                "% ancienneté emploi / âge",
    "INCOME_CREDIT_PERC":                "% revenus / crédit",
    "INCOME_PER_CHILD":                  "Revenus par enfant",
    "ANNUITY_LENGTH":                    "Durée du prêt estimée (années)",
}


def get_label(col: str) -> str:
    """Retourne le libellé lisible d'une colonne, ou le nom brut si inconnu."""
    return FEATURE_LABELS.get(col, col.replace("_", " ").title())


def get_label_with_raw(col: str) -> str:
    """Libellé lisible + nom technique entre parenthèses."""
    friendly = FEATURE_LABELS.get(col)
    if friendly:
        return f"{friendly}  <span style=\'font-size:0.7em;color:#6b7280\'>[{col}]</span>"
    return col.replace("_", " ").title()


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

/* Header principal */
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

/* Verdict cards */
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

/* Metric boxes */
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

/* Section titles */
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

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0c10;
    border-right: 1px solid #1e2028;
}
section[data-testid="stSidebar"] .main-title {
    font-size: 1.4rem;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #161920;
    border: 1px dashed #2d3040;
    border-radius: 4px;
    padding: 1rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2d3040; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#9ca3af", size=11),
)

# Defaults séparés pour éviter les conflits de keyword args dans update_layout
DEFAULT_MARGIN = dict(l=10, r=10, t=30, b=10)
DEFAULT_XAXIS  = dict(gridcolor="#1e2028", zerolinecolor="#2d3040")
DEFAULT_YAXIS  = dict(gridcolor="#1e2028", zerolinecolor="#2d3040")

COLOR_ACCEPT = "#22c55e"
COLOR_REJECT = "#ef4444"
COLOR_NEUTRAL = "#3b82f6"
COLOR_HIGHLIGHT = "#f59e0b"


def load_data(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def get_shap_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("shap_")]


def feature_name(shap_col: str) -> str:
    """shap_EXT_SOURCE_3 → EXT_SOURCE_3 (nom technique)"""
    return re.sub(r"^shap_", "", shap_col)


def feature_label(shap_col: str) -> str:
    """shap_EXT_SOURCE_3 → libellé lisible depuis FEATURE_LABELS."""
    return get_label(feature_name(shap_col))


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="main-title">🏦 Scoring<br>Crédit</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Home Credit · Analyse client</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Charger le fichier de prédictions",
        type=["csv", "xlsx"],
        help="Fichier _explained.csv ou predictions.csv avec colonnes shap_*",
    )

    if uploaded:
        df = load_data(uploaded)
        shap_cols = get_shap_cols(df)

        if "SK_ID_CURR" not in df.columns:
            st.error("❌ Colonne SK_ID_CURR absente.")
            st.stop()
        if "predicted_label" not in df.columns:
            st.error("❌ Colonne predicted_label absente.")
            st.stop()
        if "proba" not in df.columns:
            st.error("❌ Colonne proba absente.")
            st.stop()

        st.markdown("---")
        st.markdown('<p class="metric-label">Fichier chargé</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{len(df):,}</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">clients</p>', unsafe_allow_html=True)

        n_rejected = int(df["predicted_label"].sum())
        n_accepted = len(df) - n_rejected
        st.markdown(f"""
        <div style="margin-top:1rem; font-size:0.75rem; color:#6b7280;">
            ✅ Acceptés : <span style="color:{COLOR_ACCEPT}">{n_accepted:,}</span><br>
            ❌ Rejetés  : <span style="color:{COLOR_REJECT}">{n_rejected:,}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Sélection client
        client_ids = sorted(df["SK_ID_CURR"].unique().tolist())
        selected_id = st.selectbox(
            "🔍 Sélectionner un client",
            options=client_ids,
            format_func=lambda x: f"#{int(x)}",
        )

        # Seuil (affiché mais non modifiable ici — déjà appliqué côté API)
        threshold_display = 0.434
        st.markdown(f"""
        <div style="margin-top:1rem; font-size:0.7rem; color:#6b7280; letter-spacing:0.1em;">
            SEUIL APPLIQUÉ : <span style="color:#f59e0b">{threshold_display}</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main — état initial (pas de fichier)
# ─────────────────────────────────────────────

if not uploaded:
    st.markdown('<h1 class="main-title">Analyse de<br><em>scoring crédit</em></h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Chargez un fichier de prédictions dans la barre latérale</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:3rem; color:#4b5563; font-size:0.8rem; line-height:2;">
        Le fichier attendu est le fichier CSV produit par l'API<br>
        <code style="color:#6b7280">/predict/explain</code> — il doit contenir :<br><br>
        &nbsp;· <code>SK_ID_CURR</code> — identifiant client<br>
        &nbsp;· <code>predicted_label</code> — 0 ou 1<br>
        &nbsp;· <code>proba</code> — probabilité de défaut<br>
        &nbsp;· <code>shap_*</code> — valeurs SHAP des top features
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# Données client sélectionné
# ─────────────────────────────────────────────

client = df[df["SK_ID_CURR"] == selected_id].iloc[0]
label = int(client["predicted_label"])
proba = float(client["proba"])
is_accepted = label == 0

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown(f'<h1 class="main-title">Client <em>#{int(selected_id)}</em></h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Analyse individuelle · Scoring crédit</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Row 1 — Verdict + métriques
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
    risk_color = COLOR_REJECT if proba >= 0.434 else COLOR_ACCEPT

    # Jauge probabilité
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
                {"range": [0, 43.4], "color": "#0a1f14"},
                {"range": [43.4, 100], "color": "#1f0a0a"},
            ],
            "threshold": {
                "line": {"color": COLOR_HIGHLIGHT, "width": 2},
                "thickness": 0.8,
                "value": 43.4,
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

if shap_cols:
    st.markdown('<p class="section-title">📊 Analyse locale SHAP — contribution des features</p>', unsafe_allow_html=True)

    shap_values = {feature_label(c): float(client[c]) for c in shap_cols}
    shap_sorted = dict(sorted(shap_values.items(), key=lambda x: x[1]))

    features = list(shap_sorted.keys())
    values = list(shap_sorted.values())
    colors = [COLOR_REJECT if v > 0 else COLOR_ACCEPT for v in values]

    fig_shap = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>SHAP : %{x:.4f}<extra></extra>",
    ))

    # Ligne zéro
    fig_shap.add_vline(x=0, line_color="#4b5563", line_width=1)

    fig_shap.update_layout(
        **PLOTLY_LAYOUT,
        margin=DEFAULT_MARGIN,
        xaxis=DEFAULT_XAXIS,
        yaxis=dict(gridcolor="#1e2028", zerolinecolor="#2d3040", tickfont=dict(size=10)),
        height=420,
        title=dict(
            text="<span style='font-size:10px;color:#6b7280'>Rouge = pousse vers DÉFAUT · Vert = pousse vers NON-DÉFAUT</span>",
            x=0, font=dict(size=10)
        ),
        xaxis_title="Valeur SHAP",
    )
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("Aucune colonne SHAP détectée dans le fichier.")

# ─────────────────────────────────────────────
# Row 3 — Distribution probabilités + Scatter
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">📈 Position du client parmi l\'ensemble des prédictions</p>', unsafe_allow_html=True)

fig_hist = go.Figure()

df_acc = df[df["predicted_label"] == 0]
df_rej = df[df["predicted_label"] == 1]

fig_hist.add_trace(go.Histogram(
    x=df_acc["proba"],
    name="Acceptés",
    marker_color=COLOR_ACCEPT,
    opacity=0.55,
    nbinsx=50,
    hovertemplate="Proba : %{x:.2f}<br>Nb : %{y}<extra>Acceptés</extra>",
))
fig_hist.add_trace(go.Histogram(
    x=df_rej["proba"],
    name="Rejetés",
    marker_color=COLOR_REJECT,
    opacity=0.55,
    nbinsx=50,
    hovertemplate="Proba : %{x:.2f}<br>Nb : %{y}<extra>Rejetés</extra>",
))

fig_hist.add_vline(
    x=proba,
    line_color=COLOR_HIGHLIGHT,
    line_width=2,
    line_dash="dash",
    annotation_text=f"  Client #{int(selected_id)}",
    annotation_font=dict(color=COLOR_HIGHLIGHT, size=11),
    annotation_position="top right",
)

fig_hist.add_vline(
    x=0.434,
    line_color="#6b7280",
    line_width=1,
    line_dash="dot",
    annotation_text="  Seuil 0.434",
    annotation_font=dict(color="#6b7280", size=10),
    annotation_position="top left",
)

fig_hist.update_layout(
    **PLOTLY_LAYOUT,
    margin=DEFAULT_MARGIN,
    xaxis=DEFAULT_XAXIS,
    yaxis=DEFAULT_YAXIS,
    barmode="overlay",
    height=480,
    title=dict(text="Distribution des probabilités de défaut", font=dict(size=12, color="#9ca3af")),
    xaxis_title="Probabilité de défaut",
    yaxis_title="Nombre de clients",
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#9ca3af"),
        x=0.75, y=0.95,
    ),
)
st.plotly_chart(fig_hist, use_container_width=True)

# ─────────────────────────────────────────────
# Row 4 — Tableau récap client
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">📋 Détail des valeurs SHAP du client</p>', unsafe_allow_html=True)

if shap_cols:
    recap = pd.DataFrame({
        "Feature": [f"{feature_label(c)}  [{feature_name(c)}]" for c in shap_cols],
        "Valeur SHAP": [round(float(client[c]), 4) for c in shap_cols],
        "Impact": ["⬆ Défaut" if float(client[c]) > 0 else "⬇ Non-défaut" for c in shap_cols],
    }).sort_values("Valeur SHAP", key=abs, ascending=False).reset_index(drop=True)

    st.dataframe(
        recap,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Valeur SHAP": st.column_config.NumberColumn(format="%.4f"),
        },
    )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("""
<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #1e2028;
     font-size:0.65rem; color:#374151; letter-spacing:0.1em; text-align:center;">
    HOME CREDIT SCORING DASHBOARD · SHAP ANALYSIS · LIGHTGBM / XGBOOST
</div>
""", unsafe_allow_html=True)