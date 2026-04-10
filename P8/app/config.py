"""
Configuration du dashboard Scoring Crédit — Home Credit
Colonnes API, labels, styles CSS, templates HTML et constantes Plotly.
"""

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

API_URL = "https://open-classrooms.onrender.com"
MODEL = "lgb"
DEFAULT_THRESHOLD = 0.434
DEFAULT_N_TOP_SHAP = 10

# ─────────────────────────────────────────────
# Colonnes attendues par l'API
# ─────────────────────────────────────────────

API_COLS = [
    "SK_ID_CURR", "DAYS_EMPLOYED", "EXT_SOURCE_2", "EXT_SOURCE_3", "FLAG_DOCUMENT_6",
    "NAME_EDUCATION_TYPE", "EMPLOYED_TO_AGE_RATIO", "PHONE_CHANGE_YEARS", "CREDIT_GOODS_RATIO",
    "REGION_RATING_CLIENT_W_CITY", "REGION_RATING_CLIENT", "OCCUPATION_TYPE_nan",
    "AMT_GOODS_PRICE", "AMT_CREDIT", "FLAG_OWN_CAR", "RECENT_PHONE_CHANGE",
    "NAME_INCOME_TYPE_State_servant", "EMERGENCYSTATE_MODE_No", "EMERGENCYSTATE_MODE_nan",
    "HOUSETYPE_MODE_block_of_flats", "OCCUPATION_TYPE_Core_staff", "HOUSETYPE_MODE_nan",
    "WALLSMATERIAL_MODE_nan", "NAME_INCOME_TYPE_Commercial_associate", "HOUR_APPR_PROCESS_START",
    "NAME_FAMILY_STATUS_Married", "OCCUPATION_TYPE_Managers", "CODE_GENDER",
    "REGION_POPULATION_RELATIVE", "NAME_CONTRACT_TYPE_Revolving_loans", "WALLSMATERIAL_MODE_Panel",
    "OCCUPATION_TYPE_Accountants", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "REG_CITY_NOT_LIVE_CITY", "FONDKAPREMONT_MODE_nan", "AMT_ANNUITY", "ANNUITY_INCOME_RATIO",
    "AMT_REQ_CREDIT_BUREAU_YEAR", "OCCUPATION_TYPE_Low_skill_Laborers", "ORGANIZATION_TYPE_School",
    "FONDKAPREMONT_MODE_reg_oper_account", "NAME_FAMILY_STATUS_Single___not_married",
    "CNT_FAM_MEMBERS", "DAYS_ID_PUBLISH", "FLAG_PHONE", "ORGANIZATION_TYPE_Medicine",
    "OCCUPATION_TYPE_High_skill_tech_staff", "FLAG_DOCUMENT_8", "OCCUPATION_TYPE_Medicine_staff",
    "NAME_FAMILY_STATUS_Civil_marriage", "REG_CITY_NOT_WORK_CITY", "ORGANIZATION_TYPE_Military",
    "ORGANIZATION_TYPE_Government", "ANNUITY_CREDIT_RATIO", "AMT_REQ_CREDIT_BUREAU_MON",
    "NAME_HOUSING_TYPE_House___apartment", "REGISTRATION_YEARS", "ORGANIZATION_TYPE_Other",
    "FLAG_DOCUMENT_16", "NAME_HOUSING_TYPE_With_parents", "ORGANIZATION_TYPE_Kindergarten",
    "ORGANIZATION_TYPE_Security_Ministries", "OCCUPATION_TYPE_Drivers", "ORGANIZATION_TYPE_Police",
    "FLAG_DOCUMENT_13", "DOCUMENT_COUNT", "NAME_FAMILY_STATUS_Widow", "FLAG_DOCUMENT_18",
    "RECENT_ID_CHANGE", "CNT_CHILDREN", "OCCUPATION_TYPE_Laborers",
    "NAME_HOUSING_TYPE_Rented_apartment", "ORGANIZATION_TYPE_Bank",
    "WALLSMATERIAL_MODE_Stone__brick", "FLAG_DOCUMENT_14", "ORGANIZATION_TYPE_Transport__type_3",
    "ORGANIZATION_TYPE_Self_employed", "ORGANIZATION_TYPE_Industry__type_9",
    "ORGANIZATION_TYPE_University", "FONDKAPREMONT_MODE_org_spec_account", "FLAG_EMAIL",
    "FLAG_DOCUMENT_3", "ORGANIZATION_TYPE_Construction", "OBS_30_CNT_SOCIAL_CIRCLE",
    "OCCUPATION_TYPE_Private_service_staff", "WALLSMATERIAL_MODE_Monolithic",
    "ORGANIZATION_TYPE_Services", "ORGANIZATION_TYPE_Trade__type_6",
    "NAME_INCOME_TYPE_Unemployed", "WALLSMATERIAL_MODE_Block",
    "WEEKDAY_APPR_PROCESS_START_SATURDAY", "ORGANIZATION_TYPE_Restaurant",
    "ORGANIZATION_TYPE_Business_Entity_Type_2", "FLAG_DOCUMENT_15",
    "ORGANIZATION_TYPE_Transport__type_2", "FONDKAPREMONT_MODE_reg_oper_spec_account",
    "NAME_HOUSING_TYPE_Office_apartment", "ORGANIZATION_TYPE_Electricity",
    "WEEKDAY_APPR_PROCESS_START_TUESDAY", "FLAG_OWN_REALTY",
]

# ─────────────────────────────────────────────
# Labels lisibles des features
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

# ─────────────────────────────────────────────
# Couleurs
# ─────────────────────────────────────────────

COLOR_ACCEPT = "#22c55e"
COLOR_REJECT = "#ef4444"
COLOR_NEUTRAL = "#3b82f6"
COLOR_HIGHLIGHT = "#f59e0b"

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

# ─────────────────────────────────────────────
# CSS global
# ─────────────────────────────────────────────

GLOBAL_CSS = """
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
"""

# ─────────────────────────────────────────────
# Templates HTML (fonctions)
# ─────────────────────────────────────────────


def html_sidebar_cache_summary(n_cached: int, n_acc: int, n_rej: int) -> str:
    return f"""
    <div style="margin-top:0.5rem; font-size:0.7rem; color:#6b7280;">
        {n_cached} client(s) analysé(s)<br>
        ✅ Acceptés : <span style="color:{COLOR_ACCEPT}">{n_acc}</span> ·
        ❌ Rejetés : <span style="color:{COLOR_REJECT}">{n_rej}</span>
    </div>
    """


def html_sidebar_status(threshold: float) -> str:
    return f"""
    <div style="margin-top:1rem; font-size:0.7rem; color:#6b7280; letter-spacing:0.1em;">
        SEUIL : <span style="color:#f59e0b">{threshold}</span> ·
        MODÈLE : <span style="color:#f59e0b">LGB</span>
    </div>
    """


def html_verdict_card(is_accepted: bool) -> str:
    cls = "verdict-accept" if is_accepted else "verdict-reject"
    text = "✅ ACCEPTÉ" if is_accepted else "❌ REJETÉ"
    desc = "Risque faible — dossier favorable" if is_accepted else "Risque élevé — dossier défavorable"
    return f"""
    <div class="verdict-card {cls}">
        <p class="verdict-label">{text}</p>
        <p class="verdict-sub">{desc}</p>
    </div>
    """


HTML_LANDING_INSTRUCTIONS = """
<div style="margin-top:3rem; color:#4b5563; font-size:0.8rem; line-height:2;">
    <strong style="color:#6b7280;">1. Données clients</strong> — fichier CSV/Excel avec <code>SK_ID_CURR</code><br>
    &nbsp;&nbsp;&nbsp;→ Pour chaque client, le dashboard appellera l'API <code>/predict/explain</code><br><br>
    <strong style="color:#6b7280;">2. Jeu d'entraînement</strong> (optionnel) — fichier <code>_explained.csv</code><br>
    &nbsp;&nbsp;&nbsp;→ Précalculé via <code>predict_client.py</code> sur le training set<br>
    &nbsp;&nbsp;&nbsp;→ Permet d'afficher la distribution de référence et le scatter plot
</div>
"""

HTML_TRAIN_MISSING = """
<div style="margin-top:1rem; padding:1rem; background:#161920; border:1px solid #252830;
     border-radius:4px; font-size:0.8rem; color:#6b7280;">
    📈 Chargez le fichier d'entraînement (précalculé) dans la barre latérale pour afficher
    la distribution de référence et le scatter plot.
</div>
"""

HTML_FOOTER = """
<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #1e2028;
     font-size:0.65rem; color:#374151; letter-spacing:0.1em; text-align:center;">
    HOME CREDIT SCORING DASHBOARD · SHAP ANALYSIS · LIGHTGBM
</div>
"""
