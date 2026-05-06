"""
Configuration du dashboard Scoring Crédit — Home Credit
Colonnes API, labels, styles CSS, templates HTML et constantes Plotly.
"""

import json
from pathlib import Path

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

API_URL = "https://open-classrooms.onrender.com"
MODEL = "lgb"
DEFAULT_THRESHOLD = 0.3646
DEFAULT_N_TOP_SHAP = 10

# ─────────────────────────────────────────────
# Colonnes attendues par l'API (chargées depuis cols_kept.json)
# ─────────────────────────────────────────────

_cols_path = Path(__file__).parent / "data" / "cols_kept.json"
with open(_cols_path) as _f:
    _feature_cols = json.load(_f)

API_COLS = ["SK_ID_CURR"] + _feature_cols

# ─────────────────────────────────────────────
# Labels lisibles des features
# ─────────────────────────────────────────────

# ============================================================
# Dictionnaire de traduction des features - Home Credit Default Risk
# ============================================================
# Sources :
#   [O] = HomeCredit_columns_description.csv (officiel Kaggle)
#   [A] = Convention d'agrégation issue des kernels Kaggle (Aguiar et al.)
#   [D] = Feature dérivée custom — traduction inférée du nom
# ============================================================

FEATURE_LABELS: dict[str, str] = {

    # ---- Identité / démographie [O] ----
    "CODE_GENDER": "Genre",
    "CNT_CHILDREN": "Nombre d'enfants",
    "CNT_FAM_MEMBERS": "Nombre de membres de la famille",
    "NAME_EDUCATION_TYPE": "Niveau d'études",
    "FLAG_OWN_CAR": "Propriétaire d'une voiture",

    # ---- Temporalité (en jours, négatif = passé) [O] ----
    "DAYS_EMPLOYED": "Ancienneté dans l'emploi (jours avant la demande)",
    "DAYS_LAST_PHONE_CHANGE": "Jours depuis le dernier changement de téléphone",
    "ID_PUBLISH_YEARS": "Ancienneté de la pièce d'identité (années)",  # [D]
    "RECENT_PHONE_CHANGE": "Changement de téléphone récent (flag)",     # [D]
    "RECENT_ID_CHANGE": "Changement de pièce d'identité récent (flag)", # [D]
    "HOUR_APPR_PROCESS_START": "Heure de dépôt de la demande",

    # ---- Scores externes [O] ----
    # Sources externes normalisées (non documentées par Kaggle)
    "EXT_SOURCE_2": "Score externe 2 (source normalisée)",
    "EXT_SOURCE_3": "Score externe 3 (source normalisée)",

    # ---- Contacts / documents [O] ----
    "FLAG_PHONE": "Téléphone fixe fourni",
    "FLAG_EMAIL": "Adresse e-mail fournie",
    "FLAG_DOCUMENT_6": "Document 6 fourni",
    "DOCUMENT_COUNT": "Nombre total de documents fournis",  # [D]

    # ---- Crédit demandé [O] ----
    "AMT_CREDIT": "Montant du crédit demandé",
    "AMT_GOODS_PRICE": "Prix du bien financé",
    "AMT_ANNUITY": "Montant de l'annuité",
    "NAME_CONTRACT_TYPE_Revolving_loans": "Type de contrat : Crédit renouvelable",

    # ---- Ratios métier [D] (formules confirmées via kernels publics) ----
    "EMPLOYED_TO_AGE_RATIO": "Ratio ancienneté emploi / âge",
    "CREDIT_GOODS_RATIO": "Ratio crédit / prix du bien",
    "ANNUITY_INCOME_RATIO": "Ratio annuité / revenu (effort financier)",
    "ANNUITY_CREDIT_RATIO": "Ratio annuité / crédit (durée implicite)",

    # ---- Région / logement [O] ----
    "REGION_RATING_CLIENT": "Note de la région du client",
    "REGION_RATING_CLIENT_W_CITY": "Note de la région (avec ville)",
    "REGION_POPULATION_RELATIVE": "Densité de population relative de la région",
    "REG_CITY_NOT_LIVE_CITY": "Adresse d'enregistrement ≠ adresse de résidence",
    "REG_CITY_NOT_WORK_CITY": "Adresse d'enregistrement ≠ adresse professionnelle",

    # ---- Statut familial [O] ----
    "NAME_FAMILY_STATUS_Married": "Statut familial : Marié(e)",
    "NAME_FAMILY_STATUS_Single___not_married": "Statut familial : Célibataire",
    "NAME_FAMILY_STATUS_Civil_marriage": "Statut familial : Union civile",

    # ---- Type de logement [O] ----
    "NAME_HOUSING_TYPE_House___apartment": "Logement : Maison / Appartement",
    "NAME_HOUSING_TYPE_With_parents": "Logement : Chez les parents",
    "NAME_HOUSING_TYPE_Rented_apartment": "Logement : Appartement loué",
    "HOUSETYPE_MODE_block_of_flats": "Type d'habitat : Immeuble collectif",
    "HOUSETYPE_MODE_nan": "Type d'habitat : Non renseigné",
    "WALLSMATERIAL_MODE_Panel": "Matériau des murs : Panneaux",
    "WALLSMATERIAL_MODE_nan": "Matériau des murs : Non renseigné",
    "EMERGENCYSTATE_MODE_No": "Bâtiment en état d'urgence : Non",
    "EMERGENCYSTATE_MODE_nan": "Bâtiment en état d'urgence : Non renseigné",
    "FONDKAPREMONT_MODE_nan": "Fonds de rénovation : Non renseigné",
    "FONDKAPREMONT_MODE_reg_oper_account": "Fonds de rénovation : Compte régulier",

    # ---- Type de revenus [O] ----
    "NAME_INCOME_TYPE_State_servant": "Type de revenu : Fonctionnaire",
    "NAME_INCOME_TYPE_Commercial_associate": "Type de revenu : Salarié du privé",

    # ---- Profession [O] ----
    "OCCUPATION_TYPE_nan": "Profession : Non renseignée",
    "OCCUPATION_TYPE_Core_staff": "Profession : Personnel administratif",
    "OCCUPATION_TYPE_Managers": "Profession : Cadres / Managers",
    "OCCUPATION_TYPE_Accountants": "Profession : Comptables",
    "OCCUPATION_TYPE_High_skill_tech_staff": "Profession : Personnel technique qualifié",
    "OCCUPATION_TYPE_Medicine_staff": "Profession : Personnel médical",
    "OCCUPATION_TYPE_Low_skill_Laborers": "Profession : Ouvriers non qualifiés",
    "OCCUPATION_TYPE_Drivers": "Profession : Chauffeurs",

    # ---- Employeur [O] ----
    "ORGANIZATION_TYPE_School": "Employeur : École",
    "ORGANIZATION_TYPE_Medicine": "Employeur : Secteur médical",
    "ORGANIZATION_TYPE_Government": "Employeur : Gouvernement",
    "ORGANIZATION_TYPE_Kindergarten": "Employeur : École maternelle",
    "ORGANIZATION_TYPE_Police": "Employeur : Police",
    "ORGANIZATION_TYPE_Self_employed": "Employeur : Indépendant",
    "ORGANIZATION_TYPE_Military": "Employeur : Armée",
    "ORGANIZATION_TYPE_Security_Ministries": "Employeur : Ministères de la sécurité",
    "ORGANIZATION_TYPE_Other": "Employeur : Autre",

    # ---- Cercle social (défauts dans l'entourage) [O] ----
    "DEF_30_CNT_SOCIAL_CIRCLE": "Défauts à 30 jours dans l'entourage",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Défauts à 60 jours dans l'entourage",

    # ---- Requêtes au bureau de crédit [O] ----
    "AMT_REQ_CREDIT_BUREAU_MON": "Requêtes au bureau de crédit (dernier mois)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Requêtes au bureau de crédit (dernière année)",

    # =========================================================
    # === Agrégations [A] : préfixe = source, suffixe = stat ===
    # =========================================================

    # ---- BURO : tous les crédits du bureau (actifs + clôturés) ----
    "BURO_DAYS_CREDIT_MIN": "Bureau - Crédit le plus récent (jours, min)",
    "BURO_DAYS_CREDIT_MEAN": "Bureau - Ancienneté moyenne des crédits (jours)",
    "BURO_DAYS_CREDIT_MAX": "Bureau - Crédit le plus ancien (jours, max)",
    "BURO_DAYS_CREDIT_ENDDATE_MIN": "Bureau - Date de fin de crédit la plus proche (jours)",
    "BURO_AMT_CREDIT_SUM_SUM": "Bureau - Somme totale des montants de crédit",
    "BURO_AMT_CREDIT_SUM_MEAN": "Bureau - Montant moyen de crédit",
    "BURO_AMT_CREDIT_SUM_DEBT_SUM": "Bureau - Dette totale en cours",
    "BURO_AMT_CREDIT_SUM_DEBT_MEAN": "Bureau - Dette moyenne en cours",
    "BURO_AMT_CREDIT_SUM_LIMIT_SUM": "Bureau - Somme totale des limites de crédit",

    # ---- CLOSED : crédits du bureau clôturés uniquement ----
    "CLOSED_DAYS_CREDIT_MIN": "Crédits clôturés - Plus récent (jours, min)",
    "CLOSED_DAYS_CREDIT_MEAN": "Crédits clôturés - Ancienneté moyenne (jours)",
    "CLOSED_DAYS_CREDIT_MAX": "Crédits clôturés - Plus ancien (jours, max)",
    "CLOSED_DAYS_CREDIT_ENDDATE_MAX": "Crédits clôturés - Fin la plus récente (jours)",
    "CLOSED_DAYS_CREDIT_ENDDATE_MEAN": "Crédits clôturés - Date moyenne de fin (jours)",
    "CLOSED_AMT_CREDIT_SUM_MEAN": "Crédits clôturés - Montant moyen",
    "CLOSED_AMT_CREDIT_SUM_SUM": "Crédits clôturés - Somme totale",
    "CLOSED_AMT_CREDIT_SUM_DEBT_MEAN": "Crédits clôturés - Dette résiduelle moyenne",
    "CLOSED_AMT_CREDIT_SUM_LIMIT_SUM": "Crédits clôturés - Somme des limites",

    # ---- PREV : précédentes demandes Home Credit ----
    "PREV_DAYS_DECISION_MEAN": "Précédents HC - Ancienneté moyenne de décision (jours)",
    "PREV_AMT_ANNUITY_MEAN": "Précédents HC - Annuité moyenne",
    "PREV_AMT_APPLICATION_SUM": "Précédents HC - Total des montants demandés",
    "PREV_AMT_APPLICATION_MEAN": "Précédents HC - Montant moyen demandé",
    "PREV_AMT_CREDIT_MEAN": "Précédents HC - Montant moyen accordé",
    "PREV_AMT_GOODS_PRICE_MEAN": "Précédents HC - Prix moyen des biens financés",
    "PREV_AMT_DOWN_PAYMENT_MEAN": "Précédents HC - Apport moyen",
    "PREV_APP_CREDIT_PERC_MEAN": "Précédents HC - Ratio moyen demandé / accordé",
    "PREV_CNT_PAYMENT_SUM": "Précédents HC - Total des échéances prévues",

    # ---- POS : Point of Sale (historique mensuel) ----
    "POS_COUNT": "POS - Nombre d'enregistrements historiques",
    "POS_MONTHS_BALANCE_MEAN": "POS - Solde mensuel moyen (mois)",

    # ---- INSTAL : historique des paiements d'échéances ----
    "INSTAL_COUNT": "Échéances - Nombre d'échéances enregistrées",
    "INSTAL_AMT_PAYMENT_SUM": "Échéances - Total des paiements effectués",
    "INSTAL_AMT_PAYMENT_MEAN": "Échéances - Paiement moyen effectué",
    "INSTAL_AMT_INSTALMENT_SUM": "Échéances - Total des montants dus",
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
    <strong style="color:#6b7280;">2. Population de référence</strong> — chargée automatiquement au démarrage<br>
    &nbsp;&nbsp;&nbsp;→ Permet d'afficher la distribution de référence et le scatter plot
</div>
"""

HTML_TRAIN_MISSING = """
<div style="margin-top:1rem; padding:1rem; background:#161920; border:1px solid #252830;
     border-radius:4px; font-size:0.8rem; color:#6b7280;">
    📈 Fichier de référence introuvable — relancez <code>generate_reference_comparaison.py</code>
    pour générer la population de référence.
</div>
"""

HTML_FOOTER = """
<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #1e2028;
     font-size:0.65rem; color:#374151; letter-spacing:0.1em; text-align:center;">
    HOME CREDIT SCORING DASHBOARD · SHAP ANALYSIS · LIGHTGBM
</div>
"""
