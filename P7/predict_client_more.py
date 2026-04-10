# TODO : Ajouter le pré-traitement des données d'entrée (ex: encodage des variables catégorielles) avant de les envoyer à l'API.

# Code type : ->
# train.columns = train.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
# api_test_sample = train[cols_kept][:30]

"""
Client Python pour l'API ML
Supporte trois routes :
  - /predict          → prédiction directe (fichier)
  - /predict/stream   → progression SSE + tqdm en temps réel
  - /predict/explain  → prédiction + top-N SHAP values par client
"""

import base64
import json
import sys
from pathlib import Path
import requests
from tqdm import tqdm


cols = ['SK_ID_CURR', 'DAYS_EMPLOYED', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'FLAG_DOCUMENT_6', 'NAME_EDUCATION_TYPE', 'EMPLOYED_TO_AGE_RATIO', 
 'PHONE_CHANGE_YEARS', 'CREDIT_GOODS_RATIO', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'OCCUPATION_TYPE_nan', 
 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'FLAG_OWN_CAR', 'RECENT_PHONE_CHANGE', 'NAME_INCOME_TYPE_State_servant', 'EMERGENCYSTATE_MODE_No', 
 'EMERGENCYSTATE_MODE_nan', 'HOUSETYPE_MODE_block_of_flats', 'OCCUPATION_TYPE_Core_staff', 'HOUSETYPE_MODE_nan', 'WALLSMATERIAL_MODE_nan', 
 'NAME_INCOME_TYPE_Commercial_associate', 'HOUR_APPR_PROCESS_START', 'NAME_FAMILY_STATUS_Married', 'OCCUPATION_TYPE_Managers', 'CODE_GENDER', 
 'REGION_POPULATION_RELATIVE', 'NAME_CONTRACT_TYPE_Revolving_loans', 'WALLSMATERIAL_MODE_Panel', 'OCCUPATION_TYPE_Accountants', 
 'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'REG_CITY_NOT_LIVE_CITY', 'FONDKAPREMONT_MODE_nan', 'AMT_ANNUITY', 'ANNUITY_INCOME_RATIO', 
 'AMT_REQ_CREDIT_BUREAU_YEAR', 'OCCUPATION_TYPE_Low_skill_Laborers', 'ORGANIZATION_TYPE_School', 'FONDKAPREMONT_MODE_reg_oper_account', 
 'NAME_FAMILY_STATUS_Single___not_married', 'CNT_FAM_MEMBERS', 'DAYS_ID_PUBLISH', 'FLAG_PHONE', 'ORGANIZATION_TYPE_Medicine', 
 'OCCUPATION_TYPE_High_skill_tech_staff', 'FLAG_DOCUMENT_8', 'OCCUPATION_TYPE_Medicine_staff', 'NAME_FAMILY_STATUS_Civil_marriage', 
 'REG_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE_Military', 'ORGANIZATION_TYPE_Government', 'ANNUITY_CREDIT_RATIO', 'AMT_REQ_CREDIT_BUREAU_MON', 
 'NAME_HOUSING_TYPE_House___apartment', 'REGISTRATION_YEARS', 'ORGANIZATION_TYPE_Other', 'FLAG_DOCUMENT_16', 'NAME_HOUSING_TYPE_With_parents', 
 'ORGANIZATION_TYPE_Kindergarten', 'ORGANIZATION_TYPE_Security_Ministries', 'OCCUPATION_TYPE_Drivers', 'ORGANIZATION_TYPE_Police', 
 'FLAG_DOCUMENT_13', 'DOCUMENT_COUNT', 'NAME_FAMILY_STATUS_Widow', 'FLAG_DOCUMENT_18', 'RECENT_ID_CHANGE', 'CNT_CHILDREN', 
 'OCCUPATION_TYPE_Laborers', 'NAME_HOUSING_TYPE_Rented_apartment', 'ORGANIZATION_TYPE_Bank', 'WALLSMATERIAL_MODE_Stone__brick', 
 'FLAG_DOCUMENT_14', 'ORGANIZATION_TYPE_Transport__type_3', 'ORGANIZATION_TYPE_Self_employed', 'ORGANIZATION_TYPE_Industry__type_9', 
 'ORGANIZATION_TYPE_University', 'FONDKAPREMONT_MODE_org_spec_account', 'FLAG_EMAIL', 'FLAG_DOCUMENT_3', 'ORGANIZATION_TYPE_Construction', 
 'OBS_30_CNT_SOCIAL_CIRCLE', 'OCCUPATION_TYPE_Private_service_staff', 'WALLSMATERIAL_MODE_Monolithic', 'ORGANIZATION_TYPE_Services', 
 'ORGANIZATION_TYPE_Trade__type_6', 'NAME_INCOME_TYPE_Unemployed', 'WALLSMATERIAL_MODE_Block', 'WEEKDAY_APPR_PROCESS_START_SATURDAY', 
 'ORGANIZATION_TYPE_Restaurant', 'ORGANIZATION_TYPE_Business_Entity_Type_2', 'FLAG_DOCUMENT_15', 'ORGANIZATION_TYPE_Transport__type_2', 
 'FONDKAPREMONT_MODE_reg_oper_spec_account', 'NAME_HOUSING_TYPE_Office_apartment', 'ORGANIZATION_TYPE_Electricity', 
 'WEEKDAY_APPR_PROCESS_START_TUESDAY', 'FLAG_OWN_REALTY']

# ─────────────────────────────────────────────
# ⚙️  CONFIGURATION — modifiez ces variables
# ─────────────────────────────────────────────

ROOT_DIR = r'C:\Users\jfurs\Pythonn\OpenClassrooms\DS\P7\data'
P8_DIR = r'C:\Users\jfurs\Pythonn\OpenClassrooms\DS\P8\data'

API_URL      = "https://open-classrooms.onrender.com"   # URL de l'API
INPUT_FILE   = f'{ROOT_DIR}\\api_test_sample.csv'       # Fichier CSV ou Excel
OUTPUT_FILE  = f'{P8_DIR}\\predictions.csv'           # Fichier de sortie (None = nom auto)

MODEL        = "lgb"     # "lgb" ou "xgb"
THRESHOLD    = 0.434     # Seuil de décision (0.0 → 1.0)
RETURN_PROBA = True      # True = ajoute une colonne 'proba'
N_TOP_SHAP   = 10        # Nombre de features SHAP (route /explain uniquement)

# prétraitement : alignement des colonnes d'entrée avec celles attendues par l'API

def align_columns(df, expected_columns):
    """
    Aligns a DataFrame to match the expected columns exactly:
    - Keeps only columns that are in expected_columns
    - Drops any extra columns not in expected_columns
    - Adds missing columns filled with NaN
    - Preserves the order of expected_columns
    """
    return df.reindex(columns=expected_columns)

# ─────────────────────────────────────────────
# Route : /predict  (réponse directe)
# ─────────────────────────────────────────────

def predict_direct(output_file: str | None = None) -> None:
    """Appelle /predict et sauvegarde le fichier retourné directement."""
    filepath = Path(INPUT_FILE)
    if not filepath.exists():
        print(f"❌ Fichier introuvable : {filepath}")
        sys.exit(1)

    url = f"{API_URL.rstrip('/')}/predict"
    params = {
        "model":        MODEL,
        "threshold":    THRESHOLD,
        "return_proba": "true" if RETURN_PROBA else "false",
    }

    print(f"📤 Envoi de '{filepath.name}' vers {url}")
    print(f"   Modèle: {MODEL} | Seuil: {THRESHOLD}\n")

    with open(filepath, "rb") as f:
        response = requests.post(
            url,
            params=params,
            files={"file": (filepath.name, f, _mime(filepath))},
            timeout=600,
        )

    if response.status_code != 200:
        print(f"❌ Erreur HTTP {response.status_code} : {response.text}")
        sys.exit(1)

    out_path = output_file or OUTPUT_FILE
    Path(out_path).write_bytes(response.content)
    print(f"✅ Fichier sauvegardé : {out_path}")


# ─────────────────────────────────────────────
# Route : /predict/stream  (SSE + tqdm)
# ─────────────────────────────────────────────

def predict_with_progress(output_file: str | None = None) -> None:
    """Appelle /predict/stream et affiche une barre tqdm en temps réel."""
    filepath = Path(INPUT_FILE)
    if not filepath.exists():
        print(f"❌ Fichier introuvable : {filepath}")
        sys.exit(1)

    url = f"{API_URL.rstrip('/')}/predict/stream"
    params = {
        "model":        MODEL,
        "threshold":    THRESHOLD,
        "return_proba": "true" if RETURN_PROBA else "false",
    }

    print(f"📤 Envoi de '{filepath.name}' vers {url}")
    print(f"   Modèle: {MODEL} | Seuil: {THRESHOLD}\n")

    with open(filepath, "rb") as f:
        response = requests.post(
            url,
            params=params,
            files={"file": (filepath.name, f, _mime(filepath))},
            stream=True,
            timeout=600,
        )

    if response.status_code != 200:
        print(f"❌ Erreur HTTP {response.status_code} : {response.text}")
        sys.exit(1)

    _consume_sse(response, output_file or OUTPUT_FILE)


# ─────────────────────────────────────────────
# Route : /predict/explain  (SHAP top-N)
# ─────────────────────────────────────────────

def predict_explain(output_file: str | None = None) -> None:
    """
    Appelle /predict/explain et sauvegarde le fichier enrichi de SHAP values.
    Le fichier de sortie contiendra SK_ID_CURR, predicted_label, proba,
    et les colonnes shap_<feature> pour les top-N features.
    """
    filepath = Path(INPUT_FILE)
    if not filepath.exists():
        print(f"❌ Fichier introuvable : {filepath}")
        sys.exit(1)

    url = f"{API_URL.rstrip('/')}/predict/explain"
    params = {
        "model":        MODEL,
        "threshold":    THRESHOLD,
        "return_proba": "true" if RETURN_PROBA else "false",
        "n_top":        N_TOP_SHAP,
    }

    print(f"📤 Envoi de '{filepath.name}' vers {url}")
    print(f"   Modèle: {MODEL} | Seuil: {THRESHOLD} | Top SHAP: {N_TOP_SHAP}\n")
    print("   ⏳ Calcul SHAP en cours — plus lent que /predict…\n")

    with open(filepath, "rb") as f:
        response = requests.post(
            url,
            params=params,
            files={"file": (filepath.name, f, _mime(filepath))},
            timeout=600,
        )

    if response.status_code != 200:
        print(f"❌ Erreur HTTP {response.status_code} : {response.text}")
        sys.exit(1)

    # Détermination du nom de sortie depuis le header Content-Disposition si OUTPUT_FILE est None
    out_path = output_file or OUTPUT_FILE
    if out_path is None:
        cd = response.headers.get("Content-Disposition", "")
        out_path = _extract_filename(cd) or "explained.csv"

    Path(out_path).write_bytes(response.content)
    print(f"✅ Fichier SHAP sauvegardé : {out_path}")
    print(f"   Colonnes disponibles : SK_ID_CURR, predicted_label, proba, shap_<feature> × {N_TOP_SHAP}")


# ─────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────

def _consume_sse(response: requests.Response, out_filename: str) -> None:
    """Lit le flux SSE ligne par ligne et affiche la progression tqdm."""
    pbar = None
    event_name = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            event_name = None
            continue

        if raw_line.startswith("event:"):
            event_name = raw_line.split(":", 1)[1].strip()
            continue

        if raw_line.startswith("data:"):
            payload = json.loads(raw_line.split(":", 1)[1].strip())

            if event_name == "progress":
                total = payload["total"]
                processed = payload["processed"]
                elapsed = payload["elapsed"]

                if pbar is None:
                    pbar = tqdm(
                        total=total,
                        unit="lignes",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        desc="Prédiction",
                        colour="green",
                    )

                pbar.n = processed
                pbar.set_postfix({"⏱": f"{elapsed}s"})
                pbar.refresh()

            elif event_name == "result":
                if pbar:
                    pbar.n = payload["total"]
                    pbar.refresh()
                    pbar.close()

                file_bytes = base64.b64decode(payload["file_b64"])
                Path(out_filename).write_bytes(file_bytes)

                print(f"\n✅ Terminé en {payload['elapsed']}s — {payload['total']} lignes traitées")
                print(f"💾 Fichier sauvegardé : {out_filename}")

            elif event_name == "error":
                if pbar:
                    pbar.close()
                print(f"\n❌ Erreur API : {payload['detail']}")
                sys.exit(1)


def _mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    elif suffix == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"


def _extract_filename(content_disposition: str) -> str | None:
    """Extrait le nom de fichier depuis un header Content-Disposition."""
    for part in content_disposition.split(";"):
        part = part.strip()
        if part.startswith("filename="):
            return part.split("=", 1)[1].strip('"')
    return None


# ─────────────────────────────────────────────
# Main  — choisissez la route à utiliser
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Décommentez la route souhaitée :

    # predict_with_progress()          # /predict/stream  (avec barre de progression)
    # predict_direct()               # /predict         (réponse directe, plus simple)
    predict_explain()              # /predict/explain (prédiction + SHAP top-10)
