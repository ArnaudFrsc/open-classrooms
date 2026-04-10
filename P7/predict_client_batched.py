"""
Client Python pour l'API ML
Supporte une route :
  - /predict/explain  → prédiction + top-N SHAP values par client
"""

import base64
import io
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm
import pandas as pd


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

API_URL      = "https://open-classrooms.onrender.com"       # URL de l'API
# INPUT_FILE   = f'{ROOT_DIR}\\api_test_sample.csv'         # Fichier CSV ou Excel
INPUT_FILE   = f'{P8_DIR}\\sampled_train_for_display.csv'   # Fichier CSV ou Excel
OUTPUT_FILE  = f'{P8_DIR}\\predictions_explained_train.csv' # Fichier de sortie (None = nom auto)

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


def _load_and_align(filepath: Path) -> Path:
    """
    Charge le fichier d'entrée (CSV ou Excel), applique align_columns
    pour correspondre aux colonnes attendues par l'API, nettoie les noms
    de colonnes, et retourne le chemin d'un fichier CSV temporaire prêt
    à être envoyé.
    """
    if not filepath.exists():
        print(f"❌ Fichier introuvable : {filepath}")
        sys.exit(1)

    # Lecture selon le format
    if filepath.suffix.lower() == ".csv":
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Nettoyage des noms de colonnes (même logique que l'entraînement)
    df.columns = df.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)

    # Alignement sur les colonnes attendues par l'API
    df = align_columns(df, cols)

    # Sauvegarde dans un fichier temporaire
    tmp_path = filepath.with_name(filepath.stem + "_aligned.csv")
    df.to_csv(tmp_path, index=False)
    print(f"🔧 Pré-traitement : {len(df)} lignes alignées sur {len(cols)} colonnes → {tmp_path.name}")
    return tmp_path

# ─────────────────────────────────────────────
# Route : /predict/explain  (SHAP top-N)
# ─────────────────────────────────────────────

def predict_explain(output_file: str | None = None) -> None:
    """
    Appelle /predict/explain par batch de BATCH_SIZE lignes,
    concatène les résultats et sauvegarde le fichier final.
    """
    BATCH_SIZE = 500

    filepath = Path(INPUT_FILE)
    aligned_path = _load_and_align(filepath)

    # Charger le CSV aligné en mémoire pour le découper
    df_full = pd.read_csv(aligned_path)
    total_rows = len(df_full)
    n_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

    url = f"{API_URL.rstrip('/')}/predict/explain"
    params = {
        "model":        MODEL,
        "threshold":    THRESHOLD,
        "return_proba": "true" if RETURN_PROBA else "false",
        "n_top":        N_TOP_SHAP,
    }

    print(f"📤 Envoi de {total_rows} lignes vers {url}")
    print(f"   Modèle: {MODEL} | Seuil: {THRESHOLD} | Top SHAP: {N_TOP_SHAP}")
    print(f"   📦 Mode batch : {n_batches} batch(s) de {BATCH_SIZE} lignes max\n")

    all_results = []
    pbar = tqdm(total=total_rows, unit="lignes", desc="Prédiction SHAP", colour="green")

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, total_rows)
        batch_df = df_full.iloc[start:end]

        # Préparer le CSV du batch en mémoire
        csv_bytes = batch_df.to_csv(index=False).encode("utf-8")

        try:
            response = requests.post(
                url,
                params=params,
                files={"file": (f"batch_{i}.csv", csv_bytes, "text/csv")},
                timeout=600,
            )
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Erreur réseau batch {i+1}/{n_batches} : {e}")
            sys.exit(1)

        if response.status_code != 200:
            print(f"\n❌ Erreur HTTP {response.status_code} batch {i+1}/{n_batches} : {response.text[:300]}")
            sys.exit(1)

        batch_result = pd.read_csv(io.BytesIO(response.content))
        all_results.append(batch_result)
        pbar.update(end - start)

    pbar.close()

    # Concaténer tous les résultats
    df_final = pd.concat(all_results, ignore_index=True)

    out_path = output_file or OUTPUT_FILE
    if out_path is None:
        out_path = "explained.csv"

    df_final.to_csv(out_path, index=False)
    print(f"\n✅ Terminé — {len(df_final)} lignes traitées en {n_batches} batch(s)")
    print(f"💾 Fichier sauvegardé : {out_path}")
    print(f"   Colonnes : SK_ID_CURR, predicted_label, proba, shap_<feature> × {N_TOP_SHAP}")


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
    predict_explain()              # /predict/explain (prédiction + SHAP top-10)
