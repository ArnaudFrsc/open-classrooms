"""
Client Python pour l'API ML
Route :
  - /predict/explain  → prédiction + top-N SHAP values par client
"""

import io
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm
import pandas as pd


P7_OUTPUT_DIR = r'C:\Users\jfurs\Pythonn\OpenClassrooms\DS\P7\output_data'

with open(f'{P7_OUTPUT_DIR}/cols_kept.json') as f:
    _cols_kept = json.load(f)

cols = ['SK_ID_CURR'] + _cols_kept

# ─────────────────────────────────────────────
# ⚙️  CONFIGURATION — modifiez ces variables
# ─────────────────────────────────────────────

P7_OUTPUT_DIR = r'C:\Users\jfurs\Pythonn\OpenClassrooms\DS\P7\output_data'

API_URL      = "https://open-classrooms.onrender.com"       # URL de l'API
INPUT_FILE   = f'{P7_OUTPUT_DIR}\\api_test_sample.csv'   # Fichier CSV ou Excel
OUTPUT_FILE  = f'{P7_OUTPUT_DIR}\\predictions_explained_train.csv' # Fichier de sortie

MODEL        = "lgb"     # "lgb" ou "xgb"
THRESHOLD    = 0.3646     # Seuil de décision (0.0 → 1.0)
RETURN_PROBA = True      # True = ajoute une colonne 'proba'
N_TOP_SHAP   = 10        # Nombre de features SHAP
BATCH_SIZE   = 500       # Nombre de lignes par batch

# ─────────────────────────────────────────────
# Prétraitement
# ─────────────────────────────────────────────

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
            print(f"\nErreur réseau batch {i+1}/{n_batches} : {e}")
            sys.exit(1)

        if response.status_code != 200:
            print(f"\nErreur HTTP {response.status_code} batch {i+1}/{n_batches} : {response.text[:300]}")
            sys.exit(1)

        batch_result = pd.read_csv(io.BytesIO(response.content))
        all_results.append(batch_result)
        pbar.update(end - start)

    pbar.close()

    # Concaténer tous les résultats
    df_final = pd.concat(all_results, ignore_index=True)

    out_path = output_file or OUTPUT_FILE
    df_final.to_csv(out_path, index=False)
    print(f"\nTerminé — {len(df_final)} lignes traitées en {n_batches} batch(s)")
    print(f"💾 Fichier sauvegardé : {out_path}")
    print(f"   Colonnes : SK_ID_CURR, predicted_label, proba, shap_<feature> × {N_TOP_SHAP}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    predict_explain()
