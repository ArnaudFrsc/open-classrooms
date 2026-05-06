"""
Génère le fichier de référence pour le dashboard Streamlit P8.

Source : complete_test.parquet — le modèle n'a pas vu ces clients pendant
l'entraînement, donc les prédictions sont plus représentatives de la production.
(Pas de colonne TARGET : le test set Kaggle n'a pas de vraies étiquettes.)

Fichier produit (P8/app/data/test_reference.csv) :
  - SK_ID_CURR       : identifiant client
  - colonnes features : les features envoyées à l'API
  - proba            : probabilité de défaut (modèle)
  - predicted_label  : décision du modèle (0 ou 1)
  - shap_*           : top-N SHAP values par client

Usage :
    python generate_train_reference.py
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ─────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parents[3]          # .../DS
P7_OUTPUT   = ROOT / "P7" / "output_data"
OUTPUT_DIR  = Path(__file__).resolve().parents[1] / "data" # .../P8/app/data
OUTPUT_FILE = OUTPUT_DIR / "test_reference.csv"

# ─────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────

API_URL     = "https://open-classrooms.onrender.com"
MODEL       = "lgb"
THRESHOLD   = 0.3646
N_TOP_SHAP  = 10
BATCH_SIZE  = 200
N_SAMPLE    = 5000    # lignes totales (réparties équitablement entre TARGET=0 et TARGET=1)
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Colonnes features attendues par l'API
# ─────────────────────────────────────────────

cols_path = P7_OUTPUT / "cols_kept.json"
if not cols_path.exists():
    print(f"❌ Fichier introuvable : {cols_path}")
    sys.exit(1)

with open(cols_path) as f:
    FEATURE_COLS = json.load(f)

# ─────────────────────────────────────────────
# 1. Charger le train complet
# ─────────────────────────────────────────────

test_path = P7_OUTPUT / "complete_test.parquet"
if not test_path.exists():
    print(f"❌ Fichier introuvable : {test_path}")
    sys.exit(1)

print(f"📂 Chargement de {test_path.name}...")
df = pd.read_parquet(test_path)

# SK_ID_CURR peut être l'index ou une colonne
if df.index.name == "SK_ID_CURR":
    df = df.reset_index()

if "SK_ID_CURR" not in df.columns:
    print("❌ Colonne SK_ID_CURR introuvable dans le parquet.")
    sys.exit(1)

print(f"   {len(df):,} lignes chargées")

# ─────────────────────────────────────────────
# 2. Nettoyage des noms de colonnes
# ─────────────────────────────────────────────

df.columns = df.columns.str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)

# ─────────────────────────────────────────────
# 3. Échantillonnage aléatoire
# ─────────────────────────────────────────────

n = min(N_SAMPLE, len(df))
df_sample = df.sample(n, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"🎲 Échantillon : {len(df_sample):,} lignes")

# Sauvegarder SK_ID_CURR pour réassembler après l'appel API
sk_ids = df_sample["SK_ID_CURR"].reset_index(drop=True)

# ─────────────────────────────────────────────
# 4. Aligner les features pour l'API
# ─────────────────────────────────────────────

df_for_api = df_sample.reindex(columns=FEATURE_COLS)

missing = [c for c in FEATURE_COLS if c not in df_sample.columns]
if missing:
    print(f"⚠️  {len(missing)} features absentes du parquet (remplies avec NaN) : {missing[:5]}...")

# ─────────────────────────────────────────────
# 5. Appel API /predict/explain par batch
# ─────────────────────────────────────────────

total    = len(df_for_api)
n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
url      = f"{API_URL.rstrip('/')}/predict/explain"
params   = {
    "model":        MODEL,
    "threshold":    THRESHOLD,
    "return_proba": "true",
    "n_top":        N_TOP_SHAP,
}

print(f"\n📤 Appel API : {total:,} lignes en {n_batches} batch(s) de {BATCH_SIZE}")
print(f"   URL : {url}")
print(f"   Modèle : {MODEL} | Seuil : {THRESHOLD} | Top SHAP : {N_TOP_SHAP}\n")

all_results = []
pbar = tqdm(total=total, unit="lignes", desc="SHAP", colour="green")

for i in range(n_batches):
    start = i * BATCH_SIZE
    end   = min(start + BATCH_SIZE, total)
    batch = df_for_api.iloc[start:end]

    csv_bytes = batch.to_csv(index=False).encode("utf-8")

    try:
        resp = requests.post(
            url,
            params=params,
            files={"file": (f"batch_{i}.csv", csv_bytes, "text/csv")},
            timeout=600,
        )
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Erreur réseau batch {i + 1}/{n_batches} : {e}")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"\n❌ Erreur HTTP {resp.status_code} batch {i + 1}/{n_batches} : {resp.text[:300]}")
        sys.exit(1)

    batch_result = pd.read_csv(io.BytesIO(resp.content))
    all_results.append(batch_result)
    pbar.update(end - start)

pbar.close()

# ─────────────────────────────────────────────
# 6. Assembler le fichier final
# ─────────────────────────────────────────────

df_api = pd.concat(all_results, ignore_index=True)

# Les top-N SHAP sont sélectionnés par batch → l'union des batchs peut donner
# plus de N colonnes shap_*, avec des NaN là où la feature n'était pas dans le
# top-N du batch concerné. On réduit à N_TOP_SHAP colonnes par importance globale.
all_shap_cols = [c for c in df_api.columns if c.startswith("shap_")]
global_importance = df_api[all_shap_cols].abs().mean()
top_shap_cols = global_importance.nlargest(N_TOP_SHAP).index.tolist()

api_keep = (
    [c for c in FEATURE_COLS if c in df_api.columns]
    + [c for c in ["predicted_label", "proba"] if c in df_api.columns]
    + top_shap_cols
)
df_api = df_api[api_keep].reset_index(drop=True)

# Construire le DataFrame final : SK_ID_CURR | features | proba | predicted_label | shap_*
df_final = pd.concat(
    [
        pd.DataFrame({"SK_ID_CURR": sk_ids}),
        df_api,
    ],
    axis=1,
)

# ─────────────────────────────────────────────
# 7. Sauvegarder
# ─────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df_final.to_csv(OUTPUT_FILE, index=False)

shap_cols = [c for c in df_final.columns if c.startswith("shap_")]
print(f"\n✅ Fichier généré : {OUTPUT_FILE}")
print(f"   {len(df_final):,} lignes · {len(df_final.columns)} colonnes")
print(f"   SK_ID_CURR | {len(FEATURE_COLS)} features | proba | predicted_label | {len(shap_cols)} SHAP")
