"""
API de prédiction binaire (label 0 ou 1)
Modèles supportés : LightGBM, XGBoost
Entrée  : fichier CSV ou Excel
Sortie  : fichier original enrichi de `predicted_label` (+ `proba`)

Routes :
  POST /predict          → réponse directe (fichier)
  POST /predict/stream   → progression SSE en temps réel, puis fichier encodé en base64
"""

import base64
import io
import json
import os
import time
from typing import AsyncGenerator

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

# ─────────────────────────────────────────────
# Chargement des modèles au démarrage
# ─────────────────────────────────────────────

MODELS_DIR = os.getenv("MODELS_DIR", "models")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))   # lignes par batch pour le streaming

AVAILABLE_MODELS: dict[str, object] = {}

for model_name, filename in [
    ("lgb", "LightGBM_best_model.pkl"),
    ("xgb", "XGBoost_best_model.pkl"),
]:
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        AVAILABLE_MODELS[model_name] = joblib.load(path)
        print(f"✅ Modèle '{model_name}' chargé depuis {path}")
    else:
        print(f"⚠️  Modèle '{model_name}' introuvable à {path} — ignoré")

if not AVAILABLE_MODELS:
    raise RuntimeError(
        f"Aucun modèle trouvé dans '{MODELS_DIR}/'. "
        "Vérifiez que LightGBM_best_model.pkl et/ou XGBoost_best_model.pkl sont présents."
    )

# ─────────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title="ML Prediction API",
    description=(
        "Upload un fichier CSV ou Excel, "
        "récupère le même fichier avec une colonne `predicted_label` (0 ou 1) "
        "et une colonne `proba` (probabilité de la classe 1).\n\n"
        "- **POST /predict** : réponse directe (fichier)\n"
        "- **POST /predict/stream** : progression SSE en temps réel (barre tqdm côté client)"
    ),
    version="1.1.0",
)


# ─────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduit le nettoyage de noms de colonnes fait à l'entraînement."""
    df.columns = df.columns.str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    return df


def _read_upload(file: UploadFile) -> tuple[pd.DataFrame, bytes]:
    """Lit un fichier CSV ou Excel uploadé. Retourne (DataFrame, contenu brut)."""
    content = file.file.read()
    filename = file.filename or ""

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(content)), content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Impossible de lire le CSV : {e}")

    elif filename.endswith((".xlsx", ".xls")):
        engine = "openpyxl" if filename.endswith(".xlsx") else "xlrd"
        try:
            return pd.read_excel(io.BytesIO(content), engine=engine), content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Impossible de lire le fichier Excel : {e}")

    else:
        raise HTTPException(
            status_code=415,
            detail="Format non supporté. Envoyez un fichier .csv, .xlsx ou .xls.",
        )


def _get_expected_features(model) -> list | None:
    """Récupère les noms de features attendus par le modèle."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    elif hasattr(model, "feature_name_"):       # LightGBM
        return list(model.feature_name_())
    elif hasattr(model, "get_booster"):         # XGBoost
        return model.get_booster().feature_names
    return None


def _validate_and_align(df_clean: pd.DataFrame, expected_features: list | None) -> pd.DataFrame:
    """Vérifie que les colonnes sont présentes et les réordonne."""
    if not expected_features:
        return df_clean
    missing = [f for f in expected_features if f not in df_clean.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{len(missing)} colonne(s) attendue(s) absente(s) du fichier : "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            ),
        )
    return df_clean[expected_features]


def _predict_full(model, df_clean: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Prédiction complète en une passe (utilisée par /predict)."""
    expected_features = _get_expected_features(model)
    df_aligned = _validate_and_align(df_clean, expected_features)
    try:
        probas = model.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")
    labels = (probas >= threshold).astype(int)
    return labels, probas


def _serialize(df: pd.DataFrame, filename: str) -> tuple[bytes, str, str]:
    """Sérialise le DataFrame en bytes. Retourne (bytes, media_type, out_filename)."""
    buf = io.BytesIO()
    if filename.endswith(".csv"):
        df.to_csv(buf, index=False)
        media_type = "text/csv"
        out_name = filename.replace(".csv", "_predictions.csv")
    else:
        df.to_excel(buf, index=False, engine="openpyxl")
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        suffix = ".xlsx" if filename.endswith(".xlsx") else ".xls"
        out_name = filename.replace(suffix, "_predictions.xlsx")
    buf.seek(0)
    return buf.read(), media_type, out_name


def _sse(event: str, data: dict) -> str:
    """Formate un message SSE."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─────────────────────────────────────────────
# Générateur SSE (progression par batch)
# ─────────────────────────────────────────────

async def _predict_stream_generator(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    model,
    threshold: float,
    return_proba: bool,
    filename: str,
) -> AsyncGenerator[str, None]:
    """
    Yields des messages SSE :
      - event: progress  → {"processed": N, "total": T, "percent": P, "elapsed": S}
      - event: result    → {"filename": "...", "file_b64": "...", "media_type": "..."}
      - event: error     → {"detail": "..."}
    """
    total = len(df_raw)
    expected_features = _get_expected_features(model)

    try:
        df_clean = _validate_and_align(df_clean, expected_features)
    except HTTPException as e:
        yield _sse("error", {"detail": e.detail})
        return

    all_labels: list = []
    all_probas: list = []
    processed = 0
    start = time.perf_counter()

    # Message initial
    yield _sse("progress", {"processed": 0, "total": total, "percent": 0, "elapsed": 0.0})

    for start_idx in range(0, total, BATCH_SIZE):
        batch = df_clean.iloc[start_idx : start_idx + BATCH_SIZE]

        try:
            probas_batch = model.predict_proba(batch)[:, 1]
        except Exception as e:
            yield _sse("error", {"detail": f"Erreur batch {start_idx}-{start_idx+len(batch)} : {e}"})
            return

        labels_batch = (probas_batch >= threshold).astype(int)
        all_labels.append(labels_batch)
        all_probas.append(probas_batch)

        processed += len(batch)
        elapsed = round(time.perf_counter() - start, 2)
        percent = round(processed / total * 100, 1)

        yield _sse("progress", {
            "processed": processed,
            "total": total,
            "percent": percent,
            "elapsed": elapsed,
        })

    # Assemblage du résultat final
    df_out = df_raw.copy()
    df_out["predicted_label"] = np.concatenate(all_labels)
    if return_proba:
        df_out["proba"] = np.round(np.concatenate(all_probas), 4)

    file_bytes, media_type, out_name = _serialize(df_out, filename)
    file_b64 = base64.b64encode(file_bytes).decode("utf-8")

    yield _sse("result", {
        "filename": out_name,
        "media_type": media_type,
        "file_b64": file_b64,
        "total": total,
        "elapsed": round(time.perf_counter() - start, 2),
    })


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "available_models": list(AVAILABLE_MODELS.keys()),
        "routes": {
            "POST /predict": "Prédiction directe — retourne le fichier enrichi",
            "POST /predict/stream": "Prédiction avec progression SSE (tqdm côté client)",
        },
    }


@app.get("/models", tags=["Health"])
def list_models():
    """Liste les modèles chargés."""
    return {"available_models": list(AVAILABLE_MODELS.keys())}


@app.post("/predict", tags=["Prediction"])
def predict(
    file: UploadFile = File(..., description="Fichier CSV ou Excel à scorer"),
    model: str = Query(default="lgb", description="'lgb' ou 'xgb'"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Seuil de décision"),
    return_proba: bool = Query(default=True, description="Ajouter la colonne 'proba'"),
):
    """Prédiction directe — retourne le fichier enrichi en une seule réponse."""
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle '{model}' inconnu. Disponibles : {list(AVAILABLE_MODELS.keys())}",
        )

    df_raw, _ = _read_upload(file)
    if df_raw.empty:
        raise HTTPException(status_code=422, detail="Le fichier est vide.")

    df_clean = _clean_columns(df_raw.copy())
    labels, probas = _predict_full(AVAILABLE_MODELS[model], df_clean, threshold)

    df_raw["predicted_label"] = labels
    if return_proba:
        df_raw["proba"] = probas.round(4)

    file_bytes, media_type, out_name = _serialize(df_raw, file.filename or "output.csv")
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


@app.post("/predict/stream", tags=["Prediction"])
async def predict_stream(
    file: UploadFile = File(..., description="Fichier CSV ou Excel à scorer"),
    model: str = Query(default="lgb", description="'lgb' ou 'xgb'"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Seuil de décision"),
    return_proba: bool = Query(default=True, description="Ajouter la colonne 'proba'"),
):
    """
    Prédiction avec **progression en temps réel** via Server-Sent Events.

    Le client reçoit des événements SSE :
    - `progress` : avancement batch par batch
    - `result`   : fichier final encodé en base64
    - `error`    : message d'erreur si quelque chose échoue

    Utilisez le script `predict_client.py` fourni pour une barre tqdm automatique.
    """
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle '{model}' inconnu. Disponibles : {list(AVAILABLE_MODELS.keys())}",
        )

    df_raw, _ = _read_upload(file)
    if df_raw.empty:
        raise HTTPException(status_code=422, detail="Le fichier est vide.")

    df_clean = _clean_columns(df_raw.copy())

    return StreamingResponse(
        _predict_stream_generator(
            df_raw=df_raw,
            df_clean=df_clean,
            model=AVAILABLE_MODELS[model],
            threshold=threshold,
            return_proba=return_proba,
            filename=file.filename or "output.csv",
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # désactive le buffering nginx sur Render
        },
    )
