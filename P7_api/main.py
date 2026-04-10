"""
API de prédiction binaire (label 0 ou 1)
Modèles supportés : LightGBM, XGBoost
Entrée  : fichier CSV ou Excel
Sortie  : fichier original enrichi de `predicted_label` (+ `proba`)

Routes :
  POST /predict          → réponse directe (fichier)
  POST /predict/stream   → progression SSE en temps réel, puis fichier encodé en base64
  POST /predict/explain  → prédiction + top-10 SHAP values par client (colonnes shap_<feature>)
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
import shap
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

# ─────────────────────────────────────────────
# Chargement des modèles au démarrage
# ─────────────────────────────────────────────

MODELS_DIR = os.getenv("MODELS_DIR", "models")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))   # lignes par batch pour le streaming

print(f"CWD: {os.getcwd()}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"Exists: {os.path.isdir(MODELS_DIR)}")
if os.path.isdir(MODELS_DIR):
    print(f"Contents: {os.listdir(MODELS_DIR)}")

ID_COLUMN = "SK_ID_CURR"   # colonne identifiant client — jamais droppée

AVAILABLE_MODELS: dict[str, object] = {}
SHAP_EXPLAINERS: dict[str, object] = {}   # explainers SHAP pré-construits par modèle

for model_name, filename in [
    ("lgb", "LightGBM_best_model.pkl"),
    ("xgb", "XGBoost_best_model.pkl"),
]:
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        loaded_model = joblib.load(path)
        AVAILABLE_MODELS[model_name] = loaded_model
        # Pré-construction de l'explainer SHAP (TreeExplainer — compatible LGB & XGB)
        try:
            SHAP_EXPLAINERS[model_name] = shap.TreeExplainer(loaded_model)
            print(f"✅ Modèle '{model_name}' + explainer SHAP chargés depuis {path}")
        except Exception as e:
            print(f"⚠️  Modèle '{model_name}' chargé mais explainer SHAP non disponible : {e}")
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
        "- **POST /predict/stream** : progression SSE en temps réel (barre tqdm côté client)\n"
        "- **POST /predict/explain** : prédiction + top-10 SHAP values par client\n\n"
        f"La colonne `{ID_COLUMN}` est toujours conservée dans le fichier de sortie."
    ),
    version="1.2.0",
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


def _extract_id_column(df: pd.DataFrame) -> pd.Series | None:
    """
    Extrait la colonne ID client si elle est présente.
    Retourne la Series (index aligné) ou None si absente.
    """
    if ID_COLUMN in df.columns:
        return df[ID_COLUMN].copy()
    return None


def _validate_and_align(df_clean: pd.DataFrame, expected_features: list | None) -> pd.DataFrame:
    """
    Vérifie que les features attendues sont présentes et réordonne.
    La colonne ID (SK_ID_CURR) est ignorée si elle n'est pas une feature du modèle.
    """
    if not expected_features:
        return df_clean

    # On retire SK_ID_CURR des colonnes à vérifier — c'est un identifiant, pas une feature
    features_to_check = [f for f in expected_features if f != ID_COLUMN]
    missing = [f for f in features_to_check if f not in df_clean.columns]

    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{len(missing)} colonne(s) attendue(s) absente(s) du fichier : "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            ),
        )
    return df_clean[features_to_check]


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


def _compute_shap_top10(
    explainer,
    df_aligned: pd.DataFrame,
    feature_names: list[str],
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Calcule les SHAP values et retourne un DataFrame avec les top-N features
    (par importance globale sur le batch) sous forme de colonnes shap_<feature>.

    Les colonnes sont triées par |mean SHAP| décroissant sur l'ensemble du batch,
    puis incluses pour chaque ligne.
    """
    try:
        shap_values = explainer.shap_values(df_aligned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul SHAP : {e}")

    # LightGBM peut retourner une liste [class0, class1] — on prend la classe 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Sélection des top-N features par |mean SHAP| sur tout le batch
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:n_top]
    top_features = [feature_names[i] for i in top_indices]

    shap_df = pd.DataFrame(
        shap_values[:, top_indices],
        columns=[f"shap_{f}" for f in top_features],
        index=df_aligned.index,
    )
    return shap_df


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


def _build_output(
    df_raw: pd.DataFrame,
    id_series: pd.Series | None,
    labels: np.ndarray,
    probas: np.ndarray,
    return_proba: bool,
    shap_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Assemble le DataFrame de sortie.
    - SK_ID_CURR est toujours en première colonne si présent dans les données d'origine.
    - Les colonnes SHAP viennent après proba.
    """
    df_out = df_raw.copy()
    df_out["predicted_label"] = labels
    if return_proba:
        df_out["proba"] = np.round(probas, 4)
    if shap_df is not None:
        df_out = pd.concat([df_out.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

    # Réordonner pour mettre SK_ID_CURR en premier si présent
    if id_series is not None and ID_COLUMN in df_out.columns:
        cols = [ID_COLUMN] + [c for c in df_out.columns if c != ID_COLUMN]
        df_out = df_out[cols]

    return df_out


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
    id_series = _extract_id_column(df_clean)

    try:
        df_aligned = _validate_and_align(df_clean, expected_features)
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
        batch = df_aligned.iloc[start_idx : start_idx + BATCH_SIZE]

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
    df_out = _build_output(
        df_raw=df_raw,
        id_series=id_series,
        labels=np.concatenate(all_labels),
        probas=np.concatenate(all_probas),
        return_proba=return_proba,
    )

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
        "id_column_preserved": ID_COLUMN,
        "routes": {
            "POST /predict": "Prédiction directe — retourne le fichier enrichi",
            "POST /predict/stream": "Prédiction avec progression SSE (tqdm côté client)",
            "POST /predict/explain": "Prédiction + top-10 SHAP values par client",
        },
    }


@app.get("/models", tags=["Health"])
def list_models():
    """Liste les modèles chargés et la disponibilité SHAP."""
    return {
        "available_models": list(AVAILABLE_MODELS.keys()),
        "shap_available": list(SHAP_EXPLAINERS.keys()),
    }


@app.post("/predict", tags=["Prediction"])
def predict(
    file: UploadFile = File(..., description="Fichier CSV ou Excel à scorer"),
    model: str = Query(default="lgb", description="'lgb' ou 'xgb'"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Seuil de décision"),
    return_proba: bool = Query(default=True, description="Ajouter la colonne 'proba'"),
):
    """
    Prédiction directe — retourne le fichier enrichi en une seule réponse.
    La colonne SK_ID_CURR est conservée et placée en première position.
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
    id_series = _extract_id_column(df_clean)
    labels, probas = _predict_full(AVAILABLE_MODELS[model], df_clean, threshold)

    df_out = _build_output(
        df_raw=df_raw,
        id_series=id_series,
        labels=labels,
        probas=probas,
        return_proba=return_proba,
    )

    file_bytes, media_type, out_name = _serialize(df_out, file.filename or "output.csv")
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
    La colonne SK_ID_CURR est conservée et placée en première position.

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


@app.post("/predict/explain", tags=["Prediction"])
def predict_explain(
    file: UploadFile = File(..., description="Fichier CSV ou Excel à scorer"),
    model: str = Query(default="lgb", description="'lgb' ou 'xgb'"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Seuil de décision"),
    return_proba: bool = Query(default=True, description="Ajouter la colonne 'proba'"),
    n_top: int = Query(default=10, ge=1, le=50, description="Nombre de features SHAP à inclure"),
):
    """
    Prédiction enrichie d'une **analyse SHAP locale** par client.

    Le fichier de sortie contient, pour chaque ligne :
    - `SK_ID_CURR` : identifiant client (en première colonne)
    - `predicted_label` : 0 ou 1
    - `proba` : probabilité classe 1 (si return_proba=True)
    - `shap_<feature>` × n_top : valeur SHAP des features les plus importantes
       (sélectionnées par |mean SHAP| décroissant sur l'ensemble du batch)

    **Interprétation SHAP** :
    - Valeur positive → pousse la prédiction vers la classe 1 (défaut)
    - Valeur négative → pousse la prédiction vers la classe 0 (non-défaut)
    - |valeur| plus grande → plus d'influence sur la décision

    ⚠️ Plus lent que /predict car le calcul SHAP est intensif.
    """
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle '{model}' inconnu. Disponibles : {list(AVAILABLE_MODELS.keys())}",
        )
    if model not in SHAP_EXPLAINERS:
        raise HTTPException(
            status_code=503,
            detail=f"Explainer SHAP non disponible pour le modèle '{model}'.",
        )

    df_raw, _ = _read_upload(file)
    if df_raw.empty:
        raise HTTPException(status_code=422, detail="Le fichier est vide.")

    df_clean = _clean_columns(df_raw.copy())
    id_series = _extract_id_column(df_clean)

    # Prédiction
    expected_features = _get_expected_features(AVAILABLE_MODELS[model])
    df_aligned = _validate_and_align(df_clean, expected_features)
    feature_names = list(df_aligned.columns)

    try:
        probas = AVAILABLE_MODELS[model].predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")
    labels = (probas >= threshold).astype(int)

    # Calcul SHAP top-N
    shap_df = _compute_shap_top10(
        explainer=SHAP_EXPLAINERS[model],
        df_aligned=df_aligned,
        feature_names=feature_names,
        n_top=n_top,
    )

    # Assemblage
    df_out = _build_output(
        df_raw=df_raw,
        id_series=id_series,
        labels=labels,
        probas=probas,
        return_proba=return_proba,
        shap_df=shap_df,
    )

    file_bytes, media_type, out_name = _serialize(df_out, file.filename or "output.csv")
    out_name = out_name.replace("_predictions.", "_explained.")

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
