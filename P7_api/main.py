"""
API de prédiction binaire (label 0 ou 1)
Modèle : LightGBM
Entrée  : fichier CSV ou Excel
Sortie  : fichier original enrichi de `predicted_label` (+ `proba`) et top-N SHAP values

Route :
  POST /predict/explain  → prédiction + top-N SHAP values par client (colonnes shap_<feature>)
"""

import io
import os

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

# ─────────────────────────────────────────────
# Chargement du modèle au démarrage
# ─────────────────────────────────────────────

MODELS_DIR = os.getenv("MODELS_DIR", "models")

print(f"CWD: {os.getcwd()}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"Exists: {os.path.isdir(MODELS_DIR)}")
if os.path.isdir(MODELS_DIR):
    print(f"Contents: {os.listdir(MODELS_DIR)}")

ID_COLUMN = "SK_ID_CURR"

AVAILABLE_MODELS: dict[str, object] = {}
SHAP_EXPLAINERS: dict[str, object] = {}

_lgb_path = os.path.join(MODELS_DIR, "LightGBM_best_model.pkl")
if os.path.exists(_lgb_path):
    _loaded = joblib.load(_lgb_path)
    AVAILABLE_MODELS["lgb"] = _loaded
    try:
        _shap_model = _loaded
        if hasattr(_loaded, "named_steps"):
            _shap_model = list(_loaded.named_steps.values())[-1]
        SHAP_EXPLAINERS["lgb"] = shap.TreeExplainer(_shap_model)
        print(f"✅ Modèle 'lgb' + explainer SHAP chargés depuis {_lgb_path}")
    except Exception as e:
        print(f"⚠️  Modèle 'lgb' chargé mais explainer SHAP non disponible : {e}")
else:
    print(f"⚠️  Modèle 'lgb' introuvable à {_lgb_path} — ignoré")

if not AVAILABLE_MODELS:
    raise RuntimeError(
        f"Aucun modèle trouvé dans '{MODELS_DIR}/'. "
        "Vérifiez que LightGBM_best_model.pkl est présent."
    )

# ─────────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title="ML Prediction API",
    description=(
        "Upload un fichier CSV ou Excel, "
        "récupère le même fichier avec une colonne `predicted_label` (0 ou 1), "
        "une colonne `proba` (probabilité de la classe 1) et les top-N SHAP values.\n\n"
        "- **POST /predict/explain** : prédiction + top-N SHAP values par client\n\n"
        f"La colonne `{ID_COLUMN}` est toujours conservée dans le fichier de sortie."
    ),
    version="2.0.0",
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
    """Récupère les noms de features attendus par le modèle LightGBM."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    elif hasattr(model, "feature_name_"):
        return list(model.feature_name_())
    return None


def _extract_id_column(df: pd.DataFrame) -> pd.Series | None:
    if ID_COLUMN in df.columns:
        return df[ID_COLUMN].copy()
    return None


def _validate_and_align(df_clean: pd.DataFrame, expected_features: list | None) -> pd.DataFrame:
    """
    Vérifie que les features attendues sont présentes et réordonne.
    SK_ID_CURR est ignoré s'il n'est pas une feature du modèle.
    """
    if not expected_features:
        return df_clean

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


def _transform_for_shap(model_obj, df_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Pour un Pipeline, applique les transformers (imputer, etc.) mais ignore
    les resamplers (SMOTE) qui n'ont pas de sens à l'inférence.
    """
    if not hasattr(model_obj, "named_steps"):
        return df_aligned
    try:
        from imblearn.base import SamplerMixin
        is_sampler = lambda step: isinstance(step, SamplerMixin)
    except ImportError:
        is_sampler = lambda step: False

    X = df_aligned.values.copy()
    steps = list(model_obj.named_steps.items())[:-1]
    for _, step in steps:
        if is_sampler(step):
            continue
        X = step.transform(X)
    return pd.DataFrame(X, columns=df_aligned.columns, index=df_aligned.index)


def _compute_shap_top10(
    explainer,
    df_aligned: pd.DataFrame,
    feature_names: list[str],
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Calcule les SHAP values et retourne un DataFrame avec les top-N features
    (par |mean SHAP| décroissant sur le batch) sous forme de colonnes shap_<feature>.
    """
    try:
        shap_values = explainer.shap_values(df_aligned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul SHAP : {e}")

    # LightGBM peut retourner une liste [class0, class1] — on prend la classe 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:n_top]
    top_features = [feature_names[i] for i in top_indices]

    return pd.DataFrame(
        shap_values[:, top_indices],
        columns=[f"shap_{f}" for f in top_features],
        index=df_aligned.index,
    )


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
    SK_ID_CURR est toujours placé en première colonne si présent dans les données d'origine.
    """
    df_out = df_raw.copy()
    df_out["predicted_label"] = labels
    if return_proba:
        df_out["proba"] = np.round(probas, 4)
    if shap_df is not None:
        df_out = pd.concat([df_out.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

    if id_series is not None and ID_COLUMN in df_out.columns:
        cols = [ID_COLUMN] + [c for c in df_out.columns if c != ID_COLUMN]
        df_out = df_out[cols]

    return df_out


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
            "POST /predict/explain": "Prédiction + top-N SHAP values par client",
        },
    }


@app.get("/models", tags=["Health"])
def list_models():
    """Liste le modèle chargé et la disponibilité SHAP."""
    return {
        "available_models": list(AVAILABLE_MODELS.keys()),
        "shap_available": list(SHAP_EXPLAINERS.keys()),
    }


@app.post("/predict/explain", tags=["Prediction"])
def predict_explain(
    file: UploadFile = File(..., description="Fichier CSV ou Excel à scorer"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Seuil de décision"),
    return_proba: bool = Query(default=True, description="Ajouter la colonne 'proba'"),
    n_top: int = Query(default=10, ge=1, le=50, description="Nombre de features SHAP à inclure"),
):
    """
    Prédiction enrichie d'une **analyse SHAP locale** par client (modèle LightGBM).

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
    """
    model_obj = AVAILABLE_MODELS["lgb"]
    explainer = SHAP_EXPLAINERS.get("lgb")

    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Explainer SHAP non disponible pour le modèle LightGBM.",
        )

    df_raw, _ = _read_upload(file)
    if df_raw.empty:
        raise HTTPException(status_code=422, detail="Le fichier est vide.")

    df_clean = _clean_columns(df_raw.copy())
    id_series = _extract_id_column(df_clean)

    expected_features = _get_expected_features(model_obj)
    df_aligned = _validate_and_align(df_clean, expected_features)
    feature_names = list(df_aligned.columns)

    try:
        probas = model_obj.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")
    labels = (probas >= threshold).astype(int)

    df_for_shap = _transform_for_shap(model_obj, df_aligned)
    shap_df = _compute_shap_top10(
        explainer=explainer,
        df_aligned=df_for_shap,
        feature_names=feature_names,
        n_top=n_top,
    )

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
