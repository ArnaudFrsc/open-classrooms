"""
API de prédiction de capacité de remboursement bancaire.
Déployable sur Render via Uvicorn/Gunicorn.
"""

import logging
import os
import joblib
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
# MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
# MAX_ROWS = int(os.getenv("MAX_ROWS", 10_000))

MODEL_PATH = r"C:\Users\jfurs\Pythonn\OpenClassrooms\DS\P7\mlruns\LightGBM_best_model.pkl"
MAX_ROWS = 10_000

# ── Chargement du modèle (au démarrage) ──────────────────────────────────────
model: Any = None  # sera rempli dans le lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois au démarrage du serveur."""
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error("Fichier modèle introuvable : %s", MODEL_PATH)
        raise RuntimeError(f"Fichier modèle introuvable : {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Modèle chargé depuis %s", MODEL_PATH)
    except Exception as exc:
        logger.exception("Échec du chargement du modèle")
        raise RuntimeError(f"Impossible de charger le modèle : {exc}") from exc
    yield
    # Nettoyage éventuel à l'arrêt
    model = None
    logger.info("Modèle déchargé.")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Default Prediction API",
    description=(
        "Reçoit un DataFrame (liste de lignes JSON) et retourne "
        "la probabilité de défaut de remboursement pour chaque ligne."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Corps de la requête POST /predict."""

    data: list[dict[str, Any]]

    @field_validator("data")
    @classmethod
    def data_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("Le champ 'data' ne peut pas être une liste vide.")
        return v

    @model_validator(mode="after")
    def check_max_rows(self) -> "PredictRequest":
        if len(self.data) > MAX_ROWS:
            raise ValueError(
                f"Trop de lignes : {len(self.data)} reçues, maximum autorisé : {MAX_ROWS}."
            )
        return self


class PredictionResult(BaseModel):
    """Une ligne de résultat."""

    row_index: int
    default_probability: float


class PredictResponse(BaseModel):
    """Réponse complète de /predict."""

    n_rows: int
    predictions: list[PredictionResult]


# ── Gestionnaires d'erreurs globaux ──────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Retourne un 422 lisible en cas d'erreur de validation Pydantic."""
    logger.warning("Erreur de validation : %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Données d'entrée invalides."},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Capture toute exception non gérée et retourne un 500."""
    logger.exception("Erreur interne non gérée")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "Erreur interne du serveur.", "detail": str(exc)},
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", summary="Vérification de l'état du service")
def health() -> dict:
    """Endpoint de health-check utilisé par Render."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Prédiction de probabilité de défaut",
    responses={
        422: {"description": "Données d'entrée invalides"},
        500: {"description": "Erreur interne"},
        503: {"description": "Modèle non disponible"},
    },
)

def predict(request: PredictRequest) -> PredictResponse:
    """
    Reçoit une liste de lignes (dict) représentant un DataFrame,
    et retourne la probabilité de défaut pour chaque ligne.

    - **data** : liste de dictionnaires (une ligne = un dict feature→valeur).
    """
    # 1. Vérifier que le modèle est chargé
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas disponible. Veuillez réessayer plus tard.",
        )

    # 2. Construire le DataFrame
    try:
        df = pd.DataFrame(request.data)
    except Exception as exc:
        logger.error("Impossible de construire le DataFrame : %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Impossible de convertir les données en DataFrame : {exc}",
        ) from exc

    # 3. Vérifier qu'il n'y a pas que des colonnes vides
    if df.empty or df.shape[1] == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Le DataFrame construit est vide (aucune colonne détectée).",
        )

    # 4. Vérifier les valeurs infinies ou 100 % NaN sur une colonne
    if np.isinf(df.select_dtypes(include=[np.number])).any().any():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Les données contiennent des valeurs infinies (inf / -inf).",
        )

    # 5. Inférence
    try:
        # predict_proba retourne [[p_class0, p_class1], ...]
        probas = model.predict_proba(df)
    except AttributeError:
        logger.error("Le modèle ne supporte pas predict_proba.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Le modèle chargé ne supporte pas predict_proba().",
        )
    except ValueError as exc:
        logger.error("Erreur de valeur lors de predict_proba : %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Erreur lors de la prédiction (features incompatibles ?) : {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Erreur inattendue lors de la prédiction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur inattendue lors de la prédiction : {exc}",
        ) from exc

    # 6. Extraire la probabilité de défaut (classe 1)
    if probas.shape[1] < 2:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Le modèle ne retourne pas deux classes (0 et 1).",
        )

    default_probs: list[float] = probas[:, 1].tolist()

    # 7. Construire la réponse
    predictions = [
        PredictionResult(row_index=i, default_probability=round(p, 6))
        for i, p in enumerate(default_probs)
    ]

    logger.info("Prédiction réussie pour %d lignes.", len(predictions))
    return PredictResponse(n_rows=len(predictions), predictions=predictions)
