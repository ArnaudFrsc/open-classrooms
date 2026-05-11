"""
Tests unitaires pour l'API de prédiction binaire (main.py)
Périmètre : route /predict/explain, routes de santé, et fonctions utilitaires.

Exécution :
    pytest tests_unitaires.py -v
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from main import (
    _build_output,
    _clean_columns,
    _extract_id_column,
    _get_expected_features,
    _serialize,
    _validate_and_align,
)

# ─────────────────────────────────────────────
# Fixtures — Mocks du modèle & explainer
# ─────────────────────────────────────────────

FEATURES = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
FIXED_PROBA = 0.7  # proba constante renvoyée par le faux modèle


def _make_fake_model(feature_names=None):
    """Faux modèle sklearn-like : predict_proba renvoie toujours [1-p, p]."""
    model = MagicMock()
    model.predict_proba = MagicMock(
        side_effect=lambda X: np.tile([1 - FIXED_PROBA, FIXED_PROBA], (len(X), 1))
    )
    if feature_names:
        model.feature_names_in_ = np.array(feature_names)
    return model


def _make_fake_explainer(n_features):
    """Faux explainer SHAP renvoyant des valeurs aléatoires."""
    explainer = MagicMock()
    explainer.shap_values = MagicMock(
        side_effect=lambda X: np.random.randn(len(X), n_features)
    )
    return explainer


@pytest.fixture(autouse=True)
def _patch_models():
    """Remplace les dicts globaux du module pour éviter de charger le vrai modèle."""
    fake_model = _make_fake_model(FEATURES)
    fake_explainer = _make_fake_explainer(len(FEATURES))
    with (
        patch.dict("main.AVAILABLE_MODELS", {"lgb": fake_model}, clear=True),
        patch.dict("main.SHAP_EXPLAINERS", {"lgb": fake_explainer}, clear=True),
    ):
        yield


@pytest.fixture()
def client():
    from main import app
    return TestClient(app)


# ─────────────────────────────────────────────
# Helpers — Génération & envoi de fichiers
# ─────────────────────────────────────────────

def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _sample_df(n: int = 5, with_id: bool = True) -> pd.DataFrame:
    """DataFrame d'exemple avec les colonnes attendues."""
    data = {f: np.random.rand(n) for f in FEATURES}
    if with_id:
        data["SK_ID_CURR"] = list(range(100_000, 100_000 + n))
    return pd.DataFrame(data)


def _post_csv(client, url, df=None, filename="data.csv"):
    """Helper : POST d'un DataFrame en CSV vers `url`."""
    df = _sample_df() if df is None else df
    return client.post(url, files={"file": (filename, _csv_bytes(df), "text/csv")})


def _read_csv_response(resp) -> pd.DataFrame:
    """Helper : parse la réponse HTTP comme un CSV."""
    return pd.read_csv(io.BytesIO(resp.content))


# ═════════════════════════════════════════════
# Tests — Routes de santé
# ═════════════════════════════════════════════

class TestHealthRoutes:
    def test_root_returns_status_ok(self, client):
        """GET / → 200 avec status 'ok' et les clés de métadonnées attendues."""
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "available_models" in body
        assert "routes" in body

    def test_root_lists_lgb_model(self, client):
        """GET / expose uniquement le modèle lgb."""
        models = client.get("/").json()["available_models"]
        assert "lgb" in models

    def test_models_endpoint(self, client):
        """GET /models renvoie les modèles dispos et ceux ayant un explainer SHAP."""
        body = client.get("/models").json()
        assert "available_models" in body
        assert "shap_available" in body
        assert "lgb" in body["shap_available"]


# ═════════════════════════════════════════════
# Tests — POST /predict/explain
# ═════════════════════════════════════════════

class TestPredictExplain:
    def test_explain_returns_shap_columns(self, client):
        """Sortie contient des colonnes 'shap_*' — une par feature (n_top par défaut)."""
        resp = _post_csv(client, "/predict/explain")
        assert resp.status_code == 200
        result = _read_csv_response(resp)
        shap_cols = [c for c in result.columns if c.startswith("shap_")]
        # 5 features et n_top=10 par défaut → min(5, 10) = 5
        assert len(shap_cols) == len(FEATURES)

    def test_explain_custom_n_top(self, client):
        """?n_top=2 → seules les 2 features les plus importantes ont leur colonne SHAP."""
        resp = _post_csv(client, "/predict/explain?n_top=2")
        assert resp.status_code == 200
        result = _read_csv_response(resp)
        shap_cols = [c for c in result.columns if c.startswith("shap_")]
        assert len(shap_cols) == 2

    def test_explain_preserves_id(self, client):
        """SK_ID_CURR conservé en première colonne dans la sortie /explain."""
        resp = _post_csv(client, "/predict/explain", _sample_df(with_id=True))
        result = _read_csv_response(resp)
        assert result.columns[0] == "SK_ID_CURR"

    def test_explain_has_predicted_label_and_proba(self, client):
        """/explain renvoie aussi les colonnes prédiction + proba (en plus de SHAP)."""
        resp = _post_csv(client, "/predict/explain")
        result = _read_csv_response(resp)
        assert "predicted_label" in result.columns
        assert "proba" in result.columns

    def test_explain_output_filename(self, client):
        """Le nom de fichier renvoyé contient 'explained'."""
        resp = _post_csv(client, "/predict/explain")
        assert "explained" in resp.headers["content-disposition"]

    def test_explain_empty_file_raises_422(self, client):
        """Fichier vide (aucune ligne) → 422."""
        empty_df = pd.DataFrame(columns=FEATURES)
        resp = _post_csv(client, "/predict/explain", empty_df)
        assert resp.status_code == 422

    def test_explain_missing_features_raises_422(self, client):
        """Fichier sans les colonnes attendues → 422."""
        bad_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        resp = _post_csv(client, "/predict/explain", bad_df)
        assert resp.status_code == 422


# ═════════════════════════════════════════════
# Tests — Fonctions utilitaires internes
# ═════════════════════════════════════════════

class TestUtilities:
    def test_clean_columns(self):
        """_clean_columns : caractères spéciaux (parens, tirets, points) → underscores."""
        df = pd.DataFrame({"col (a)": [1], "col-b": [2], "col.c": [3]})
        cleaned = _clean_columns(df)
        assert list(cleaned.columns) == ["col__a_", "col_b", "col_c"]

    def test_extract_id_column_present(self):
        """_extract_id_column : si SK_ID_CURR existe, renvoie la Series correspondante."""
        df = pd.DataFrame({"SK_ID_CURR": [1, 2], "feat": [3, 4]})
        result = _extract_id_column(df)
        assert result is not None
        assert list(result) == [1, 2]

    def test_extract_id_column_absent(self):
        """_extract_id_column : si SK_ID_CURR absent, renvoie None."""
        df = pd.DataFrame({"feat": [1, 2]})
        assert _extract_id_column(df) is None

    def test_validate_and_align_reorders(self):
        """_validate_and_align : réordonne les colonnes selon l'ordre du modèle."""
        df = pd.DataFrame({"b": [1], "a": [2], "c": [3]})
        result = _validate_and_align(df, ["a", "b", "c"])
        assert list(result.columns) == ["a", "b", "c"]

    def test_validate_and_align_ignores_id(self):
        """_validate_and_align : SK_ID_CURR est ignoré même s'il est dans la liste attendue."""
        df = pd.DataFrame({"feat_a": [1], "feat_b": [2], "SK_ID_CURR": [100]})
        result = _validate_and_align(df, ["feat_a", "feat_b", "SK_ID_CURR"])
        assert "SK_ID_CURR" not in result.columns

    def test_validate_and_align_missing_raises(self):
        """_validate_and_align : feature manquante → HTTPException 422."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(HTTPException) as exc_info:
            _validate_and_align(df, ["a", "b", "c"])
        assert exc_info.value.status_code == 422

    def test_serialize_csv(self):
        """_serialize : extension .csv → media type text/csv et nom suffixé '_predictions'."""
        data, media, name = _serialize(pd.DataFrame({"a": [1, 2]}), "test.csv")
        assert media == "text/csv"
        assert name == "test_predictions.csv"
        assert len(data) > 0

    def test_serialize_xlsx(self):
        """_serialize : extension .xlsx → media type spreadsheet et nom suffixé."""
        _, media, name = _serialize(pd.DataFrame({"a": [1, 2]}), "test.xlsx")
        assert "spreadsheetml" in media
        assert name == "test_predictions.xlsx"

    def test_build_output_with_proba(self):
        """_build_output : ajoute predicted_label et proba ; SK_ID_CURR placé en tête."""
        df_raw = pd.DataFrame({"SK_ID_CURR": [1, 2], "feat": [3, 4]})
        labels = np.array([0, 1])
        probas = np.array([0.3, 0.8])
        result = _build_output(df_raw, df_raw["SK_ID_CURR"].copy(), labels, probas, True)
        assert "predicted_label" in result.columns
        assert "proba" in result.columns
        assert result.columns[0] == "SK_ID_CURR"

    def test_build_output_without_proba(self):
        """_build_output avec return_proba=False : pas de colonne 'proba'."""
        df_raw = pd.DataFrame({"feat": [1, 2]})
        labels = np.array([0, 1])
        probas = np.array([0.3, 0.8])
        result = _build_output(df_raw, None, labels, probas, False)
        assert "proba" not in result.columns

    def test_build_output_with_shap(self):
        """_build_output : si on passe un shap_df, ses colonnes sont concaténées au résultat."""
        df_raw = pd.DataFrame({"feat": [1, 2]})
        labels = np.array([0, 1])
        probas = np.array([0.3, 0.8])
        shap_df = pd.DataFrame({"shap_feat": [0.1, -0.2]})
        result = _build_output(df_raw, None, labels, probas, True, shap_df)
        assert "shap_feat" in result.columns

    def test_get_expected_features_sklearn_style(self):
        """_get_expected_features : lit feature_names_in_ (convention sklearn) → liste."""
        model = MagicMock()
        model.feature_names_in_ = np.array(["a", "b"])
        assert _get_expected_features(model) == ["a", "b"]

    def test_get_expected_features_none(self):
        """_get_expected_features : modèle sans attribut connu → None."""
        model = MagicMock(spec=[])
        assert _get_expected_features(model) is None
