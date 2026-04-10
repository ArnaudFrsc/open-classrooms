"""
Tests unitaires pour l'API de prédiction binaire (main.py)

Exécution :
    pytest test_main.py -v
    pytest test_main.py -v --cov=main   (avec couverture)

Dépendances :
    pip install pytest httpx pytest-asyncio
"""

import io
import json
import base64
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────
# Fixtures — Mocks des modèles & explainers
# ─────────────────────────────────────────────

def _make_fake_model(feature_names: list[str] | None = None):
    """Crée un faux modèle avec predict_proba et feature_names_in_."""
    model = MagicMock()
    # predict_proba retourne [[p_class0, p_class1], ...]
    model.predict_proba = MagicMock(
        side_effect=lambda X: np.column_stack([1 - np.full(len(X), 0.7), np.full(len(X), 0.7)])
    )
    if feature_names:
        model.feature_names_in_ = np.array(feature_names)
    return model


def _make_fake_explainer(n_features: int):
    """Crée un faux explainer SHAP."""
    explainer = MagicMock()
    explainer.shap_values = MagicMock(
        side_effect=lambda X: np.random.randn(len(X), n_features)
    )
    return explainer


FEATURES = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]


@pytest.fixture(autouse=True)
def _patch_models():
    """
    Patch les dictionnaires globaux AVAILABLE_MODELS et SHAP_EXPLAINERS
    avant l'import de l'app, pour éviter le chargement réel de modèles.
    """
    fake_model = _make_fake_model(FEATURES)
    fake_explainer = _make_fake_explainer(len(FEATURES))

    with (
        patch.dict("main.AVAILABLE_MODELS", {"lgb": fake_model, "xgb": fake_model}, clear=True),
        patch.dict("main.SHAP_EXPLAINERS", {"lgb": fake_explainer, "xgb": fake_explainer}, clear=True),
    ):
        yield


@pytest.fixture()
def client():
    from main import app
    return TestClient(app)


# ─────────────────────────────────────────────
# Helpers — Génération de fichiers uploadables
# ─────────────────────────────────────────────

def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.read()


def _xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()


def _sample_df(n: int = 5, with_id: bool = True) -> pd.DataFrame:
    """DataFrame d'exemple avec les colonnes attendues."""
    data = {f: np.random.rand(n) for f in FEATURES}
    if with_id:
        data["SK_ID_CURR"] = list(range(100_000, 100_000 + n))
    return pd.DataFrame(data)


# ═════════════════════════════════════════════
# Tests — Routes de santé
# ═════════════════════════════════════════════

class TestHealthRoutes:
    def test_root_returns_status_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "available_models" in body
        assert "routes" in body

    def test_root_lists_models(self, client):
        resp = client.get("/")
        models = resp.json()["available_models"]
        assert "lgb" in models
        assert "xgb" in models

    def test_models_endpoint(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        body = resp.json()
        assert "available_models" in body
        assert "shap_available" in body
        assert "lgb" in body["shap_available"]


# ═════════════════════════════════════════════
# Tests — POST /predict
# ═════════════════════════════════════════════

class TestPredict:
    def test_predict_csv_returns_file(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "predictions" in resp.headers["content-disposition"]

        result = pd.read_csv(io.BytesIO(resp.content))
        assert "predicted_label" in result.columns
        assert "proba" in result.columns
        assert len(result) == len(df)

    def test_predict_xlsx_returns_file(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict",
            files={"file": ("data.xlsx", _xlsx_bytes(df), "application/octet-stream")},
        )
        assert resp.status_code == 200
        assert "spreadsheetml" in resp.headers["content-type"]

    def test_predict_with_xgb_model(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict?model=xgb",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 200

    def test_predict_custom_threshold(self, client):
        """Avec threshold=0.9, toutes les probas à 0.7 → label 0."""
        df = _sample_df()
        resp = client.post(
            "/predict?threshold=0.9",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        result = pd.read_csv(io.BytesIO(resp.content))
        assert (result["predicted_label"] == 0).all()

    def test_predict_threshold_low(self, client):
        """Avec threshold=0.1, toutes les probas à 0.7 → label 1."""
        df = _sample_df()
        resp = client.post(
            "/predict?threshold=0.1",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        result = pd.read_csv(io.BytesIO(resp.content))
        assert (result["predicted_label"] == 1).all()

    def test_predict_without_proba(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict?return_proba=false",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        result = pd.read_csv(io.BytesIO(resp.content))
        assert "predicted_label" in result.columns
        assert "proba" not in result.columns

    def test_predict_preserves_id_column(self, client):
        df = _sample_df(with_id=True)
        resp = client.post(
            "/predict",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        result = pd.read_csv(io.BytesIO(resp.content))
        assert "SK_ID_CURR" in result.columns
        # SK_ID_CURR doit être en première position
        assert result.columns[0] == "SK_ID_CURR"

    def test_predict_without_id_column(self, client):
        df = _sample_df(with_id=False)
        resp = client.post(
            "/predict",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 200
        result = pd.read_csv(io.BytesIO(resp.content))
        assert "SK_ID_CURR" not in result.columns


# ═════════════════════════════════════════════
# Tests — Gestion d'erreurs
# ═════════════════════════════════════════════

class TestPredictErrors:
    def test_unknown_model(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict?model=unknown",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 400
        assert "inconnu" in resp.json()["detail"]

    def test_unsupported_format(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("data.json", b'{"a":1}', "application/json")},
        )
        assert resp.status_code == 415
        assert "Format non supporté" in resp.json()["detail"]

    def test_empty_csv(self, client):
        empty = _csv_bytes(pd.DataFrame(columns=FEATURES))
        resp = client.post(
            "/predict",
            files={"file": ("empty.csv", empty, "text/csv")},
        )
        assert resp.status_code == 422
        assert "vide" in resp.json()["detail"]

    def test_missing_features(self, client):
        """CSV avec des colonnes incorrectes → 422."""
        df = pd.DataFrame({"wrong_col": [1, 2], "another": [3, 4]})
        resp = client.post(
            "/predict",
            files={"file": ("bad.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 422
        assert "absente" in resp.json()["detail"]

    def test_malformed_csv(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("bad.csv", b"not,a,valid\x00csv\xff\xfe", "text/csv")},
        )
        # Doit retourner 400 ou réussir le parsing partiel
        assert resp.status_code in (200, 400, 422)

    def test_threshold_out_of_range(self, client):
        """FastAPI valide ge=0, le=1 → 422 si hors bornes."""
        df = _sample_df()
        resp = client.post(
            "/predict?threshold=1.5",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 422


# ═════════════════════════════════════════════
# Tests — POST /predict/explain
# ═════════════════════════════════════════════

class TestPredictExplain:
    def test_explain_returns_shap_columns(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict/explain",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 200
        result = pd.read_csv(io.BytesIO(resp.content))
        shap_cols = [c for c in result.columns if c.startswith("shap_")]
        # Avec 5 features et n_top=10 par défaut, on a min(5, 10) = 5 shap cols
        assert len(shap_cols) == len(FEATURES)

    def test_explain_custom_n_top(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict/explain?n_top=2",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 200
        result = pd.read_csv(io.BytesIO(resp.content))
        shap_cols = [c for c in result.columns if c.startswith("shap_")]
        assert len(shap_cols) == 2

    def test_explain_preserves_id(self, client):
        df = _sample_df(with_id=True)
        resp = client.post(
            "/predict/explain",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        result = pd.read_csv(io.BytesIO(resp.content))
        assert result.columns[0] == "SK_ID_CURR"

    def test_explain_has_predicted_label_and_proba(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict/explain",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        result = pd.read_csv(io.BytesIO(resp.content))
        assert "predicted_label" in result.columns
        assert "proba" in result.columns

    def test_explain_unknown_model(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict/explain?model=unknown",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 400

    def test_explain_output_filename(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict/explain",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert "explained" in resp.headers["content-disposition"]


# ═════════════════════════════════════════════
# Tests — POST /predict/stream
# ═════════════════════════════════════════════

class TestPredictStream:
    def test_stream_returns_sse(self, client):
        df = _sample_df(n=10)
        resp = client.post(
            "/predict/stream",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_contains_progress_and_result(self, client):
        df = _sample_df(n=10)
        resp = client.post(
            "/predict/stream",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        body = resp.text
        assert "event: progress" in body
        assert "event: result" in body

    def test_stream_result_is_valid_base64(self, client):
        df = _sample_df(n=5)
        resp = client.post(
            "/predict/stream",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        # Extraire le dernier événement "result"
        for line in resp.text.split("\n"):
            if line.startswith("data:") and "file_b64" in line:
                data = json.loads(line[len("data:"):])
                decoded = base64.b64decode(data["file_b64"])
                result = pd.read_csv(io.BytesIO(decoded))
                assert "predicted_label" in result.columns
                assert len(result) == len(df)
                break
        else:
            pytest.fail("Aucun événement 'result' avec 'file_b64' trouvé dans le flux SSE")

    def test_stream_unknown_model(self, client):
        df = _sample_df()
        resp = client.post(
            "/predict/stream?model=unknown",
            files={"file": ("data.csv", _csv_bytes(df), "text/csv")},
        )
        assert resp.status_code == 400

    def test_stream_empty_file(self, client):
        empty = _csv_bytes(pd.DataFrame(columns=FEATURES))
        resp = client.post(
            "/predict/stream",
            files={"file": ("empty.csv", empty, "text/csv")},
        )
        assert resp.status_code == 422


# ═════════════════════════════════════════════
# Tests — Fonctions utilitaires internes
# ═════════════════════════════════════════════

class TestUtilities:
    def test_clean_columns(self):
        from main import _clean_columns
        df = pd.DataFrame({"col (a)": [1], "col-b": [2], "col.c": [3]})
        cleaned = _clean_columns(df)
        assert list(cleaned.columns) == ["col__a_", "col_b", "col_c"]

    def test_extract_id_column_present(self):
        from main import _extract_id_column
        df = pd.DataFrame({"SK_ID_CURR": [1, 2], "feat": [3, 4]})
        result = _extract_id_column(df)
        assert result is not None
        assert list(result) == [1, 2]

    def test_extract_id_column_absent(self):
        from main import _extract_id_column
        df = pd.DataFrame({"feat": [1, 2]})
        result = _extract_id_column(df)
        assert result is None

    def test_validate_and_align_reorders(self):
        from main import _validate_and_align
        df = pd.DataFrame({"b": [1], "a": [2], "c": [3]})
        result = _validate_and_align(df, ["a", "b", "c"])
        assert list(result.columns) == ["a", "b", "c"]

    def test_validate_and_align_ignores_id(self):
        from main import _validate_and_align
        df = pd.DataFrame({"feat_a": [1], "feat_b": [2], "SK_ID_CURR": [100]})
        # SK_ID_CURR dans les features attendues → ignoré
        result = _validate_and_align(df, ["feat_a", "feat_b", "SK_ID_CURR"])
        assert "SK_ID_CURR" not in result.columns

    def test_validate_and_align_missing_raises(self):
        from main import _validate_and_align
        from fastapi import HTTPException
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(HTTPException) as exc_info:
            _validate_and_align(df, ["a", "b", "c"])
        assert exc_info.value.status_code == 422

    def test_serialize_csv(self):
        from main import _serialize
        df = pd.DataFrame({"a": [1, 2]})
        data, media, name = _serialize(df, "test.csv")
        assert media == "text/csv"
        assert name == "test_predictions.csv"
        assert len(data) > 0

    def test_serialize_xlsx(self):
        from main import _serialize
        df = pd.DataFrame({"a": [1, 2]})
        data, media, name = _serialize(df, "test.xlsx")
        assert "spreadsheetml" in media
        assert name == "test_predictions.xlsx"

    def test_sse_format(self):
        from main import _sse
        result = _sse("progress", {"processed": 5, "total": 10})
        assert result.startswith("event: progress\n")
        assert "data: " in result
        parsed = json.loads(result.split("data: ")[1].strip())
        assert parsed["processed"] == 5

    def test_build_output_with_proba(self):
        from main import _build_output
        df_raw = pd.DataFrame({"SK_ID_CURR": [1, 2], "feat": [3, 4]})
        labels = np.array([0, 1])
        probas = np.array([0.3, 0.8])
        result = _build_output(df_raw, df_raw["SK_ID_CURR"].copy(), labels, probas, True)
        assert "predicted_label" in result.columns
        assert "proba" in result.columns
        assert result.columns[0] == "SK_ID_CURR"

    def test_build_output_without_proba(self):
        from main import _build_output
        df_raw = pd.DataFrame({"feat": [1, 2]})
        labels = np.array([0, 1])
        probas = np.array([0.3, 0.8])
        result = _build_output(df_raw, None, labels, probas, False)
        assert "proba" not in result.columns

    def test_build_output_with_shap(self):
        from main import _build_output
        df_raw = pd.DataFrame({"feat": [1, 2]})
        labels = np.array([0, 1])
        probas = np.array([0.3, 0.8])
        shap_df = pd.DataFrame({"shap_feat": [0.1, -0.2]})
        result = _build_output(df_raw, None, labels, probas, True, shap_df)
        assert "shap_feat" in result.columns

    def test_get_expected_features_sklearn_style(self):
        from main import _get_expected_features
        model = MagicMock()
        model.feature_names_in_ = np.array(["a", "b"])
        assert _get_expected_features(model) == ["a", "b"]

    def test_get_expected_features_none(self):
        from main import _get_expected_features
        model = MagicMock(spec=[])  # aucun attribut
        assert _get_expected_features(model) is None
