"""
Tests unitaires pour l'API de prédiction de défaut bancaire.
Lancer avec : pytest test_main.py -v

Stratégie : on patche joblib.load dans main.py pour retourner un MagicMock
directement en mémoire — aucune écriture de fichier nécessaire.
"""

import os
import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

MODEL_PATH = r"C:\Users\jfurs\Pythonn\OpenClassrooms\DS\P7\mlruns\LightGBM_best_model.pkl"

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_model(proba_value: float = 0.3):
    """Faux modèle sklearn avec predict_proba pour une ligne."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[1 - proba_value, proba_value]])
    return mock


def make_mock_model_multi_rows(probas: list[float]):
    """Faux modèle retournant plusieurs lignes."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[1 - p, p] for p in probas])
    return mock


# ── Fixture principale ────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path):
    """
    Crée un faux fichier model.pkl vide (pour que os.path.exists() passe),
    puis patche joblib.load pour retourner un MagicMock sans écriture réelle.
    """
    model_file = tmp_path / "model.pkl"
    model_file.write_bytes(b"placeholder")

    MODEL_PATH = str(model_file)

    mock_model = make_mock_model()

    with patch("main.joblib.load", return_value=mock_model):
        import main as main_module
        importlib.reload(main_module)
        with TestClient(main_module.app) as c:
            yield c, main_module


# ── Tests : /health ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, client):
        c, _ = client
        body = c.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


# ── Tests : /predict – cas nominaux ──────────────────────────────────────────

class TestPredictNominal:
    def test_single_row(self, client):
        c, module = client
        module.model = make_mock_model(proba_value=0.42)
        resp = c.post("/predict", json={"data": [{"feat_a": 1, "feat_b": 2.5}]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_rows"] == 1
        assert len(body["predictions"]) == 1
        pred = body["predictions"][0]
        assert pred["row_index"] == 0
        assert abs(pred["default_probability"] - 0.42) < 1e-5

    def test_multiple_rows(self, client):
        c, module = client
        probas = [0.1, 0.5, 0.9]
        module.model = make_mock_model_multi_rows(probas)
        data = [{"feat": i} for i in range(3)]
        resp = c.post("/predict", json={"data": data})
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_rows"] == 3
        for i, p in enumerate(probas):
            assert abs(body["predictions"][i]["default_probability"] - p) < 1e-5

    def test_row_index_is_sequential(self, client):
        c, module = client
        module.model = make_mock_model_multi_rows([0.2, 0.4, 0.6])
        data = [{"x": i} for i in range(3)]
        body = c.post("/predict", json={"data": data}).json()
        indices = [p["row_index"] for p in body["predictions"]]
        assert indices == [0, 1, 2]

    def test_probability_rounded_to_6_decimals(self, client):
        c, module = client
        module.model = make_mock_model(proba_value=1 / 3)
        body = c.post("/predict", json={"data": [{"x": 1}]}).json()
        prob = body["predictions"][0]["default_probability"]
        assert prob == round(1 / 3, 6)

    def test_extreme_probability_zero(self, client):
        c, module = client
        module.model = make_mock_model(proba_value=0.0)
        body = c.post("/predict", json={"data": [{"x": 1}]}).json()
        assert body["predictions"][0]["default_probability"] == 0.0

    def test_extreme_probability_one(self, client):
        c, module = client
        module.model = make_mock_model(proba_value=1.0)
        body = c.post("/predict", json={"data": [{"x": 1}]}).json()
        assert body["predictions"][0]["default_probability"] == 1.0

    def test_mixed_feature_types(self, client):
        c, module = client
        module.model = make_mock_model(0.55)
        data = [{"int_col": 1, "float_col": 3.14, "str_col": "A", "none_col": None}]
        resp = c.post("/predict", json={"data": data})
        assert resp.status_code == 200


# ── Tests : /predict – erreurs d'entrée ──────────────────────────────────────

class TestPredictInputErrors:
    def test_empty_data_list(self, client):
        c, _ = client
        resp = c.post("/predict", json={"data": []})
        assert resp.status_code == 422

    def test_missing_data_field(self, client):
        c, _ = client
        resp = c.post("/predict", json={"rows": [{"x": 1}]})
        assert resp.status_code == 422

    def test_data_is_not_a_list(self, client):
        c, _ = client
        resp = c.post("/predict", json={"data": {"x": 1}})
        assert resp.status_code == 422

    def test_list_of_non_dicts(self, client):
        c, _ = client
        resp = c.post("/predict", json={"data": [1, 2, 3]})
        assert resp.status_code in (422, 500)

    def test_empty_body(self, client):
        c, _ = client
        resp = c.post("/predict", json={})
        assert resp.status_code == 422

    def test_infinite_values(self, client):
        c, module = client
        module.model = make_mock_model()
        data = [{"feat": float("inf")}]
        resp = c.post("/predict", json={"data": data})
        assert resp.status_code == 422
        assert "infinie" in resp.json()["detail"].lower()

    def test_max_rows_exceeded(self, client, monkeypatch):
        c, module = client
        monkeypatch.setattr(module, "MAX_ROWS", 3)
        data = [{"x": i} for i in range(4)]
        resp = c.post("/predict", json={"data": data})
        assert resp.status_code == 422

    def test_max_rows_at_limit_accepted(self, client, monkeypatch):
        c, module = client
        monkeypatch.setattr(module, "MAX_ROWS", 3)
        module.model = make_mock_model_multi_rows([0.1, 0.2, 0.3])
        data = [{"x": i} for i in range(3)]
        resp = c.post("/predict", json={"data": data})
        assert resp.status_code == 200


# ── Tests : /predict – erreurs modèle ────────────────────────────────────────

class TestPredictModelErrors:
    def test_model_not_loaded(self, client):
        c, module = client
        original = module.model
        module.model = None
        resp = c.post("/predict", json={"data": [{"x": 1}]})
        assert resp.status_code == 503
        module.model = original

    def test_model_raises_value_error(self, client):
        c, module = client
        mock = MagicMock()
        mock.predict_proba.side_effect = ValueError("Feature mismatch")
        module.model = mock
        resp = c.post("/predict", json={"data": [{"x": 1}]})
        assert resp.status_code == 422
        assert "Feature mismatch" in resp.json()["detail"]

    def test_model_has_no_predict_proba(self, client):
        c, module = client
        mock = MagicMock(spec=[])  # aucun attribut
        module.model = mock
        resp = c.post("/predict", json={"data": [{"x": 1}]})
        assert resp.status_code == 500

    def test_model_returns_single_class(self, client):
        c, module = client
        mock = MagicMock()
        mock.predict_proba.return_value = np.array([[0.7]])
        module.model = mock
        resp = c.post("/predict", json={"data": [{"x": 1}]})
        assert resp.status_code == 500
        assert "deux classes" in resp.json()["detail"]

    def test_model_raises_unexpected_exception(self, client):
        c, module = client
        mock = MagicMock()
        mock.predict_proba.side_effect = RuntimeError("GPU crash")
        module.model = mock
        resp = c.post("/predict", json={"data": [{"x": 1}]})
        assert resp.status_code == 500


# ── Tests : chargement du modèle ─────────────────────────────────────────────

class TestModelLoading:
    def test_missing_model_file_raises(self, tmp_path):
        """L'API doit échouer au démarrage si le fichier est absent."""
        MODEL_PATH = str(tmp_path / "nonexistent.pkl")
        import main as m
        importlib.reload(m)
        with pytest.raises(Exception):
            with TestClient(m.app):
                pass

    def test_corrupt_model_file_raises(self, tmp_path):
        """L'API doit échouer au démarrage si le fichier n'est pas un modèle valide."""
        bad_file = tmp_path / "bad.pkl"
        bad_file.write_bytes(b"not a valid model")
        MODEL_PATH = str(bad_file)
        import main as m
        importlib.reload(m)
        with pytest.raises(Exception):
            with TestClient(m.app):
                pass
