"""
Pipeline Crédit Scoring — Code Complet
=======================================
Préprocessing → IMBLearn Pipeline → SMOTE → LightGBM / XGBoost → MLflow
"""

import os
import re
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# CHEMINS & CONSTANTES
# ══════════════════════════════════════════════════════════════════════
ROOT_DIR     = Path(r"C:\Users\jfurs\Pythonn\OpenClassrooms\DS\P7")
DATA_DIR     = ROOT_DIR / "data"
MODELS_DIR   = ROOT_DIR / "models"
MLRUNS_DIR   = ROOT_DIR / "mlruns"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE       = 42
TARGET_COL         = "TARGET"
N_FEATURES         = 100
CORR_THRESHOLD     = 0.99
NULL_THRESHOLD     = 0.60
MAX_MISSING_ROW    = 1
SMOTE_RATIO        = 0.5     # classe 1 = 50% de classe 0 après SMOTE
BETA               = 2
DECISION_THRESHOLD = 0.35
CV_FOLDS           = 5
N_ITER             = 20
COST_FN            = 10
COST_FP            = 1


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES DE LOG
# ══════════════════════════════════════════════════════════════════════
def _ts():
    return time.strftime("%H:%M:%S")

def _banner(title):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")

def _section(title):
    print(f"\n  ┌─ {title}")

def _step(msg, indent=2):
    print(f"{'  '*indent}[{_ts()}] {msg}")

def _ok(msg, indent=2):
    print(f"{'  '*indent}[{_ts()}] ✓ {msg}")

def _dist(y, label="Distribution"):
    counts = pd.Series(y).value_counts().sort_index()
    total  = len(y)
    print(f"    {label} :")
    for cls, cnt in counts.items():
        bar = "█" * int(cnt / total * 30)
        print(f"      Classe {cls} : {cnt:>7,}  ({cnt/total*100:5.1f}%)  {bar}")


# ══════════════════════════════════════════════════════════════════════
# 1.  CHARGEMENT
# ══════════════════════════════════════════════════════════════════════
def load_data():
    _banner("ÉTAPE 1 / 4  —  Chargement")

    _step("Lecture complete_train.parquet ...")
    t0 = time.time()
    train = pd.read_parquet(DATA_DIR / "complete_train.parquet")
    _ok(f"Train : {train.shape[0]:,} lignes × {train.shape[1]} cols  ({time.time()-t0:.1f}s)")

    _step("Lecture complete_test.parquet ...")
    t0 = time.time()
    test = pd.read_parquet(DATA_DIR / "complete_test.parquet")
    _ok(f"Test  : {test.shape[0]:,} lignes × {test.shape[1]} cols  ({time.time()-t0:.1f}s)")

    _dist(train[TARGET_COL], "Classes dans train brut")
    return train, test


# ══════════════════════════════════════════════════════════════════════
# 2.  SÉLECTION DE FEATURES PAR CORRÉLATION
# ══════════════════════════════════════════════════════════════════════
def select_features_by_correlation(df, target_col, n_features=100,
                                   collinearity_threshold=0.99):
    _step(f"Calcul matrice de corrélation ({df.shape[1]} colonnes) ...", indent=3)
    df_numeric  = df.select_dtypes(include=[np.number, "bool"]).copy()
    corr_matrix = df_numeric.corr()
    target_corr = corr_matrix[target_col].drop(target_col).abs()
    ranked      = target_corr.sort_values(ascending=False).index.tolist()

    selected = []
    for feat in ranked:
        if len(selected) >= n_features:
            break
        if all(abs(df_numeric[[feat, s]].corr().iloc[0, 1]) <= collinearity_threshold
               for s in selected):
            selected.append(feat)
        if len(selected) % 20 == 0 and len(selected) > 0:
            _step(f"{len(selected)}/{n_features} features sélectionnées ...", indent=4)

    removed = [c for c in ranked if c not in selected]
    _ok(f"Sélection terminée : {len(selected)} gardées  |  {len(removed)} supprimées", indent=3)
    return df_numeric[selected + [target_col]], selected, removed


# ══════════════════════════════════════════════════════════════════════
# 3.  PRÉPROCESSING COMPLET
# ══════════════════════════════════════════════════════════════════════
def preprocess(train_raw, test_raw):
    _banner("ÉTAPE 2 / 4  —  Préprocessing")

    # ── 3a. Filtrage colonnes ────────────────────────────────────────
    _section("3a — Filtre colonnes (valeurs manquantes)")
    non_null_ratio = train_raw.notnull().mean()
    cols_keep = non_null_ratio[non_null_ratio >= NULL_THRESHOLD].index
    cols_drop = non_null_ratio[non_null_ratio <  NULL_THRESHOLD].index
    train = train_raw[cols_keep].copy()
    _ok(f"Colonnes gardées : {len(cols_keep)}  |  supprimées : {len(cols_drop)}  (seuil {NULL_THRESHOLD*100:.0f}%)")

    # ── 3b. Nettoyage lignes classe 0 ───────────────────────────────
    _section("3b — Filtre lignes classe 0 (trop incomplètes)")
    df_majority       = train[train[TARGET_COL] != 0]
    df_minority       = train[train[TARGET_COL] == 0]
    missing_per_row   = df_minority.isnull().sum(axis=1)
    df_minority_clean = df_minority[missing_per_row <= MAX_MISSING_ROW]
    n_dropped = len(df_minority) - len(df_minority_clean)
    _step(f"Classe 0 avant : {len(df_minority):,}  →  après : {len(df_minority_clean):,}  ({n_dropped:,} supprimées)")
    train_clean = pd.concat([df_majority, df_minority_clean]).reset_index(drop=True)
    _ok(f"Dataset nettoyé : {train_clean.shape[0]:,} lignes")

    # ── 3c. Sélection de features ────────────────────────────────────
    _section("3c — Sélection de features par corrélation")
    train_reduced, kept_features, removed = select_features_by_correlation(
        train_clean, TARGET_COL, N_FEATURES, CORR_THRESHOLD
    )

    # ── 3d. Drop NaN résiduels classe 0 ─────────────────────────────
    _section("3d — Drop NaN résiduels (classe 0)")
    n_before = len(train_reduced[train_reduced[TARGET_COL] == 0])
    df = pd.concat([
        train_reduced[train_reduced[TARGET_COL] != 0],
        train_reduced[train_reduced[TARGET_COL] == 0].dropna()
    ]).reset_index(drop=True)
    n_after = len(df[df[TARGET_COL] == 0])
    _ok(f"Classe 0 : {n_before:,} → {n_after:,}  ({n_before - n_after:,} lignes droppées)")

    # ── 3e. Nettoyage noms de colonnes ───────────────────────────────
    _section("3e — Nettoyage noms de colonnes (LightGBM / XGBoost)")
    clean_map = {c: re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in df.columns}
    df = df.rename(columns=clean_map)
    kept_features_clean = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in kept_features]
    _ok("Caractères spéciaux → '_'")

    # ── 3f. Split ────────────────────────────────────────────────────
    _section("3f — Split stratifié train / validation (80/20)")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    _ok(f"X_train : {X_train.shape}  |  X_val : {X_val.shape}")
    _dist(y_train, "y_train  (avant SMOTE — dans la pipeline)")
    _dist(y_val,   "y_val    (jamais rééchantillonné)")

    # ── 3g. Test set ─────────────────────────────────────────────────
    _section("3g — Alignement colonnes X_test")
    test_clean = test_raw.copy()
    test_clean.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in test_clean.columns]
    common_cols = [c for c in kept_features_clean if c in test_clean.columns]
    X_test = test_clean[common_cols]
    missing_test = len(kept_features_clean) - len(common_cols)
    _ok(f"X_test : {X_test.shape}  ({missing_test} features absentes du test)")

    return X_train, X_val, X_test, y_train, y_val, kept_features_clean


# ══════════════════════════════════════════════════════════════════════
# 4.  PIPELINE IMBLEARN
# ══════════════════════════════════════════════════════════════════════
def build_imblearn_pipeline(classifier, use_undersampler=False):
    steps = [
        ("imputer",    IterativeImputer(estimator=BayesianRidge(),
                                        max_iter=10, random_state=RANDOM_STATE)),
        ("scaler",     StandardScaler()),
        ("smote",      SMOTE(sampling_strategy=SMOTE_RATIO,
                             k_neighbors=5, random_state=RANDOM_STATE)),
    ]
    if use_undersampler:
        steps.append(("undersampler", RandomUnderSampler(
            sampling_strategy=0.7, random_state=RANDOM_STATE)))
    steps.append(("classifier", classifier))

    pipe = ImbPipeline(steps=steps)
    step_names = " → ".join(s[0] for s in steps)
    _step(f"  {step_names}")
    return pipe


# ══════════════════════════════════════════════════════════════════════
# 5.  ÉVALUATION
# ══════════════════════════════════════════════════════════════════════
def evaluate_model(y_true, y_pred, y_proba, beta=BETA):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc":       roc_auc_score(y_true, y_proba),
        "f_beta":        fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "f1":            f1_score(y_true, y_pred, zero_division=0),
        "recall":        recall_score(y_true, y_pred, zero_division=0),
        "log_loss":      log_loss(y_true, y_proba),
        "business_cost": float(COST_FN * fn + COST_FP * fp),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def log_feature_importance(classifier, feature_names, model_name, top_n=20):
    if not hasattr(classifier, "feature_importances_"):
        return
    fi = pd.Series(
        classifier.feature_importances_,
        index=feature_names[:len(classifier.feature_importances_)]
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    fi.plot(kind="barh", ax=ax, color="#1a6faf")
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Features — {model_name}")
    plt.tight_layout()
    path = MODELS_DIR / f"fi_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="plots")
    _ok(f"Feature importance → {path.name}")


# ══════════════════════════════════════════════════════════════════════
# 6.  ENTRAÎNEMENT + MLFLOW
# ══════════════════════════════════════════════════════════════════════
def train_and_log_model(
    model_name, pipeline, param_grid,
    X_train, y_train, X_val, y_val,
    beta=BETA, n_iter=N_ITER, threshold=DECISION_THRESHOLD,
):
    _banner(f"ÉTAPE 3 / 4  —  Entraînement  [{model_name}]")
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(model_name)
    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── RandomizedSearchCV ──────────────────────────────────────────
    _section("RandomizedSearchCV")
    _step(f"{n_iter} combinaisons × {CV_FOLDS} folds = {n_iter * CV_FOLDS} fits au total")
    _step("Chaque fold applique : Imputer → Scaler → SMOTE → Classifier")
    _step(f"Démarrage  ...")
    t0 = time.time()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=kf,
        refit=True,
        verbose=0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    _ok(f"Recherche terminée en {elapsed:.0f}s  |  "
        f"meilleur CV roc_auc = {search.best_score_:.4f}")
    _step(f"Meilleurs params CV : {search.best_params_}")

    # ── Vérification SMOTE sur le meilleur pipeline ─────────────────
    _section("Vérification distribution après SMOTE")
    _step("Application de la pipeline sans classifieur sur X_train complet ...")
    steps_pre = ImbPipeline(search.best_estimator_.steps[:-1])
    _, y_resampled = steps_pre.fit_resample(X_train, y_train)
    _dist(y_resampled, "y après Imputer → Scaler → SMOTE")

    # ── Re-fit + éval de chaque trial sur X_val ─────────────────────
    _section(f"Éval de chaque trial sur X_val  (seuil={threshold}, β={beta})")
    print(f"\n  {'Trial':>7}  {'f_beta':>7}  {'roc_auc':>8}  "
          f"{'recall':>7}  {'cost':>8}  {'temps':>6}")
    print(f"  {'─'*55}")

    best_fbeta = -np.inf
    best_model = None

    for i, params in enumerate(search.cv_results_["params"]):
        t_trial = time.time()

        pipeline_i = clone(pipeline)
        pipeline_i.set_params(**params)
        pipeline_i.fit(X_train, y_train)

        y_proba = pipeline_i.predict_proba(X_val)[:, 1]
        y_pred  = (y_proba >= threshold).astype(int)
        metrics = evaluate_model(y_val, y_pred, y_proba, beta=beta)

        clean_params = {k.replace("classifier__", ""): v for k, v in params.items()}
        with mlflow.start_run(run_name=f"{model_name}_trial_{i+1:02d}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("threshold",  threshold)
            mlflow.log_param("beta",       beta)
            mlflow.log_params(clean_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline_i, artifact_path="model")

        marker = ""
        if metrics["f_beta"] > best_fbeta:
            best_fbeta = metrics["f_beta"]
            best_model = pipeline_i
            joblib.dump(best_model, MODELS_DIR / f"{model_name}_best.pkl")
            marker = "  ★ NOUVEAU MEILLEUR"

        print(f"  {i+1:>5}/{n_iter:<2}  "
              f"{metrics['f_beta']:>7.4f}  "
              f"{metrics['roc_auc']:>8.4f}  "
              f"{metrics['recall']:>7.4f}  "
              f"{metrics['business_cost']:>8,.0f}  "
              f"{time.time()-t_trial:>5.0f}s"
              f"{marker}")

    # ── Résumé final ─────────────────────────────────────────────────
    _section("Résumé meilleur modèle sur X_val")
    y_proba_best = best_model.predict_proba(X_val)[:, 1]
    y_pred_best  = (y_proba_best >= threshold).astype(int)
    _ok(f"Meilleur f_beta : {best_fbeta:.4f}")
    print()
    print(confusion_matrix(y_val, y_pred_best))
    print()
    print(classification_report(y_val, y_pred_best,
                                target_names=["Bon payeur", "Défaut"]))

    clf = best_model.named_steps["classifier"]
    log_feature_importance(clf, list(X_train.columns), model_name)

    return best_model


# ══════════════════════════════════════════════════════════════════════
# 7.  HYPERPARAMÈTRES
# ══════════════════════════════════════════════════════════════════════
lgb_param_grid = {
    "classifier__n_estimators":      [200, 300, 500, 700],
    "classifier__learning_rate":     [0.01, 0.03, 0.05, 0.1],
    "classifier__num_leaves":        [31, 63, 127],
    "classifier__max_depth":         [-1, 6, 8, 10],
    "classifier__min_child_samples": [20, 50, 100],
    "classifier__subsample":         [0.7, 0.8, 1.0],
    "classifier__colsample_bytree":  [0.7, 0.8, 1.0],
    "classifier__reg_alpha":         [0.0, 0.1, 0.5],
    "classifier__reg_lambda":        [0.0, 0.1, 1.0],
    "classifier__class_weight":      ["balanced"],
}

xgb_param_grid = {
    "classifier__n_estimators":     [200, 300, 500],
    "classifier__learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "classifier__max_depth":        [4, 6, 8],
    "classifier__min_child_weight": [1, 5, 10],
    "classifier__subsample":        [0.7, 0.8, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 1.0],
    "classifier__gamma":            [0, 0.1, 0.5],
    "classifier__reg_alpha":        [0.0, 0.1, 0.5],
    "classifier__reg_lambda":       [1.0, 2.0, 5.0],
}


# ══════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    t_global = time.time()
    _banner("CRÉDIT SCORING — Pipeline complète")
    _step(f"Démarrage  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _step(f"MLflow     → {MLRUNS_DIR}")
    _step(f"Modèles    → {MODELS_DIR}")

    train_raw, test_raw = load_data()

    X_train, X_val, X_test, y_train, y_val, features = preprocess(train_raw, test_raw)

    _banner("Construction des pipelines IMBLearn")
    _step("LightGBM :")
    lgb_pipeline = build_imblearn_pipeline(
        LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
    )
    _step("XGBoost :")
    xgb_pipeline = build_imblearn_pipeline(
        XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                      random_state=RANDOM_STATE, n_jobs=-1)
    )

    best_lgb = train_and_log_model(
        "LightGBM", lgb_pipeline, lgb_param_grid,
        X_train, y_train, X_val, y_val,
    )
    best_xgb = train_and_log_model(
        "XGBoost", xgb_pipeline, xgb_param_grid,
        X_train, y_train, X_val, y_val,
    )

    _banner("ÉTAPE 4 / 4  —  Comparaison finale")
    results = {}
    for name, model in [("LightGBM", best_lgb), ("XGBoost", best_xgb)]:
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred  = (y_proba >= DECISION_THRESHOLD).astype(int)
        results[name] = evaluate_model(y_val, y_pred, y_proba)

    df_cmp = pd.DataFrame(results).T
    print(df_cmp[["roc_auc", "f_beta", "recall", "f1", "business_cost"]].to_string())

    winner = df_cmp["business_cost"].idxmin()
    _ok(f"Meilleur modèle (coût métier) : {winner}")
    _step(f"Durée totale : {(time.time()-t_global)/60:.1f} min")
    _step(f"MLflow UI   : mlflow ui --backend-store-uri {MLRUNS_DIR}")

    return best_lgb, best_xgb


if __name__ == "__main__":
    main()
