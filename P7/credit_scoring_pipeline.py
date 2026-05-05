"""
Pipeline Crédit Scoring — aligné sur ml_research.ipynb
=======================================================
Préprocessing → Undersampling manuel → Sélection features →
SimpleImputer → SMOTE → [StandardScaler] → LR / XGBoost / LightGBM → MLflow (nested runs)
"""

import time
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# CHEMINS & CONSTANTES
# ══════════════════════════════════════════════════════════════════════
ROOT_DIR    = Path(r"C:\Users\jfurs\Pythonn\OpenClassrooms\DS\P7")
DATA_DIR    = ROOT_DIR / "output_data"
MODELS_DIR  = ROOT_DIR / "models"
MLRUNS_DIR  = ROOT_DIR / "mlruns"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE    = 42
TARGET_COL      = "TARGET"
N_FEATURES      = 100
CORR_THRESHOLD  = 0.99
NULL_THRESHOLD  = 0.60
MAX_MISSING_ROW = 1
SMOTE_RATIO     = 1.5   # n1_target = class1_count × SMOTE_RATIO
BETA            = 2
CV_FOLDS        = 5
N_ITER          = 20
COST_FN         = 10
COST_FP         = 2
COST_TP         = 4
COST_TN         = 1


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
# MÉTRIQUES MÉTIER
# ══════════════════════════════════════════════════════════════════════
def business_cost(y_true, y_proba, threshold,
                  cost_fn=COST_FN, cost_fp=COST_FP,
                  reward_tp=COST_TP, reward_tn=COST_TN):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    losses = (fn * cost_fn) + (fp * cost_fp)
    gains  = (tp * reward_tp) + (tn * reward_tn)
    return losses - gains


def find_optimal_threshold(y_true, y_proba,
                           cost_fn=COST_FN, cost_fp=COST_FP,
                           reward_tp=COST_TP, reward_tn=COST_TN):
    thresholds = np.linspace(0.01, 0.99, 200)
    results = [
        business_cost(y_true, y_proba, t, cost_fn, cost_fp, reward_tp, reward_tn)
        for t in thresholds
    ]
    best_idx = np.argmin(results)
    return thresholds[best_idx], results, thresholds


def evaluate_model_from_pred(y_true, y_pred, y_proba, beta=BETA):
    return {
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f_beta":    fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba),
    }


# ══════════════════════════════════════════════════════════════════════
# 1.  CHARGEMENT
# ══════════════════════════════════════════════════════════════════════
def load_data():
    _banner("ÉTAPE 1 / 3  —  Chargement")

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
def select_features_by_correlation(df, target_col, n_features=N_FEATURES,
                                   collinearity_threshold=CORR_THRESHOLD):
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

    removed = [c for c in ranked if c not in selected]
    _ok(f"Features gardées : {len(selected)}  |  supprimées : {len(removed)}", indent=3)
    return df_numeric[selected + [target_col]], selected, removed


# ══════════════════════════════════════════════════════════════════════
# 3.  PRÉPROCESSING COMPLET
# ══════════════════════════════════════════════════════════════════════
def preprocess(train_raw, test_raw):
    _banner("ÉTAPE 2 / 3  —  Préprocessing")

    # ── a. Filtrage colonnes ─────────────────────────────────────────
    _section("a — Filtre colonnes (valeurs manquantes)")
    non_null_ratio = train_raw.notnull().mean()
    cols_keep = non_null_ratio[non_null_ratio >= NULL_THRESHOLD].index
    cols_drop = non_null_ratio[non_null_ratio <  NULL_THRESHOLD].index
    train = train_raw[cols_keep].copy()
    _ok(f"Colonnes gardées : {len(cols_keep)}  |  supprimées : {len(cols_drop)}  (seuil {NULL_THRESHOLD*100:.0f}%)")

    # ── b. Filtrage lignes classe 0 (trop de NaN) ───────────────────
    _section("b — Filtre lignes classe 0 (> 1 valeur manquante)")
    df_majority       = train[train[TARGET_COL] != 0]
    df_minority       = train[train[TARGET_COL] == 0]
    missing_per_row   = df_minority.isnull().sum(axis=1)
    df_minority_clean = df_minority[missing_per_row <= MAX_MISSING_ROW]
    n_dropped = len(df_minority) - len(df_minority_clean)
    _step(f"Classe 0 : {len(df_minority):,} → {len(df_minority_clean):,}  ({n_dropped:,} supprimées)")
    train_clean = pd.concat([df_majority, df_minority_clean])
    _ok(f"Dataset nettoyé : {train_clean.shape[0]:,} lignes")
    _dist(train_clean[TARGET_COL])

    # ── c. Nettoyage noms de colonnes ───────────────────────────────
    _section("c — Nettoyage noms de colonnes")
    train_clean.columns = train_clean.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
    test_raw    = test_raw.copy()
    test_raw.columns = test_raw.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
    _ok("Caractères spéciaux → '_'")

    # ── d. Split stratifié train / validation (80/20) ───────────────
    _section("d — Split stratifié train / validation (80/20)")
    X = train_clean.drop(columns=[TARGET_COL])
    y = train_clean[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    _ok(f"X_train : {X_train.shape}  |  X_val : {X_val.shape}")

    # ── e. Nettoyage NaN classe 0 dans X_train ──────────────────────
    _section("e — Suppression lignes classe 0 avec NaN dans X_train")
    X_train[TARGET_COL] = y_train
    X_train = pd.concat([
        X_train[X_train[TARGET_COL] != 0],
        X_train[X_train[TARGET_COL] == 0].dropna(),
    ])
    y_train = y_train.loc[X_train.index]
    _ok(f"X_train après nettoyage : {X_train.shape[0]:,} lignes")

    # ── f. Undersampling manuel : classe 0 → 2× classe 1 ────────────
    _section("f — Undersampling (classe 0 → 2× classe 1)")
    X_train_maj = X_train[X_train[TARGET_COL] == 0]
    X_train_min = X_train[X_train[TARGET_COL] == 1]
    X_train_maj_down = X_train_maj.sample(n=2 * len(X_train_min), random_state=RANDOM_STATE)
    X_train_balanced = pd.concat([X_train_maj_down, X_train_min]).sample(
        frac=1, random_state=RANDOM_STATE
    )
    y_train = y_train.loc[X_train_balanced.index]
    _ok(f"X_train_balanced : {X_train_balanced.shape[0]:,} lignes")
    _dist(y_train, "y_train après undersampling")

    # ── g. Sélection de features par corrélation ─────────────────────
    _section("g — Sélection de features par corrélation")
    _step(f"Calcul matrice de corrélation ({X_train_balanced.shape[1]} colonnes) ...", indent=3)
    X_train_reduced, cols_kept, _ = select_features_by_correlation(
        X_train_balanced, TARGET_COL, N_FEATURES, CORR_THRESHOLD
    )
    X_train_reduced.drop(columns=[TARGET_COL], inplace=True)
    X_train_reduced.sort_index(inplace=True)
    y_train.sort_index(inplace=True)

    # ── h. Alignement val / test ─────────────────────────────────────
    _section("h — Alignement val / test sur cols_kept")
    val_reduced  = X_val[cols_kept]
    test_reduced = test_raw[cols_kept]
    _ok(f"X_train_reduced : {X_train_reduced.shape}  |  val : {val_reduced.shape}  |  test : {test_reduced.shape}")

    _step(f"NaN X_train : {X_train_reduced.isna().sum().sum():,}  "
          f"|  NaN val : {val_reduced.isna().sum().sum():,}")

    return X_train_reduced, val_reduced, test_reduced, y_train, y_val


# ══════════════════════════════════════════════════════════════════════
# 4.  ENTRAÎNEMENT + MLFLOW
# ══════════════════════════════════════════════════════════════════════
def train_and_log_model(
    model_name, model, param_grid,
    X_train, y_train, X_val, y_val,
    beta=BETA, n_iter=N_ITER,
    cost_fn=COST_FN, cost_fp=COST_FP,
    reward_tp=COST_TP, reward_tn=COST_TN,
    smote_ratio=SMOTE_RATIO,
    scale=False,
):
    _banner(f"ENTRAÎNEMENT  [{model_name}]")
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(model_name)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"  {n_iter} combinaisons × {CV_FOLDS} folds")
    print(f"  Coûts → FN: {cost_fn} | FP: {cost_fp} | TP reward: {reward_tp} | TN reward: {reward_tn} | Beta: {beta}")

    # ── Construction de la pipeline ──────────────────────────────────
    n1_target = int((y_train == 1).sum() * smote_ratio)

    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scaler', StandardScaler()))
    steps += [
        ('smote', SMOTE(sampling_strategy={1: n1_target}, random_state=RANDOM_STATE)),
        ('model', model),
    ]
    pipeline = Pipeline(steps=steps)

    # ── RandomizedSearchCV (refit=False — on re-fit manuellement) ───
    _section("RandomizedSearchCV")
    _step(f"scoring=roc_auc | refit=False | n_jobs=-1", indent=3)
    t0 = time.time()

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=skf,
        refit=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )
    random_search.fit(X_train, y_train)
    _ok(f"Recherche terminée en {time.time()-t0:.0f}s", indent=3)

    # ── Éval de chaque trial sur X_val ──────────────────────────────
    _section(f"Éval des {n_iter} trials sur X_val (threshold optimal par coût métier)")
    print(f"\n  {'Trial':>7}  {'Threshold':>9}  {'F'+str(beta):>7}  "
          f"{'Recall':>7}  {'AUC':>7}  {'CV AUC':>10}")
    print(f"  {'─'*60}")

    best_fbeta    = -np.inf
    best_pipeline = None
    best_cm       = None

    with mlflow.start_run(run_name=f"{model_name}_search"):

        mlflow.log_params({
            "model_type":  model_name,
            "n_features":  X_train.shape[1],
            "n_train":     len(X_train),
            "smote_ratio": smote_ratio,
            "scale":       scale,
            "beta":        beta,
            "n_iter":      n_iter,
            "cost_fn":     cost_fn,
            "cost_fp":     cost_fp,
            "reward_tp":   reward_tp,
            "reward_tn":   reward_tn,
        })

        for i, params in enumerate(random_search.cv_results_["params"]):

            pipeline_i = clone(pipeline).set_params(**params)
            pipeline_i.fit(X_train, y_train)

            y_proba = pipeline_i.predict_proba(X_val)[:, 1]

            optimal_threshold, _, _ = find_optimal_threshold(
                y_val, y_proba,
                cost_fn=cost_fn, cost_fp=cost_fp,
                reward_tp=reward_tp, reward_tn=reward_tn,
            )

            y_pred  = (y_proba >= optimal_threshold).astype(int)
            metrics = evaluate_model_from_pred(y_val, y_pred, y_proba, beta=beta)
            cm      = confusion_matrix(y_val, y_pred)
            tn, fp, fn, tp = cm.ravel()

            bc          = business_cost(y_val, y_proba, optimal_threshold, cost_fn, cost_fp, reward_tp, reward_tn)
            cv_auc_mean = random_search.cv_results_["mean_test_score"][i]
            cv_auc_std  = random_search.cv_results_["std_test_score"][i]

            is_best = metrics["f_beta"] > best_fbeta
            marker  = "  ⭐ MEILLEUR" if is_best else ""

            print(f"  {i+1:>5}/{n_iter:<2}  "
                  f"{optimal_threshold:>9.3f}  "
                  f"{metrics['f_beta']:>7.4f}  "
                  f"{metrics['recall']:>7.4f}  "
                  f"{metrics['roc_auc']:>7.4f}  "
                  f"{cv_auc_mean:.4f} ± {cv_auc_std:.4f}"
                  f"{marker}")

            with mlflow.start_run(run_name=f"trial_{i+1}", nested=True):
                mlflow.log_param("optimal_threshold", optimal_threshold)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_metrics({
                    "business_cost": bc,
                    "cv_auc_mean":   cv_auc_mean,
                    "cv_auc_std":    cv_auc_std,
                    "tp":            int(tp),
                    "tn":            int(tn),
                    "fp":            int(fp),
                    "fn":            int(fn),
                })
                mlflow.set_tag("is_best", is_best)

            if is_best:
                best_fbeta    = metrics["f_beta"]
                best_pipeline = pipeline_i
                best_cm       = cm

        mlflow.sklearn.log_model(best_pipeline, name="best_model")
        mlflow.log_metric("best_f_beta", best_fbeta)

    print(f"\n  ✓ {model_name} terminé — meilleur F{beta} : {best_fbeta:.4f}")

    output_path = MLRUNS_DIR / f"{model_name}_best_model.pkl"
    joblib.dump(best_pipeline, output_path)
    _ok(f"Modèle sauvegardé → {output_path.name}")

    return best_pipeline, best_cm


# ══════════════════════════════════════════════════════════════════════
# 5.  HYPERPARAMÈTRES
# ══════════════════════════════════════════════════════════════════════
lr_param_grid = {
    'model__C':        [0.001, 0.01, 0.1, 1, 10, 100],
    'model__penalty':  ['l1', 'l2'],
    'model__solver':   ['liblinear'],
    'model__max_iter': [1000],
}

lgb_param_grid = {
    'model__n_estimators':   [100, 200, 500],
    'model__learning_rate':  [0.01, 0.05, 0.1],
    'model__num_leaves':     [31, 50, 100],
    'model__subsample':      [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
}

xgb_param_grid = {
    'model__n_estimators':      [100, 200, 500],
    'model__learning_rate':     [0.01, 0.05, 0.1],
    'model__max_depth':         [3, 5, 7],
    'model__subsample':         [0.6, 0.8, 1.0],
    'model__colsample_bytree':  [0.6, 0.8, 1.0],
    'model__scale_pos_weight':  [1, 5, 10],
}


# ══════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    t_global = time.time()
    _banner("CRÉDIT SCORING — Pipeline complète")
    _step(f"Démarrage  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _step(f"MLflow     → {MLRUNS_DIR}")
    _step(f"Modèles    → {MODELS_DIR}")

    train_raw, test_raw = load_data()

    X_train, X_val, X_test, y_train, y_val = preprocess(train_raw, test_raw)

    # ── Baseline DummyClassifier ─────────────────────────────────────
    _banner("BASELINE — DummyClassifier")
    dummy = DummyClassifier(strategy='prior', random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    y_proba_dummy = dummy.predict_proba(X_val)[:, 1]
    threshold_dummy, _, _ = find_optimal_threshold(y_val, y_proba_dummy)
    y_pred_dummy  = (y_proba_dummy >= threshold_dummy).astype(int)
    metrics_dummy = evaluate_model_from_pred(y_val, y_pred_dummy, y_proba_dummy)
    bc_dummy      = business_cost(y_val, y_proba_dummy, threshold_dummy)
    print(f"  Recall: {metrics_dummy['recall']:.4f}  |  F{BETA}: {metrics_dummy['f_beta']:.4f}  "
          f"|  AUC: {metrics_dummy['roc_auc']:.4f}  |  Business cost: {bc_dummy:.0f}")

    # ── Entraînement des modèles ─────────────────────────────────────
    lr_model, _ = train_and_log_model(
        model_name = "LogisticRegression",
        model      = LogisticRegression(random_state=RANDOM_STATE),
        param_grid = lr_param_grid,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        scale=True,
    )

    xgb_model, _ = train_and_log_model(
        model_name = "XGBoost",
        model      = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        param_grid = xgb_param_grid,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
    )

    lgb_model, _ = train_and_log_model(
        model_name = "LightGBM",
        model      = LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
        param_grid = lgb_param_grid,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
    )

    # ── Récap final ──────────────────────────────────────────────────
    _banner("RÉCAP FINAL — classé par F2 score")
    recap_rows = []
    for name, model in [("LightGBM", lgb_model), ("XGBoost", xgb_model), ("LogisticRegression", lr_model)]:
        y_proba = model.predict_proba(X_val)[:, 1]
        threshold, _, _ = find_optimal_threshold(y_val, y_proba)
        y_pred  = (y_proba >= threshold).astype(int)
        metrics = evaluate_model_from_pred(y_val, y_pred, y_proba)
        bc      = business_cost(y_val, y_proba, threshold)
        recap_rows.append({
            "Model":          name,
            f"F{BETA} Score": round(metrics["f_beta"],    4),
            "Recall":         round(metrics["recall"],     4),
            "Precision":      round(metrics["precision"],  4),
            "ROC-AUC":        round(metrics["roc_auc"],    4),
            "Business Cost":  int(bc),
            "Threshold":      round(threshold,             4),
        })

    df_recap = (
        pd.DataFrame(recap_rows)
        .sort_values(f"F{BETA} Score", ascending=False)
        .reset_index(drop=True)
    )
    print(df_recap.to_string(index=False))
    best_model_name = df_recap.iloc[0]["Model"]
    _ok(f"Meilleur modèle (F{BETA}) : {best_model_name}")
    _step(f"Durée totale : {(time.time()-t_global)/60:.1f} min")
    _step(f"MLflow UI   : mlflow ui --backend-store-uri {MLRUNS_DIR}")

    return lgb_model, xgb_model, lr_model


if __name__ == "__main__":
    main()
