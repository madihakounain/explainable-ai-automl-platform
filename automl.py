"""
ml/automl.py
------------
AutoML model selection, training, evaluation, and global feature importance.

Extracted from xai_app.py train route (lines 1232-1321).
"""

import traceback

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

from config import (
    RF_N_ESTIMATORS, HIST_GB_MAX_ITER, LOGREG_MAX_ITER,
    CV_FOLDS, CV_SAMPLE_LIMIT, RANDOM_STATE,
    GLOBAL_TOP_FEATURES, GLOBAL_CHART_TOP, GLOBAL_CMAP,
)
from utils.plotting import style_fig, fig_to_b64


# ── Model candidates ───────────────────────────────────────────────────────────

def get_candidates(task: str) -> list:
    """
    Return a list of (name, unfitted_model) tuples for the given task.
    All hyper-parameters come from config so they're easy to adjust.
    """
    if task == "classification":
        return [
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "Gradient Boosting",
                HistGradientBoostingClassifier(
                    max_iter=HIST_GB_MAX_ITER,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "Logistic Regression",
                LogisticRegression(
                    max_iter=LOGREG_MAX_ITER,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    else:
        return [
            (
                "Random Forest",
                RandomForestRegressor(
                    n_estimators=RF_N_ESTIMATORS,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "Gradient Boosting",
                HistGradientBoostingRegressor(
                    max_iter=HIST_GB_MAX_ITER,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "Ridge Regression",
                Ridge(alpha=1.0),
            ),
        ]


# ── Training loop ──────────────────────────────────────────────────────────────

def train_all(X_scaled: np.ndarray, y: np.ndarray, task: str) -> tuple:
    """
    Fit every candidate model, evaluate it, pick the best one.

    Parameters
    ----------
    X_scaled : scaled feature matrix
    y        : target array
    task     : 'classification' | 'regression'

    Returns
    -------
    results    : list[dict]  — per-model metric dicts for the UI
    best_model : fitted sklearn estimator
    best_name  : str
    """
    # Subsample for CV speed
    if len(X_scaled) > CV_SAMPLE_LIMIT:
        rng = np.random.default_rng(RANDOM_STATE)
        cv_idx = rng.choice(len(X_scaled), CV_SAMPLE_LIMIT, replace=False)
        X_cv, y_cv = X_scaled[cv_idx], y[cv_idx]
    else:
        X_cv, y_cv = X_scaled, y

    scoring   = "accuracy" if task == "classification" else "r2"
    results   = []
    best_score = -np.inf
    best_model = None
    best_name  = None

    for name, model in get_candidates(task):
        try:
            model.fit(X_scaled, y)

            cv = cross_val_score(
                model, X_cv, y_cv,
                cv=CV_FOLDS,
                n_jobs=-1,
                scoring=scoring,
            )
            cv_mean = float(np.mean(cv))

            if task == "classification":
                acc   = float(accuracy_score(y, model.predict(X_scaled)))
                entry = {"name": name, "accuracy": acc, "cv_score": cv_mean,
                         "r2": 0, "rmse": 0}
                score = cv_mean
            else:
                pred  = model.predict(X_scaled)
                r2    = float(r2_score(y, pred))
                rmse  = float(np.sqrt(mean_squared_error(y, pred)))
                entry = {"name": name, "accuracy": 0, "cv_score": cv_mean,
                         "r2": r2, "rmse": rmse}
                score = r2

            results.append(entry)

            if score > best_score:
                best_score = score
                best_model = model
                best_name  = name

        except Exception as ex:
            results.append({
                "name": name, "accuracy": 0, "cv_score": 0,
                "r2": 0, "rmse": 0, "error": str(ex),
            })

    return results, best_model, best_name


# ── Global feature importance ──────────────────────────────────────────────────

def compute_global_importance(
    best_model,
    best_name: str,
    X_scaled: np.ndarray,
    y: np.ndarray,
    feature_cols: list,
) -> tuple:
    """
    Compute global feature importance (tree built-in or permutation fallback)
    and return a sorted list of dicts plus a base64-encoded bar-chart PNG.

    Extracted from xai_app.py train route (lines 1294-1321).

    Returns
    -------
    global_importance : list[dict]  — [{feature, importance}, ...]
    global_chart_b64  : str | None  — base64 PNG
    """
    global_importance = []
    global_chart_b64  = None

    try:
        if hasattr(best_model, "feature_importances_"):
            imps = best_model.feature_importances_
        else:
            perm = permutation_importance(
                best_model, X_scaled, y,
                n_repeats=2, n_jobs=-1, random_state=RANDOM_STATE,
            )
            imps = perm.importances_mean

        sorted_idx = np.argsort(imps)[::-1]
        for i in sorted_idx[:GLOBAL_TOP_FEATURES]:
            global_importance.append({
                "feature":    feature_cols[i],
                "importance": float(imps[i]),
            })

        # Chart — horizontal bar, top N features
        style_fig()
        n_plot = min(GLOBAL_CHART_TOP, len(global_importance))
        fig, ax = plt.subplots(figsize=(7, max(3, n_plot * 0.38)))
        names  = [x["feature"]    for x in reversed(global_importance[:n_plot])]
        vals   = [x["importance"] for x in reversed(global_importance[:n_plot])]
        colors = plt.cm.get_cmap(GLOBAL_CMAP)(np.linspace(0.4, 0.9, len(names)))
        ax.barh(names, vals, color=colors, edgecolor="none", height=0.6)
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_title(f"Global Feature Importance — {best_name}", fontsize=10, pad=10)
        plt.tight_layout()
        global_chart_b64 = fig_to_b64(fig)

    except Exception:
        traceback.print_exc()

    return global_importance, global_chart_b64
