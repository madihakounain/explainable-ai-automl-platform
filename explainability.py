"""
ml/explainability.py
---------------------
Local (LIME-style) and global-local (SHAP-approximation) explanations,
plus plain-English narrative generation for predictions and training results.

Original: xai_app.py predict route (lines 1418-1495).
Added:    generate_narrative(), generate_model_summary()
"""

import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance

from config import (
    LIME_N_PERTURBATIONS, LIME_NOISE_SCALE, LIME_TOP_FEATURES,
    PERM_N_REPEATS, SHAP_TOP_FEATURES, RANDOM_STATE,
    LIME_POS_COLOR, LIME_NEG_COLOR, SHAP_CMAP,
)
from utils.plotting import style_fig, fig_to_b64


# ── LIME-style local explanation ───────────────────────────────────────────────

def lime_explanation(
    model,
    task: str,
    X_row_scaled: np.ndarray,
    pred_raw,
    row_raw: dict,
    feature_cols: list,
) -> tuple:
    """
    Approximate a LIME explanation by perturbing the scaled input, predicting
    on each perturbation with the black-box model, then fitting a weighted Ridge
    regression to recover per-feature contributions.

    Parameters
    ----------
    model        : fitted sklearn estimator
    task         : 'classification' | 'regression'
    X_row_scaled : (1, n_features) scaled input
    pred_raw     : raw model prediction for the input row
    row_raw      : original user-facing values dict (for display labels)
    feature_cols : list of feature names in column order

    Returns
    -------
    local_explanation : list[dict]  — [{feature, value, contribution}, ...]
    lime_chart_b64    : str | None  — base64 PNG
    """
    local_explanation: list = []
    lime_chart_b64: str | None = None

    try:
        np.random.seed(RANDOM_STATE)
        n_features = len(feature_cols)

        # 1. Perturb around the scaled input
        noise     = np.random.randn(LIME_N_PERTURBATIONS, n_features) * LIME_NOISE_SCALE
        perturbed = X_row_scaled + noise

        # 2. Get black-box predictions on perturbations
        if task == "classification":
            try:
                perturbed_preds = model.predict_proba(perturbed)[:, int(pred_raw)]
            except Exception:
                perturbed_preds = (model.predict(perturbed) == pred_raw).astype(float)
        else:
            perturbed_preds = model.predict(perturbed)

        # 3. Fit weighted local linear model (Ridge)
        weights    = np.exp(-0.5 * np.sum(noise ** 2, axis=1))
        lime_model = Ridge(alpha=1.0)
        lime_model.fit(perturbed, perturbed_preds, sample_weight=weights)
        contributions = lime_model.coef_

        # 4. Rank by absolute contribution
        sorted_idx = np.argsort(np.abs(contributions))[::-1]
        for i in sorted_idx[:LIME_TOP_FEATURES]:
            val_disp = row_raw.get(feature_cols[i], "?")
            local_explanation.append({
                "feature":      feature_cols[i],
                "value":        str(val_disp),
                "contribution": float(contributions[i]),
            })

        # 5. Horizontal bar chart
        prediction_label = str(pred_raw)  # used only in chart title
        style_fig()
        n_plot = min(10, len(local_explanation))
        fig, ax = plt.subplots(figsize=(7, max(3, n_plot * 0.4)))
        names  = [f"{x['feature']} ({x['value']})" for x in reversed(local_explanation[:n_plot])]
        vals   = [x["contribution"]                  for x in reversed(local_explanation[:n_plot])]
        colors = [LIME_POS_COLOR if v >= 0 else LIME_NEG_COLOR for v in vals]
        ax.barh(names, vals, color=colors, edgecolor="none", height=0.6)
        ax.axvline(0, color="#64748b", linewidth=0.8)
        ax.set_xlabel("Contribution", fontsize=9)
        ax.set_title(f"LIME Explanation — Prediction: {prediction_label}", fontsize=10, pad=10)
        plt.tight_layout()
        lime_chart_b64 = fig_to_b64(fig)

    except Exception:
        traceback.print_exc()

    return local_explanation, lime_chart_b64


# ── SHAP-style approximation chart ────────────────────────────────────────────

def shap_approximation(
    model,
    X_scaled_all: np.ndarray,
    y_all: np.ndarray,
    X_df_all: pd.DataFrame,
    row: list,
    feature_cols: list,
    prediction: str,
) -> str | None:
    """
    Approximate per-feature SHAP values as:
        shap_approx[i] = global_importance[i] × |deviation_from_mean[i]|

    This captures both the model's global view of a feature and how far the
    current input deviates from the training distribution.

    Parameters
    ----------
    model        : fitted sklearn estimator
    X_scaled_all : full training matrix (scaled)
    y_all        : full training labels
    X_df_all     : full training DataFrame (unscaled, for mean/std)
    row          : unscaled numeric input row (list)
    feature_cols : feature names
    prediction   : string label shown in the chart title

    Returns
    -------
    shap_chart_b64 : str | None — base64 PNG
    """
    shap_chart_b64 = None

    try:
        # Global importance
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
        else:
            perm = permutation_importance(
                model, X_scaled_all, y_all,
                n_repeats=PERM_N_REPEATS, n_jobs=-1, random_state=RANDOM_STATE,
            )
            imps = perm.importances_mean

        # Local deviation from training mean
        means      = X_df_all.mean().values
        stds       = X_df_all.std().values + 1e-9
        deviations = np.abs((np.array(row) - means) / stds)
        shap_approx = imps * deviations

        sorted_idx = np.argsort(shap_approx)[::-1][:SHAP_TOP_FEATURES]

        style_fig()
        fig, ax = plt.subplots(figsize=(7, max(3, len(sorted_idx) * 0.4)))
        feat_names = [feature_cols[i] for i in reversed(sorted_idx)]
        feat_vals  = [shap_approx[i]  for i in reversed(sorted_idx)]
        colors = plt.cm.get_cmap(SHAP_CMAP)(np.linspace(0.2, 0.8, len(feat_names)))
        ax.barh(feat_names, feat_vals, color=colors, edgecolor="none", height=0.6)
        ax.set_xlabel("SHAP Approximation (|Impact| × Deviation)", fontsize=9)
        ax.set_title(f"SHAP-style Feature Impact — {prediction}", fontsize=10, pad=10)
        plt.tight_layout()
        shap_chart_b64 = fig_to_b64(fig)

    except Exception:
        traceback.print_exc()

    return shap_chart_b64


# ── Plain-English prediction narrative ────────────────────────────────────────

def generate_narrative(
    prediction: str,
    task: str,
    confidence: float | None,
    class_probs: dict | None,
    local_explanation: list,
    global_importance: list,
    row_raw: dict,
    target: str,
    best_model_name: str,
) -> dict:
    """
    Build a structured plain-English explanation of a single prediction.

    Returns a dict with:
      verdict     — one-sentence bottom line
      confidence_note — confidence / probability phrasing
      drivers     — bullet-point sentences for the top driving features
      neutrals    — features with negligible contribution
      global_note — which features matter most across the whole dataset
      method_note — what the model and XAI method are doing
    """
    # ── Verdict ───────────────────────────────────────────────────────────────
    if task == "classification":
        verdict = (
            f"The model predicts the outcome as <strong>{prediction}</strong> "
            f"for the given input."
        )
    else:
        verdict = (
            f"The model estimates the value of <em>{target}</em> "
            f"to be <strong>{prediction}</strong>."
        )

    # ── Confidence note ───────────────────────────────────────────────────────
    confidence_note = ""
    if confidence is not None:
        pct = round(confidence * 100, 1)
        if pct >= 90:
            level = "very high"
        elif pct >= 75:
            level = "high"
        elif pct >= 55:
            level = "moderate"
        else:
            level = "low"
        confidence_note = (
            f"The model is <strong>{level} confidence ({pct}%)</strong> in this prediction."
        )
        if class_probs and len(class_probs) > 1:
            probs_str = ", ".join(
                f"{cls}: {round(p*100,1)}%"
                for cls, p in sorted(class_probs.items(), key=lambda x: -x[1])
            )
            confidence_note += f" Class probabilities — {probs_str}."

    # ── Feature drivers ───────────────────────────────────────────────────────
    drivers  = []
    neutrals = []
    if local_explanation:
        max_contrib = max(abs(e["contribution"]) for e in local_explanation) or 1
        for entry in local_explanation[:8]:
            feat  = entry["feature"]
            val   = entry["value"]
            c     = entry["contribution"]
            ratio = abs(c) / max_contrib

            if ratio < 0.05:          # negligible — list as neutral
                neutrals.append(feat)
                continue

            direction = "pushed the prediction <strong>higher</strong>" \
                        if c > 0 else "pushed the prediction <strong>lower</strong>"

            if ratio >= 0.6:
                strength = "strongly"
            elif ratio >= 0.3:
                strength = "moderately"
            else:
                strength = "slightly"

            drivers.append(
                f"<strong>{feat}</strong> = {val} — {strength} {direction} "
                f"(contribution: {c:+.4f})."
            )

    # ── Global context ────────────────────────────────────────────────────────
    global_note = ""
    if global_importance:
        top3 = [g["feature"] for g in global_importance[:3]]
        global_note = (
            f"Across the entire training dataset, the most influential features overall "
            f"are <strong>{', '.join(top3)}</strong>."
        )

    # ── Method note ───────────────────────────────────────────────────────────
    method_note = (
        f"This prediction was made by <strong>{best_model_name}</strong>. "
        f"The local explanation uses a <strong>LIME-style perturbation approach</strong>: "
        f"the model was queried on hundreds of nearby inputs to estimate how much "
        f"each feature nudged this specific prediction. "
        f"The SHAP chart approximates feature impact as global importance weighted "
        f"by how far each input value deviates from the training average."
    )

    return dict(
        verdict=verdict,
        confidence_note=confidence_note,
        drivers=drivers,
        neutrals=neutrals,
        global_note=global_note,
        method_note=method_note,
    )


# ── Plain-English training / model summary ─────────────────────────────────────

def generate_model_summary(
    task: str,
    results: list,
    best_model_name: str,
    global_importance: list,
    target: str,
    n_rows: int,
    n_features: int,
) -> dict:
    """
    Build a plain-English summary of the AutoML training run.

    Returns a dict with:
      overview     — what was trained and on what data
      best         — why the winner won
      comparison   — brief comparison of all models
      features     — top features that drive predictions globally
      advice       — actionable next-step tip
    """
    # ── Overview ─────────────────────────────────────────────────────────────
    task_label = "classification (predicting categories)" \
                 if task == "classification" else "regression (predicting a number)"
    overview = (
        f"AutoML trained <strong>3 models</strong> on {n_rows:,} rows × "
        f"{n_features} features to solve a <strong>{task_label}</strong> "
        f"problem targeting <em>{target}</em>."
    )

    # ── Best model ────────────────────────────────────────────────────────────
    best_entry = next((r for r in results if r["name"] == best_model_name), None)
    best = ""
    if best_entry:
        if task == "classification":
            cv  = round(best_entry["cv_score"] * 100, 1)
            acc = round(best_entry["accuracy"]  * 100, 1)
            best = (
                f"<strong>{best_model_name}</strong> was selected as the best model "
                f"with a cross-validated accuracy of <strong>{cv}%</strong> "
                f"and a training accuracy of {acc}%."
            )
        else:
            r2   = round(best_entry["r2"],   4)
            rmse = round(best_entry["rmse"],  4)
            best = (
                f"<strong>{best_model_name}</strong> was selected as the best model "
                f"with R² = <strong>{r2}</strong> and RMSE = {rmse}."
            )

    # ── Model comparison ──────────────────────────────────────────────────────
    comparison = []
    for r in results:
        if task == "classification":
            score_str = f"CV accuracy {round(r['cv_score']*100,1)}%"
        else:
            score_str = f"R² {round(r['r2'],4)}"
        marker = " ✦ best" if r["name"] == best_model_name else ""
        comparison.append(f"<strong>{r['name']}</strong>: {score_str}{marker}")

    # ── Feature importance ────────────────────────────────────────────────────
    features_note = ""
    if global_importance:
        top5 = global_importance[:5]
        feat_lines = [
            f"<strong>{g['feature']}</strong> (importance {g['importance']:.4f})"
            for g in top5
        ]
        features_note = (
            f"The top predictive features are: {', '.join(feat_lines)}."
        )

    # ── Advice ────────────────────────────────────────────────────────────────
    if task == "classification":
        best_cv = best_entry["cv_score"] if best_entry else 0
        if best_cv >= 0.90:
            advice = "Excellent performance — consider checking for data leakage if accuracy seems too high."
        elif best_cv >= 0.75:
            advice = "Good performance. Try feature engineering on the top important features to push accuracy higher."
        elif best_cv >= 0.60:
            advice = "Moderate performance. Collecting more data or adding domain-specific features could help."
        else:
            advice = "Low accuracy. Review class balance, try removing noisy features, or gather more data."
    else:
        best_r2 = best_entry["r2"] if best_entry else 0
        if best_r2 >= 0.90:
            advice = "Very strong fit. Double-check for data leakage — near-perfect R² can be suspicious."
        elif best_r2 >= 0.70:
            advice = "Good fit. Experiment with polynomial features or interaction terms to improve further."
        elif best_r2 >= 0.40:
            advice = "Moderate fit. The model captures some signal but may be missing important features."
        else:
            advice = "Weak fit. Consider revisiting the target variable definition or adding more predictive features."

    return dict(
        overview=overview,
        best=best,
        comparison=comparison,
        features_note=features_note,
        advice=advice,
    )
