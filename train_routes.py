"""
routes/train_routes.py
----------------------
Blueprint for the AutoML training endpoint.

Delegates to:
  - ml.preprocessing   (feature + target preparation)
  - ml.automl          (model training + global importance)
  - ml.explainability  (generate_model_summary)
"""

import traceback
import uuid

from flask import Blueprint, request, jsonify, session

from config import STORE
from ml.preprocessing import detect_task, preprocess_features, encode_target
from ml.automl import train_all, compute_global_importance
from ml.explainability import generate_model_summary

train_bp = Blueprint("train", __name__)


@train_bp.route("/train", methods=["POST"])
def train():
    try:
        sid   = session.get("sid")
        store = STORE.get(sid)
        if not store or "df" not in store:
            return jsonify(error="No dataset found. Upload first."), 400

        body   = request.get_json()
        target = body.get("target")
        df     = store["df"].copy()

        if target not in df.columns:
            return jsonify(error=f"Target column '{target}' not found."), 400

        # ── Determine task ────────────────────────────────────────────────────
        y_raw = df[target]
        task  = detect_task(y_raw)

        # ── Preprocess features ───────────────────────────────────────────────
        X_raw = df[[c for c in df.columns if c != target]]
        X_scaled, feature_cols, encoders, feature_info, scaler, X_clean = \
            preprocess_features(X_raw)

        # ── Encode target ─────────────────────────────────────────────────────
        y, target_encoder, classes = encode_target(y_raw, task)

        # ── Train all candidates, pick best ───────────────────────────────────
        results, best_model, best_name = train_all(X_scaled, y, task)

        # ── Global feature importance + chart ─────────────────────────────────
        global_importance, global_chart_b64 = compute_global_importance(
            best_model, best_name, X_scaled, y, feature_cols,
        )

        # ── Plain-English training summary ─────────────────────────────────────
        model_summary = generate_model_summary(
            task=task,
            results=results,
            best_model_name=best_name,
            global_importance=global_importance,
            target=target,
            n_rows=len(df),
            n_features=len(feature_cols),
        )

        # ── Persist everything needed for prediction ──────────────────────────
        store.update({
            "model":            best_model,
            "task":             task,
            "encoders":         encoders,
            "scaler":           scaler,
            "feature_cols":     feature_cols,
            "feature_info":     feature_info,
            "target":           target,
            "target_encoder":   target_encoder,
            "classes":          classes,
            "X_scaled":         X_scaled,
            "y":                y,
            "X_df":             X_clean,
            "best_model_name":  best_name,
            "global_importance": global_importance,
        })

        return jsonify(
            task=task,
            models=results,
            best_model=best_name,
            features=feature_cols,
            feature_info=feature_info,
            global_importance=global_importance,
            global_chart=global_chart_b64,
            model_summary=model_summary,
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500
