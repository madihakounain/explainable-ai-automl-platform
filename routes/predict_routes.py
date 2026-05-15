"""
routes/predict_routes.py
------------------------
Blueprint for the prediction + XAI explanation endpoint.

Delegates to:
  - ml.preprocessing    (encode_input_row)
  - ml.explainability   (lime_explanation, shap_approximation, generate_narrative)
"""

import traceback

from flask import Blueprint, request, jsonify, session

from config import STORE
from ml.preprocessing import encode_input_row
from ml.explainability import lime_explanation, shap_approximation, generate_narrative

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        sid   = session.get("sid")
        store = STORE.get(sid)
        if not store or "model" not in store:
            return jsonify(error="No trained model. Train first."), 400

        body         = request.get_json()
        row_raw      = body.get("row", {})

        # Retrieve everything saved during training
        model            = store["model"]
        task             = store["task"]
        encoders         = store["encoders"]
        scaler           = store["scaler"]
        feature_cols     = store["feature_cols"]
        feature_info     = store["feature_info"]
        target_enc       = store.get("target_encoder")
        classes          = store.get("classes")
        X_scaled_all     = store.get("X_scaled")
        y_all            = store.get("y")
        X_df_all         = store.get("X_df")
        target           = store.get("target", "target")
        best_model_name  = store.get("best_model_name", type(model).__name__)
        global_importance = store.get("global_importance", [])

        # ── Encode the input row ───────────────────────────────────────────────
        X_row_scaled, row = encode_input_row(
            row_raw, feature_cols, feature_info, encoders, scaler,
        )

        # ── Predict ───────────────────────────────────────────────────────────
        pred_raw    = model.predict(X_row_scaled)[0]
        confidence  = None
        class_probs = None

        if task == "classification":
            if target_enc is not None:
                prediction = str(target_enc.inverse_transform([int(pred_raw)])[0])
            else:
                prediction = str(pred_raw)

            if hasattr(model, "predict_proba"):
                proba      = model.predict_proba(X_row_scaled)[0]
                confidence = float(max(proba))
                if classes:
                    class_probs = {
                        str(classes[i]): float(p) for i, p in enumerate(proba)
                    }
        else:
            prediction = f"{float(pred_raw):.4f}"

        # ── LIME local explanation ─────────────────────────────────────────────
        local_explanation, lime_chart_b64 = lime_explanation(
            model, task, X_row_scaled, pred_raw, row_raw, feature_cols,
        )

        # ── SHAP-style chart ───────────────────────────────────────────────────
        shap_chart_b64 = shap_approximation(
            model, X_scaled_all, y_all, X_df_all,
            row, feature_cols, prediction,
        )

        # ── Descriptive narrative ──────────────────────────────────────────────
        narrative = generate_narrative(
            prediction=prediction,
            task=task,
            confidence=confidence,
            class_probs=class_probs,
            local_explanation=local_explanation,
            global_importance=global_importance,
            row_raw=row_raw,
            target=target,
            best_model_name=best_model_name,
        )

        return jsonify(
            prediction=prediction,
            confidence=confidence,
            class_probs=class_probs,
            local_explanation=local_explanation,
            lime_chart=lime_chart_b64,
            shap_chart=shap_chart_b64,
            narrative=narrative,
            shap_ok=shap_chart_b64 is not None,
            lime_ok=lime_chart_b64 is not None,
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500
