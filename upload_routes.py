"""
routes/upload_routes.py
-----------------------
Blueprint for the index page and CSV upload + analysis endpoint.

Extracted from xai_app.py:
  - GET  /          (index route,  line 1096)
  - POST /upload    (upload route, line 1102)
"""

import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Blueprint, request, jsonify, session, render_template

from config import STORE, PREVIEW_ROWS, MAX_DIST_CHARTS
from utils.plotting import style_fig, fig_to_b64

upload_bp = Blueprint("upload", __name__)


# ── GET / ──────────────────────────────────────────────────────────────────────

@upload_bp.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("index.html")


# ── POST /upload ───────────────────────────────────────────────────────────────

@upload_bp.route("/upload", methods=["POST"])
def upload():
    try:
        sid = session.setdefault("sid", str(uuid.uuid4()))
        f   = request.files.get("file")
        if not f:
            return jsonify(error="No file received"), 400

        df = pd.read_csv(f)
        STORE[sid] = {"df": df}

        rows, cols       = df.shape
        dup              = int(df.duplicated().sum())
        total_missing    = int(df.isnull().sum().sum())
        numeric_cols     = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        # Per-column metadata for the Columns tab
        col_info = []
        for c in df.columns:
            s         = df[c]
            missing   = int(s.isnull().sum())
            miss_pct  = round(missing / rows * 100, 1)
            dtype     = str(s.dtype)
            unique    = int(s.nunique())
            kind      = "numeric" if c in numeric_cols else "categorical"
            samples   = [str(x) for x in s.dropna().unique()[:5].tolist()]
            col_info.append(dict(
                name=c, dtype=dtype, kind=kind,
                missing=missing, miss_pct=miss_pct,
                unique=unique, samples=samples,
            ))

        # Data preview
        prev         = df.head(PREVIEW_ROWS).fillna("").astype(str)
        preview_rows = prev.to_dict(orient="records")
        preview_cols = df.columns.tolist()

        # Distribution histograms (numeric columns only)
        style_fig()
        dist_charts = []
        for col in numeric_cols[:MAX_DIST_CHARTS]:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.hist(df[col].dropna(), bins=30, color="#6ee7b7",
                    alpha=0.85, edgecolor="none")
            ax.set_title(col, fontsize=9, pad=6)
            ax.set_ylabel("count", fontsize=8)
            dist_charts.append({"col": col, "img": fig_to_b64(fig)})

        return jsonify(
            rows=rows, cols=cols,
            duplicates=dup, total_missing=total_missing,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols,
            columns=col_info,
            preview_rows=preview_rows, preview_cols=preview_cols,
            dist_charts=dist_charts,
        )

    except Exception as e:
        return jsonify(error=str(e)), 500
