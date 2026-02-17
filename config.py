"""
config.py
---------
Central configuration and constants for the XAI Platform.
All tunable values live here so every other module imports
from this single source of truth.
"""

import os

# ── Flask ──────────────────────────────────────────────────────────────────────
SECRET_KEY = os.urandom(24)
DEBUG      = True
PORT       = 5000
HOST       = "127.0.0.1"

# ── In-memory session store ────────────────────────────────────────────────────
# Shared dict keyed by session-id (sid).
# All modules import this object directly — never reassign it.
STORE: dict = {}

# ── Data analysis ──────────────────────────────────────────────────────────────
PREVIEW_ROWS        = 8      # rows shown in the data preview tab
MAX_DIST_CHARTS     = 6      # maximum histogram charts rendered on upload
MAX_MISSING_DROP    = 0.80   # drop column if more than this fraction is NaN

# ── AutoML training ────────────────────────────────────────────────────────────
CV_FOLDS            = 3      # k-fold cross-validation folds
CV_SAMPLE_LIMIT     = 2000   # max rows used for CV (subsampled for speed)
RF_N_ESTIMATORS     = 50     # Random Forest tree count
HIST_GB_MAX_ITER    = 100    # HistGradientBoosting iterations
LOGREG_MAX_ITER     = 300    # Logistic Regression max iterations
RANDOM_STATE        = 42

# Regression target: if numeric AND more than this many unique values → regression
REGRESSION_UNIQUE_THRESHOLD = 10

# ── XAI / Explainability ───────────────────────────────────────────────────────
LIME_N_PERTURBATIONS = 100   # perturbation samples for LIME approximation
LIME_NOISE_SCALE     = 0.2   # Gaussian noise std for perturbations
LIME_TOP_FEATURES    = 12    # features shown in LIME explanation
PERM_N_REPEATS       = 2     # permutation-importance repeat count
GLOBAL_TOP_FEATURES  = 15    # features shown in global importance list
GLOBAL_CHART_TOP     = 12    # features shown in global importance chart
SHAP_TOP_FEATURES    = 12    # features shown in SHAP-style chart

# ── Plot styling (dark theme) ──────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.facecolor": "#111318",
    "axes.facecolor":   "#181c24",
    "axes.edgecolor":   "#252b3b",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#e2e8f0",
    "grid.color":       "#252b3b",
    "grid.linewidth":   0.6,
    "axes.grid":        True,
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right": False,
}
PLOT_DPI          = 130
PLOT_BG_COLOR     = "#111318"
DIST_CHART_COLOR  = "#6ee7b7"
LIME_POS_COLOR    = "#6ee7b7"
LIME_NEG_COLOR    = "#f87171"
GLOBAL_CMAP       = "GnBu"
SHAP_CMAP         = "plasma"
