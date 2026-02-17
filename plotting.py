"""
utils/plotting.py
-----------------
Shared matplotlib utilities used by every module that produces charts.

Moved from xai_app.py:
  - fig_to_b64()  (line 1069)
  - style_fig()   (line 1078)
"""

import io
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PLOT_STYLE, PLOT_DPI, PLOT_BG_COLOR


def style_fig() -> None:
    """Apply the dark-theme rcParams to every subsequent matplotlib figure."""
    plt.rcParams.update(PLOT_STYLE)


def fig_to_b64(fig) -> str:
    """
    Render a matplotlib Figure to a PNG and return it as a base64 string
    suitable for embedding in a data-URI inside HTML/JSON.

    The figure is closed after encoding to free memory.
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
        facecolor=PLOT_BG_COLOR,
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
