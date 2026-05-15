"""
app.py
------
Flask application entry point for the XAI Platform.

Responsibilities:
  - Create the Flask app with the correct template / static folders
  - Load configuration from config.py
  - Register all route Blueprints
  - Run the development server

Run with:
    python app.py
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")          # must be set before any other matplotlib import

from flask import Flask

from config import SECRET_KEY, DEBUG, PORT, HOST

# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.secret_key = SECRET_KEY

    # ── Register Blueprints ────────────────────────────────────────────────────
    from routes.upload_routes  import upload_bp
    from routes.train_routes   import train_bp
    from routes.predict_routes import predict_bp

    app.register_blueprint(upload_bp)
    app.register_blueprint(train_bp)
    app.register_blueprint(predict_bp)

    return app


# ── Entry point ────────────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  XAI Platform — Explainable AI")
    print(f"  Open → http://{HOST}:{PORT}")
    print("═" * 55 + "\n")
    app.run(debug=DEBUG, port=PORT, host=HOST)
