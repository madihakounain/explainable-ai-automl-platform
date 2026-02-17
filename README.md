The Explainable AI AutoML Platform is a modular Flask-based web application that enables users to upload datasets, automatically train machine learning models, generate predictions, and understand model behavior through interactive Explainable AI visualizations.

The platform transforms traditional machine learning workflows into a complete end-to-end AI system by integrating automated preprocessing, model benchmarking, prediction services, and interpretable AI insights within a scalable project architecture.

🚀 Features

Upload CSV datasets and perform automatic dataset analysis

Intelligent detection of Classification or Regression problems

Automated preprocessing pipeline:

Missing value handling

Feature encoding

Feature scaling

AutoML training with multiple ML algorithms

Cross-validation based model comparison

Automatic best model selection

Dynamic prediction interface generated from dataset features

Prediction confidence scores and class probabilities

Explainable AI capabilities:

Global feature importance analysis

LIME-style local explanations

SHAP-style feature impact approximations

Interactive web dashboard for data exploration and visualization

🏗️ Project Architecture

The application follows a modular Flask architecture using Blueprints and separated ML pipelines for maintainability and scalability.

xai_platform/
├── app.py                      # Flask factory & Blueprint registration
├── config.py                   # Global configuration and constants
│
├── routes/                     # API and application routes
│   ├── upload_routes.py        # Dataset upload & homepage
│   ├── train_routes.py         # AutoML training endpoint
│   └── predict_routes.py       # Prediction & explanation endpoint
│
├── ml/                         # Machine Learning pipeline
│   ├── preprocessing.py        # Data preprocessing & encoding
│   ├── automl.py               # Model training & evaluation
│   └── explainability.py       # LIME & SHAP-style explanations
│
├── utils/
│   └── plotting.py             # Visualization utilities
│
├── templates/
│   └── index.html              # Jinja2 frontend template
│
└── static/
    ├── css/styles.css
    └── js/app.js

⚙️ Tech Stack

Backend: Python, Flask (Blueprint Architecture)

Machine Learning: Scikit-learn, Pandas, NumPy

Explainable AI: Custom LIME-style and SHAP-style implementations

Visualization: Matplotlib, Seaborn

Frontend: HTML, CSS, JavaScript (Jinja2 templates)

🎯 Key Highlights

End-to-end machine learning lifecycle implementation

Modular production-style architecture

Automated model benchmarking and selection

Interpretable AI predictions for transparency

Full-stack interactive AI application

▶️ Running the Project
pip install -r requirements.txt
python app.py


Open in browser:

http://127.0.0.1:5000

💡 Project Goal

This project focuses on making machine learning systems transparent and accessible by allowing users to not only generate predictions but also understand why those predictions occur through explainable AI techniques.

👩‍💻 Author

Built as an applied Machine Learning and Explainable AI engineering project demonstrating real-world AI system design and modular software architecture.
