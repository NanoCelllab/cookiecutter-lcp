import importlib

packages = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "plotly",
    "pyarrow",
    "sqlalchemy",
    "statsmodels",
    "umap",
    "lightgbm",
    "shap",
    "pycytominer",
    "copairs",
    "marimo",
    "hdbscan",
    "harmonypy",
    "tifffile",
    "PIL",
    "joblib",
]

for package in packages:
    importlib.import_module(package)

print("Environment check passed.")
