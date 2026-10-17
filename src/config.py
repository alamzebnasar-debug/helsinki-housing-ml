from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "raw" / "helsinki_housing.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "helsinki_housing_processed.csv"
MODEL_SAVE_PATH = MODELS_DIR / "housing_model.joblib"

# Target Variable
TARGET_COL = "price_eur"

# Feature Schema Definitions
NUMERICAL_FEATURES = [
    "size_sqm",
    "rooms",
    "construction_year",
    "floor",
    "monthly_maintenance_fee_eur",
]

CATEGORICAL_FEATURES = [
    "district",          # e.g., 'Kamppi', 'Kallio', 'Pasila', 'Tapiola'
    "condition",         # e.g., 'Good', 'Satisfactory', 'New'
]

BOOLEAN_FEATURES = [
    "has_sauna",
    "has_elevator",
    "has_balcony",
]

# All features used by the model pipeline
FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES