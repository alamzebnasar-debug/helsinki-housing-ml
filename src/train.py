import joblib
import pandas as pd
from config import FEATURE_COLS, MODEL_SAVE_PATH, RAW_DATA_PATH, TARGET_COL
from features import build_preprocessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_model():
    # 1. Load Data
    print(f"Loading data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Assemble Full Machine Learning Pipeline
    preprocessor = build_preprocessor()

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100, random_state=42, max_depth=12
                ),
            ),
        ]
    )

    # 4. Train the Pipeline
    print("Training Random Forest Housing Model...")
    model_pipeline.fit(X_train, y_train)

    # 5. Evaluate Performance
    y_pred = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5  # Standard RMSE calculation across all versions
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): €{mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): €{rmse:,.2f}")

    # 6. Save Serialized Model Artifact
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_SAVE_PATH)
    print(f"\n✅ Model pipeline successfully saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()