# Helsinki Housing Price Predictor (Production-First ML Pipeline)

An end-to-end machine learning system built to predict residential apartment prices in the Greater Helsinki area. This project demonstrates how enterprise software engineering principles (clean code, pipeline modularity, Pydantic validation, and REST API deployment) integrate with machine learning.

## Project Architecture
- **Data Pipeline:** Custom synthetic/statistical data generator mirroring Helsinki district valuation metrics (Kamppi, Kallio, Töölö, etc.).
- **ML Core:** Scikit-Learn Pipeline (`ColumnTransformer` + `RandomForestRegressor`) achieving ~0.94 R² score.
- **API Service:** FastAPI backend with strict Pydantic input validation and real-time inference.

---

## Project Structure

```text
helsinki-housing-ml/
├── data/                  # Raw and processed datasets
├── notebooks/             # Exploratory Data Analysis (EDA)
├── src/
│   ├── config.py          # Configuration and paths
│   ├── generate_data.py   # Dataset generation script
│   ├── features.py        # Feature engineering pipeline
│   ├── train.py           # Model training and evaluation
│   └── evaluate.py        # Metrics calculations
├── api/
│   └── main.py            # FastAPI application and endpoints
├── tests/                 # Automated unit tests (pytest)
├── models/                # Serialized model artifact (.joblib)
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies