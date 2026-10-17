from config import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor() -> ColumnTransformer:
    """Creates a Scikit-Learn ColumnTransformer for preprocessing input features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "bool",
                "passthrough",  # Boolean values (True/False or 1/0) pass as-is
                BOOLEAN_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor