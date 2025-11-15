"""
Complete version with tests - Your implementation with fixes and tests added
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json


class FeaturePipeline:
    """Feature engineering pipeline with fit/transform pattern."""

    def __init__(self, categorical_columns: List[str], numerical_columns: List[str]):
        """Initialize the pipeline."""
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.missing_strategy: Dict[str, str] = {}
        self.is_fitted = False  # FIX: Added this

    def load_data(self, data: Union[str, Dict, List[Dict]]) -> pd.DataFrame:
        """Load data from various formats."""
        try:
            if isinstance(data, str):
                try:
                    data_dict = json.loads(data)
                except json.JSONDecodeError:
                    with open(data, 'r') as f:
                        data_dict = json.load(f)
                return pd.DataFrame(data_dict)
            elif isinstance(data, dict):
                return pd.DataFrame(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def handle_missing_values(self, data: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        df_clean = data.copy()

        # FIX: Handle empty dataframe
        if df_clean.empty:
            return df_clean

        for col, strat in strategy.items():
            if col not in df_clean.columns:
                continue

            if strat == "drop":
                df_clean = df_clean.dropna(subset=[col])
            elif col in self.numerical_columns:
                if strat == "mean":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif strat == "median":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif strat == "mode":
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
            elif col in self.categorical_columns:
                if strat == "mode":
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                    else:
                        df_clean[col] = df_clean[col].fillna("unknown")
                elif strat == "unknown":
                    df_clean[col] = df_clean[col].fillna("unknown")

        return df_clean

    def encode_categorical(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df_encoded = data.copy()
        for col in self.categorical_columns:
            if col not in df_encoded.columns:
                continue
            if fit:
                self.encoders[col] = LabelEncoder()
                df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                if col in self.encoders:
                    # Handle unseen categories
                    known_classes = set(self.encoders[col].classes_)
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if str(x) in known_classes else 'unknown'
                    )
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col].astype(str))
                else:
                    raise ValueError(f"Encoder for column {col} not found")
        return df_encoded

    def normalize_numerical(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize numerical variables."""
        df_normalized = data.copy()
        if not self.numerical_columns:
            return df_normalized

        for col in self.numerical_columns:
            if col not in df_normalized.columns:
                continue
            if fit:
                self.scalers[col] = StandardScaler()
                # FIX: Use double brackets for 2D array
                df_normalized[[col]] = self.scalers[col].fit_transform(df_normalized[[col]])
            else:
                if col in self.scalers:
                    # FIX: Use double brackets for 2D array
                    df_normalized[[col]] = self.scalers[col].transform(df_normalized[[col]])
                else:
                    raise ValueError(f"Scaler for column {col} not found")
        return df_normalized

    def fit_transform(self, data: Union[str, Dict, List[Dict]]) -> pd.DataFrame:
        """Fit the pipeline and transform the data."""
        # Load data
        df = self.load_data(data)

        # FIX: Set default strategy if not provided
        if not self.missing_strategy:
            self.missing_strategy = {
                **{col: 'mean' for col in self.numerical_columns},
                **{col: 'mode' for col in self.categorical_columns}
            }

        # Handle missing values
        df = self.handle_missing_values(df, strategy=self.missing_strategy)

        # Encode categorical variables
        df = self.encode_categorical(df, fit=True)

        # Normalize numerical variables
        df = self.normalize_numerical(df, fit=True)

        # Mark as fitted
        self.is_fitted = True

        return df

    def transform(self, data: Union[str, Dict, List[Dict]]) -> pd.DataFrame:
        """Transform new data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")

        # Load data
        df = self.load_data(data)

        # Handle missing values
        df = self.handle_missing_values(df, strategy=self.missing_strategy)

        # Encode categorical variables
        df = self.encode_categorical(df, fit=False)

        # Normalize numerical variables
        df = self.normalize_numerical(df, fit=False)

        return df


# TEST YOUR CODE
if __name__ == "__main__":
    print("="*60)
    print("FEATURE PIPELINE TESTING")
    print("="*60 + "\n")

    # Test 1: Normal case with missing values
    print("TEST 1: Normal case with missing values")
    print("-" * 40)
    sample_data = [
        {"age": 25, "income": 50000, "category": "A", "city": "NYC"},
        {"age": 30, "income": 60000, "category": "B", "city": "LA"},
        {"age": None, "income": 55000, "category": "A", "city": "NYC"},
        {"age": 35, "income": None, "category": "C", "city": "Chicago"},
        {"age": 28, "income": 52000, "category": "B", "city": "LA"},
    ]

    pipeline = FeaturePipeline(
        categorical_columns=["category", "city"],
        numerical_columns=["age", "income"]
    )

    print("Original data:")
    print(pd.DataFrame(sample_data))
    print()

    processed_data = pipeline.fit_transform(sample_data)
    print("Processed data:")
    print(processed_data)
    print()

    # Test 2: Transform new data
    print("\nTEST 2: Transform new data")
    print("-" * 40)
    new_data = [
        {"age": 27, "income": 58000, "category": "A", "city": "NYC"},
        {"age": 32, "income": None, "category": "B", "city": "LA"},
    ]

    new_processed = pipeline.transform(new_data)
    print("New data processed:")
    print(new_processed)
    print()

    # Test 3: Edge case - empty data
    print("\nTEST 3: Edge case - Empty data")
    print("-" * 40)
    empty_pipeline = FeaturePipeline(
        categorical_columns=["category"],
        numerical_columns=["age"]
    )
    try:
        empty_result = empty_pipeline.fit_transform([])
        print(f"Empty dataframe handled: shape = {empty_result.shape}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # Test 4: Edge case - all values missing
    print("\nTEST 4: Edge case - All values missing in a column")
    print("-" * 40)
    all_missing_data = [
        {"age": None, "income": 50000, "category": "A"},
        {"age": None, "income": 60000, "category": "B"},
        {"age": None, "income": 55000, "category": "A"},
    ]

    missing_pipeline = FeaturePipeline(
        categorical_columns=["category"],
        numerical_columns=["age", "income"]
    )

    try:
        result = missing_pipeline.fit_transform(all_missing_data)
        print("Result with all ages missing:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
    print()

    # Test 5: Edge case - unknown category in transform
    print("\nTEST 5: Edge case - Unknown category in transform")
    print("-" * 40)
    train_data = [
        {"age": 25, "category": "A"},
        {"age": 30, "category": "B"},
    ]

    test_data = [
        {"age": 27, "category": "A"},
        {"age": 32, "category": "C"},  # Unknown category!
    ]

    cat_pipeline = FeaturePipeline(
        categorical_columns=["category"],
        numerical_columns=["age"]
    )

    cat_pipeline.fit_transform(train_data)

    try:
        result = cat_pipeline.transform(test_data)
        print("Handled unknown category 'C':")
        print(result)
    except Exception as e:
        print(f"Error handling unknown category: {e}")
    print()

    print("\n" + "="*60)
    print("INTERVIEW DISCUSSION POINTS")
    print("="*60)
    print("""
    STRENGTHS:
    1. Good separation of concerns (load, clean, encode, normalize)
    2. Followed fit/transform pattern (sklearn style)
    3. Type hints throughout
    4. Handled unknown categories
    5. Added save/load (bonus feature)

    ISSUES FOUND:
    1. Duplicate assignment on line 56
    2. StandardScaler needs 2D array (should use df[[col]])
    3. Missing is_fitted attribute initialization
    4. Save/load won't work (can't serialize sklearn objects to JSON)
    5. Missing edge case handling (empty df, all NaN)

    TRADE-OFFS TO DISCUSS:
    1. Mean imputation vs Median (outliers?)
    2. Drop vs Impute (data loss vs noise?)
    3. LabelEncoder vs OneHotEncoder (for linear models)
    4. Per-column scalers vs single scaler (more flexibility vs memory)
    5. Fail on unknown category vs map to 'unknown' (strict vs lenient)

    PRODUCTION IMPROVEMENTS:
    1. Use joblib/pickle for save/load instead of JSON
    2. Add validation (check for duplicate columns, invalid types)
    3. Add logging for monitoring
    4. Support for custom imputation strategies
    5. Handle datetime columns
    6. Add unit tests
    """)
