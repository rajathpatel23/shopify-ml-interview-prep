"""
Problem 1: Feature Engineering Pipeline

Build a feature transformation pipeline that:
1. Reads JSON data (from API or file)
2. Handles missing values
3. Applies transformations (normalization, encoding)
4. Returns processed features ready for ML model

Interview Tips:
- Think aloud: "I'm creating this to handle missing values because..."
- Ask questions: "Should we drop rows with missing values or impute them?"
- Use type hints and docstrings
- Handle edge cases: empty data, unexpected fields, wrong types
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json


class FeaturePipeline:
    """
    A pipeline for processing raw data into ML-ready features.

    This handles common preprocessing tasks like handling missing values,
    encoding categorical variables, and scaling numerical features.
    """

    def __init__(self, categorical_columns: Optional[List[str]] = None,
                 numerical_columns: Optional[List[str]] = None):
        """
        Initialize the feature pipeline.

        Args:
            categorical_columns: List of column names to encode
            numerical_columns: List of column names to normalize
        """
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False

    def load_data(self, data: Union[str, Dict, List[Dict]]) -> pd.DataFrame:
        """
        Load data from various formats.

        Args:
            data: Can be JSON string, dict, or list of dicts

        Returns:
            DataFrame with loaded data

        Raises:
            ValueError: If data format is invalid
        """
        # Interview tip: Think aloud - "I'm handling multiple input formats for flexibility"

        try:
            if isinstance(data, str):
                # Assume it's a JSON string or file path
                try:
                    # Try parsing as JSON string first
                    data_dict = json.loads(data)
                except json.JSONDecodeError:
                    # If that fails, try reading as file path
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

    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataframe.

        Args:
            df: Input dataframe
            strategy: Strategy for handling missing values
                     'mean' - fill with column mean (numerical)
                     'median' - fill with median
                     'mode' - fill with most frequent value
                     'drop' - drop rows with missing values

        Returns:
            DataFrame with missing values handled
        """
        # Interview tip: "I'm creating a copy to avoid modifying the original data"
        df_clean = df.copy()

        # Edge case: empty dataframe
        if df_clean.empty:
            return df_clean

        if strategy == 'drop':
            # Interview tip: "Dropping rows might reduce our dataset significantly"
            return df_clean.dropna()

        # Handle numerical columns
        for col in self.numerical_columns:
            if col in df_clean.columns:
                if strategy == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)

        # Handle categorical columns - use mode (most frequent)
        for col in self.categorical_columns:
            if col in df_clean.columns:
                mode_value = df_clean[col].mode()
                if len(mode_value) > 0:
                    df_clean[col].fillna(mode_value[0], inplace=True)
                else:
                    # If no mode exists, fill with 'unknown'
                    df_clean[col].fillna('unknown', inplace=True)

        return df_clean

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables to numerical values.

        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)

        Returns:
            DataFrame with encoded categorical columns
        """
        df_encoded = df.copy()

        for col in self.categorical_columns:
            if col not in df_encoded.columns:
                continue

            if fit:
                # Interview tip: "I'm fitting the encoder on training data"
                self.encoders[col] = LabelEncoder()
                df_encoded[col] = self.encoders[col].fit_transform(
                    df_encoded[col].astype(str)
                )
            else:
                # For test/inference data, use already fitted encoder
                if col in self.encoders:
                    # Handle unseen categories
                    # Interview tip: "I need to handle categories not seen during training"
                    known_classes = set(self.encoders[col].classes_)
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in known_classes else 'unknown'
                    )
                    df_encoded[col] = self.encoders[col].transform(
                        df_encoded[col].astype(str)
                    )

        return df_encoded

    def normalize_numerical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.

        Args:
            df: Input dataframe
            fit: Whether to fit scaler (True for training, False for inference)

        Returns:
            DataFrame with normalized numerical columns
        """
        df_normalized = df.copy()

        # Edge case: no numerical columns
        if not self.numerical_columns:
            return df_normalized

        # Filter to only existing columns
        cols_to_scale = [col for col in self.numerical_columns
                        if col in df_normalized.columns]

        if not cols_to_scale:
            return df_normalized

        if fit:
            # Interview tip: "Fitting scaler to learn mean and std from training data"
            df_normalized[cols_to_scale] = self.scaler.fit_transform(
                df_normalized[cols_to_scale]
            )
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call fit() first.")
            df_normalized[cols_to_scale] = self.scaler.transform(
                df_normalized[cols_to_scale]
            )

        return df_normalized

    def fit_transform(self, data: Union[str, Dict, List[Dict]],
                     missing_strategy: str = 'mean') -> pd.DataFrame:
        """
        Fit the pipeline and transform the data (use for training data).

        Args:
            data: Input data in various formats
            missing_strategy: Strategy for handling missing values

        Returns:
            Transformed DataFrame ready for ML model
        """
        # Interview tip: "Let me walk through each step of the pipeline"

        # Step 1: Load data
        df = self.load_data(data)

        # Step 2: Handle missing values
        df = self.handle_missing_values(df, strategy=missing_strategy)

        # Step 3: Encode categorical variables
        df = self.encode_categorical(df, fit=True)

        # Step 4: Normalize numerical features
        df = self.normalize_numerical(df, fit=True)

        return df

    def transform(self, data: Union[str, Dict, List[Dict]],
                 missing_strategy: str = 'mean') -> pd.DataFrame:
        """
        Transform new data using fitted pipeline (use for test/inference data).

        Args:
            data: Input data in various formats
            missing_strategy: Strategy for handling missing values

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")

        # Step 1: Load data
        df = self.load_data(data)

        # Step 2: Handle missing values
        df = self.handle_missing_values(df, strategy=missing_strategy)

        # Step 3: Encode categorical variables (using fitted encoders)
        df = self.encode_categorical(df, fit=False)

        # Step 4: Normalize numerical features (using fitted scaler)
        df = self.normalize_numerical(df, fit=False)

        return df


# Example usage - This is what you'd walk through in the interview
if __name__ == "__main__":
    # Interview tip: "Let me create some sample data to test this"

    # Sample data representing customer information
    sample_data = [
        {"age": 25, "income": 50000, "category": "A", "city": "NYC"},
        {"age": 30, "income": 60000, "category": "B", "city": "LA"},
        {"age": None, "income": 55000, "category": "A", "city": "NYC"},  # Missing age
        {"age": 35, "income": None, "category": "C", "city": "Chicago"},  # Missing income
        {"age": 28, "income": 52000, "category": "B", "city": "LA"},
    ]

    # Initialize pipeline
    # Interview tip: "I'm specifying which columns are categorical vs numerical"
    pipeline = FeaturePipeline(
        categorical_columns=["category", "city"],
        numerical_columns=["age", "income"]
    )

    # Fit and transform training data
    print("Original data:")
    print(pd.DataFrame(sample_data))
    print("\n" + "="*50 + "\n")

    # Interview tip: "I'm using fit_transform for training data"
    processed_data = pipeline.fit_transform(sample_data)
    print("Processed data:")
    print(processed_data)
    print("\n" + "="*50 + "\n")

    # Test with new data
    # Interview tip: "Now let's test with new data to ensure the pipeline works for inference"
    new_data = [
        {"age": 27, "income": 58000, "category": "A", "city": "NYC"},
        {"age": 32, "income": None, "category": "B", "city": "SF"},  # New city (edge case!)
    ]

    try:
        new_processed = pipeline.transform(new_data)
        print("New data processed:")
        print(new_processed)
    except Exception as e:
        print(f"Error: {e}")
        # Interview tip: "We'd need to handle this edge case better in production"
