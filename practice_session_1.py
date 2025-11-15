"""
Practice Session 1: Feature Engineering Pipeline

YOUR TASK:
Build a FeaturePipeline class from scratch.

Requirements:
1. Load data from JSON or dict
2. Handle missing values:
   - Numerical: fill with mean
   - Categorical: fill with mode
3. Encode categorical variables (LabelEncoder)
4. Normalize numerical features (StandardScaler)
5. Follow fit/transform pattern (like sklearn)

Time Limit: 45 minutes

PRACTICE INTERVIEW SKILLS:
- Think out loud the entire time
- Explain your design decisions
- Mention trade-offs
- Use Cursor/AI and explain what it suggests
- Handle edge cases
- Test as you go

Start coding below!
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

# START YOUR IMPLEMENTATION HERE

class FeaturePipeline:
    """
    Your implementation of a feature engineering pipeline.

    As you code, think aloud:
    - "I'm creating this to handle..."
    - "The trade-off here is..."
    - "An edge case I need to consider is..."
    """

    def __init__(self, categorical_columns: List[str], numerical_columns: List[str]):
        """
        Initialize the pipeline.

        THINK ALOUD: "I'm storing the column names so we know which
        columns to encode vs normalize..."
        """
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.numerical_columns = numerical_columns
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.missing_strategy: Dict[str, str] = {}

    def load_data(self, data: Union[str, Dict, List[Dict]]) -> pd.DataFrame:
        """
        Load data from a list of dictionaries.
        """
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
        """
        handle missing values in the data based on the strategy.
        strategy is a dictionary with the column names as keys and the strategy as values.
        the strategy can be "drop", "mean", "median", "mode", "unknown".

        Args:
            data (pd.DataFrame): the data to handle missing values
            strategy (Dict[str, str]): the strategy to use for each column

        Returns:
            pd.DataFrame: the data with missing values handled
        """
        df_clean = data.copy()
        for col, strategy in strategy.items():
            if strategy == "drop":
                df_clean = df_clean.dropna(subset=[col])
            if col in self.numerical_columns:
                if strategy == "mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategy == "median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif strategy == "mode":
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            if col in self.categorical_columns:
                if strategy == "mode":
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                elif strategy == "unknown":
                    df_clean[col].fillna("unknown", inplace=True)

        return df_clean



    def encode_categorical(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables in the data.
        """
        df_encoded = data.copy()
        for col in self.categorical_columns:
            if col not in df_encoded.columns:
                continue
            if fit:
                self.encoders[col] = LabelEncoder()
                df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                if col in self.encoders:
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in self.encoders[col].classes_ else 'unknown'
                    )
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col].astype(str))
                else:
                    raise ValueError(f"Encoder for column {col} not found")
        return df_encoded

    def normalize_numerical(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical variables in the data.
        """
        df_normalized = data.copy()
        if not self.numerical_columns:
            return df_normalized
        for col in self.numerical_columns:
            if col not in df_normalized.columns:
                continue
            if fit:
                self.scalers[col] = StandardScaler()
                df_normalized[col] = self.scalers[col].fit_transform(df_normalized[col])
            else:
                if col in self.scalers:
                    df_normalized[col] = self.scalers[col].transform(df_normalized[col])
                else:
                    raise ValueError(f"Scaler for column {col} not found")
        return df_normalized


    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline and transform the data.
        Args:
            data (pd.DataFrame): the data to fit the pipeline

        Returns:
            pd.DataFrame: the data with the pipeline fitted
        """
        # load data
        df = self.load_data(data)

        # handle missing values
        df = self.handle_missing_values(df, strategy=self.missing_strategy)

        # encode categorical variables
        df = self.encode_categorical(df, fit=True)

        # normalize numerical variables
        df = self.normalize_numerical(df, fit=True)

        return df

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted pipeline. This is used for test/inference data.

        Args:
            data (pd.DataFrame): the data to transform

        Returns:
            pd.DataFrame: the data with the pipeline transformed
        """
        # load data
        df = self.load_data(data)

        # handle missing values
        df = self.handle_missing_values(df, strategy=self.missing_strategy)

        # encode categorical variables
        df = self.encode_categorical(df, fit=False)
        
        # normalize numerical variables
        df = self.normalize_numerical(df, fit=False)

        return df

    def save_pipeline(self, save_path: str) -> None:
        """
        Save the pipeline to a file, maybe in .json format
        """
        with open(save_path, 'w') as f:
            json.dump({
                'categorical_columns': self.categorical_columns,
                'numerical_columns': self.numerical_columns,
                'encoders': self.encoders,
                'scalers': self.scalers,
                'is_fitted': self.is_fitted,
                'missing_strategy': self.missing_strategy
            }, f, indent=4)

    def load_pipeline(self, load_path: str) -> None:
        """
        Load the pipeline from a file, maybe in .json format
        """
        with open(load_path, 'r') as f:
            pipeline_dict = json.load(f)
        self.categorical_columns = pipeline_dict['categorical_columns']
        self.numerical_columns = pipeline_dict['numerical_columns']
        self.encoders = pipeline_dict['encoders']
        self.scalers = pipeline_dict['scalers']
        self.is_fitted = pipeline_dict['is_fitted']
        self.missing_strategy = pipeline_dict['missing_strategy']
        return None

# TEST YOUR CODE
if __name__ == "__main__":
    print("Testing Feature Pipeline")
    print("=" * 60)

    # Test data with missing values
    sample_data = [
        {"age": 25, "income": 50000, "category": "A", "city": "NYC"},
        {"age": 30, "income": 60000, "category": "B", "city": "LA"},
        {"age": None, "income": 55000, "category": "A", "city": "NYC"},
        {"age": 35, "income": None, "category": "C", "city": "Chicago"},
        {"age": 28, "income": 52000, "category": "B", "city": "LA"},
    ]

    # THINK ALOUD: "Let me test with this data that has missing values
    # in both age and income to ensure my handling works..."

    # TODO: Create your pipeline and test it
    # pipeline = FeaturePipeline(...)
    # processed_data = pipeline.fit_transform(sample_data)
    # print(processed_data)

    print("\nTest your edge cases:")
    print("1. Empty data")
    print("2. All values missing in a column")
    print("3. Unknown category in transform")

    # TODO: Add edge case tests

"""
AFTER YOU'RE DONE:
1. Did you explain your thinking out loud?
2. Did you handle edge cases?
3. Did you use type hints?
4. Did you test your code?
5. Compare with problem_1_feature_pipeline/feature_pipeline.py

REFLECTION QUESTIONS:
- What would you do differently?
- What trade-offs did you make?
- What would you add for production?
- How did using Cursor/AI help?
"""
