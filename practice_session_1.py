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
        pass

    # TODO: Implement the rest of the methods
    # - load_data()
    # - handle_missing_values()
    # - encode_categorical()
    # - normalize_numerical()
    # - fit_transform()
    # - transform()


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
