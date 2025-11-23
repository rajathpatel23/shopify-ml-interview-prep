"""
Problem 3: Data Processing Pipeline

Build a data processing pipeline for analyzing user activity logs.

Shopify Context:
- Process millions of log entries
- Aggregate metrics (daily/weekly/monthly)
- Detect anomalies
- Export results

Interview Focus:
- Pandas operations
- Efficient processing
- Error handling
- Memory management
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json


class LogProcessor:
    """
    Process and analyze user activity logs.

    Interview Tip: "I'm building this to handle large-scale log analysis.
    The key trade-offs are memory usage vs processing speed."
    """

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None

    def load_logs(self, log_file: str, format: str = 'csv') -> pd.DataFrame:
        """
        Load logs from file.

        Args:
            log_file: Path to log file
            format: File format ('csv', 'json', 'parquet')

        Returns:
            DataFrame with logs

        Interview Tip: "Supporting multiple formats makes this more flexible.
        Parquet is faster for large files but CSV is more universal."
        """
        if format == 'csv':
            df = pd.read_csv(log_file)
        elif format == 'json':
            df = pd.read_json(log_file, lines=True)  # JSONL format
        elif format == 'parquet':
            df = pd.read_parquet(log_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Validate required columns
        required_cols = ['timestamp', 'user_id', 'action']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp (important for time-series analysis)
        df = df.sort_values('timestamp')

        self.data = df
        return df

    def aggregate_daily(self, metric: str = 'count') -> pd.DataFrame:
        """
        Aggregate logs by day.

        Args:
            metric: Aggregation metric ('count', 'unique_users', 'actions')

        Returns:
            DataFrame with daily aggregates

        Interview Tip: "I'm using pandas groupby which is optimized for
        aggregation. The trade-off is memory usage vs clarity."
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_logs() first.")

        # Create date column
        df = self.data.copy()
        df['date'] = df['timestamp'].dt.date

        if metric == 'count':
            # Total events per day
            result = df.groupby('date').size().reset_index(name='count')

        elif metric == 'unique_users':
            # Unique users per day
            result = df.groupby('date')['user_id'].nunique().reset_index(name='unique_users')

        elif metric == 'actions':
            # Count by action type per day
            result = df.groupby(['date', 'action']).size().reset_index(name='count')
            # Pivot to wide format
            result = result.pivot(index='date', columns='action', values='count').fillna(0)
            result = result.reset_index()

        else:
            raise ValueError(f"Unknown metric: {metric}")

        return result

    def detect_anomalies(self, column: str = 'count',
                        method: str = 'zscore',
                        threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies in time series data.

        Args:
            column: Column to analyze
            method: Detection method ('zscore', 'iqr', 'moving_avg')
            threshold: Threshold for anomaly detection

        Returns:
            DataFrame with anomaly flags

        Interview Tip: "I'm using z-score for simplicity. For production,
        I'd consider more sophisticated methods like Isolation Forest or
        seasonal decomposition."
        """
        if self.data is None:
            raise ValueError("No data loaded")

        # Aggregate by day first
        daily = self.aggregate_daily('count')

        if method == 'zscore':
            # Z-score method: (x - mean) / std
            mean = daily[column].mean()
            std = daily[column].std()

            if std == 0:
                # No variation, no anomalies
                daily['is_anomaly'] = False
                daily['anomaly_score'] = 0.0
            else:
                daily['anomaly_score'] = (daily[column] - mean) / std
                daily['is_anomaly'] = np.abs(daily['anomaly_score']) > threshold

        elif method == 'iqr':
            # IQR method: values outside 1.5 * IQR
            q1 = daily[column].quantile(0.25)
            q3 = daily[column].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            daily['is_anomaly'] = (daily[column] < lower_bound) | (daily[column] > upper_bound)
            daily['anomaly_score'] = np.where(
                daily[column] < lower_bound,
                (lower_bound - daily[column]) / iqr,
                np.where(
                    daily[column] > upper_bound,
                    (daily[column] - upper_bound) / iqr,
                    0
                )
            )

        elif method == 'moving_avg':
            # Moving average method
            window = 7  # 7-day moving average
            daily['moving_avg'] = daily[column].rolling(window=window, center=True).mean()
            daily['moving_std'] = daily[column].rolling(window=window, center=True).std()

            # Anomaly if outside threshold * std from moving average
            daily['anomaly_score'] = np.abs(
                (daily[column] - daily['moving_avg']) / daily['moving_std']
            )
            daily['is_anomaly'] = daily['anomaly_score'] > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        return daily

    def user_cohort_analysis(self, cohort_period: str = 'M') -> pd.DataFrame:
        """
        Perform cohort analysis (users grouped by signup period).

        Args:
            cohort_period: Cohort period ('D', 'W', 'M' for day, week, month)

        Returns:
            DataFrame with cohort retention data

        Interview Tip: "Cohort analysis is crucial for e-commerce to understand
        user retention. This helps answer questions like 'Do users from Black
        Friday return?'"
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data.copy()

        # Get first activity for each user (cohort assignment)
        first_activity = df.groupby('user_id')['timestamp'].min().reset_index()
        first_activity.columns = ['user_id', 'cohort_date']

        # Assign cohort period
        first_activity['cohort'] = first_activity['cohort_date'].dt.to_period(cohort_period)

        # Merge back to main data
        df = df.merge(first_activity[['user_id', 'cohort']], on='user_id')

        # Calculate period index (periods since cohort start)
        df['activity_period'] = df['timestamp'].dt.to_period(cohort_period)
        df['period_index'] = (df['activity_period'] - df['cohort']).apply(lambda x: x.n)

        # Cohort analysis: count unique users per cohort per period
        cohort_data = df.groupby(['cohort', 'period_index'])['user_id'].nunique().reset_index()
        cohort_data.columns = ['cohort', 'period_index', 'users']

        # Pivot to wide format
        cohort_matrix = cohort_data.pivot(
            index='cohort',
            columns='period_index',
            values='users'
        )

        # Calculate retention percentage
        cohort_sizes = cohort_matrix[0]  # Period 0 = cohort size
        retention = cohort_matrix.div(cohort_sizes, axis=0) * 100

        return retention

    def export_results(self, df: pd.DataFrame, output_file: str,
                      format: str = 'csv') -> None:
        """
        Export processed results.

        Args:
            df: DataFrame to export
            output_file: Output file path
            format: Output format ('csv', 'json', 'parquet')

        Interview Tip: "Different formats for different use cases:
        CSV for human readability, Parquet for efficiency, JSON for APIs."
        """
        if format == 'csv':
            df.to_csv(output_file, index=False)
        elif format == 'json':
            df.to_json(output_file, orient='records', lines=True)
        elif format == 'parquet':
            df.to_parquet(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Exported {len(df)} rows to {output_file}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("DATA PROCESSING PIPELINE DEMO")
    print("="*60 + "\n")

    # Create sample data
    print("Creating sample log data...")
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='H')
    sample_logs = []

    np.random.seed(42)
    for date in dates:
        # Normal traffic: 100-200 events per hour
        num_events = np.random.randint(100, 200)

        # Add anomaly on specific days
        if date.day == 15:  # Spike on 15th
            num_events *= 3

        for _ in range(num_events):
            sample_logs.append({
                'timestamp': date,
                'user_id': f"user_{np.random.randint(1, 1000)}",
                'action': np.random.choice(['view', 'click', 'purchase'], p=[0.7, 0.25, 0.05])
            })

    # Save to CSV
    pd.DataFrame(sample_logs).to_csv('/tmp/sample_logs.csv', index=False)
    print(f"Created {len(sample_logs)} log entries\n")

    # Process logs
    processor = LogProcessor()

    # Load
    print("1. Loading logs...")
    df = processor.load_logs('/tmp/sample_logs.csv', format='csv')
    print(f"   Loaded {len(df)} entries")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")

    # Aggregate daily
    print("2. Daily aggregation...")
    daily = processor.aggregate_daily('count')
    print(daily.head())
    print()

    # Detect anomalies
    print("3. Anomaly detection...")
    anomalies = processor.detect_anomalies(threshold=2.0)
    anomaly_days = anomalies[anomalies['is_anomaly']]
    print(f"   Found {len(anomaly_days)} anomalous days:")
    print(anomaly_days[['date', 'count', 'anomaly_score']])
    print()

    # Cohort analysis
    print("4. Cohort analysis...")
    cohorts = processor.user_cohort_analysis('M')
    print("   Retention by cohort (%):")
    print(cohorts.head())
    print()

    print("="*60)
    print("INTERVIEW DISCUSSION POINTS")
    print("="*60)
    print("""
    1. SCALABILITY:
       - For millions of rows, use chunking: pd.read_csv(chunksize=10000)
       - Use Dask for distributed processing
       - Use Parquet for faster I/O

    2. MEMORY OPTIMIZATION:
       - Use categorical dtype for repeated strings
       - Process in chunks instead of loading all at once
       - Use generators for streaming processing

    3. ANOMALY DETECTION:
       - Z-score: Simple but assumes normal distribution
       - IQR: More robust to outliers
       - Moving average: Captures trends
       - Production: Use Isolation Forest, Prophet, or LSTM

    4. COHORT ANALYSIS:
       - Critical for understanding user retention
       - Helps measure product-market fit
       - Can segment by acquisition channel, geography, etc.

    5. PRODUCTION CONSIDERATIONS:
       - Add logging for monitoring
       - Handle missing data gracefully
       - Validate data quality
       - Add unit tests
       - Use type hints
       - Document assumptions
    """)
