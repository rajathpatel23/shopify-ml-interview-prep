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

from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import tempfile
import unittest
from datetime import datetime, date

class LogProcessor:
    """ Process and analyze user activity logs.
    
    Interview Tip: "I'm building this to handle large-scale log analysis.
    The key trade-offs are memory usage vs processing speed."
    """

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None

    def load_logs(self, log_file: str, format: str = 'csv') -> pd.DataFrame:
        """ Load logs from file.
        Args:
            log_file: Path to log file
            format: File format ('csv', 'json', 'parquet')

        Returns:
            pd.DataFrame: DataFrame containing the logs
        """
        try:
            if format == 'csv':
                df = pd.read_csv(log_file)
            elif format == 'json':
                df = pd.read_json(log_file)
            elif format == 'parquet':
                df = pd.read_parquet(log_file)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logging.error(f"Error loading logs: {e}")
            raise
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
        """ Aggregate logs by day.
        Args:
            metric: Aggregation metric ('count', 'unique_users', 'actions')

        Returns:
            pd.DataFrame: DataFrame containing the aggregated logs
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
            # Total actions per day
            result = df.groupby('date')['action'].count().reset_index(name='actions')
            # Pivot to wide format
            result = result.pivot(index='date', columns='action', values='actions').fillna(0)
            result = result.reset_index()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        return result

    
    def detect_anomalies(self, column: str = 'count',
                        method: str = 'zscore',
                        threshold: float = 3.0) -> pd.DataFrame:
        """ Detect anomalies in the data.
        Args:
            column (str): Column to detect anomalies in
            method (str): Method to use for anomaly detection ('zscore', 'iqr', 'lof')
            threshold (float): Threshold for anomaly detection

        Returns:
            pd.DataFrame: DataFrame containing the anomalies
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_logs() first.")

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
            # Calculate moving average
            daily['moving_avg'] = daily[column].rolling(window=7, center=True).mean()
            daily['moving_std'] = daily[column].rolling(window=7, center=True).std()
            daily['anomaly_score'] = np.abs(
                (daily[column] - daily['moving_avg']) / daily['moving_std']
            )
            daily['is_anomaly'] = daily['anomaly_score'] > threshold
        else:
            raise ValueError(f"Unsupported method: {method}")
        return daily

    def user_cohort_analysis(self, cohort_period: str = 'M') -> pd.DataFrame:
        """ Analyze user cohorts.
        Args:
            cohort_period: Period to analyze ('D', 'W', 'M')

        Returns:
            pd.DataFrame: DataFrame containing the user cohorts
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_logs() first.")

        df = self.data.copy()
        first_activity = df.groupby('user_id')['timestamp'].min().reset_index()
        first_activity.columns = ['user_id', 'cohort_date']
        first_activity['cohort'] = first_activity['cohort_date'].dt.to_period(cohort_period)
        
        df = df.merge(first_activity[['user_id', 'cohort']], on='user_id')
        df['activity_period'] = df['timestamp'].dt.to_period(cohort_period)
        df['period_index'] = (df['activity_period'] - df['cohort']).apply(lambda x: x.n)

        cohort_data = df.groupby(['cohort', 'period_index'])['user_id'].nunique().reset_index()
        cohort_data.columns = ['cohort', 'period_index', 'users']
       
        # Pivot to wide format
        cohort_matrix = cohort_data.pivot(index='cohort', columns='period_index', values='users')
        # Calculate retention percentage
        cohort_sizes = cohort_matrix[0]  # Period 0 = cohort size
        retention = cohort_matrix.div(cohort_sizes, axis=0) * 100
        return retention

    def export_results(self, df: pd.DataFrame, output_file: str,
                      format: str = 'csv') -> None:
        """ Export the results to a file.
        Args:
            df: DataFrame containing the results
            output_file: Path to the output file
            format: File format ('csv', 'json', 'parquet')
        """
        if format == 'csv':
            df.to_csv(output_file, index=False)
        elif format == 'json':
            df.to_json(output_file, index=False)
        elif format == 'parquet':
            df.to_parquet(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


class TestLogProcessor(unittest.TestCase):

    def setUp(self):
        self.sample_file = Path(tempfile.gettempdir()) / "test_sample_logs.csv"
        self._write_sample_logs(self.sample_file)
        self.processor = LogProcessor()
        self.processor.load_logs(str(self.sample_file), format='csv')
        self.df = self.processor.data.copy()

    def tearDown(self):
        try:
            self.sample_file.unlink()
        except FileNotFoundError:
            pass

    def _write_sample_logs(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        counts_per_day = [5, 5, 120]
        records = []
        for day, count in zip(pd.date_range("2024-01-01", periods=len(counts_per_day), freq="D"), counts_per_day):
            for idx in range(count):
                records.append({
                    "timestamp": day + pd.Timedelta(minutes=idx),
                    "user_id": f"user_{idx % 4}",
                    "action": ["view", "click", "purchase"][idx % 3],
                })
        pd.DataFrame(records).to_csv(path, index=False)
    
    def test_load_logs(self):
        self.assertEqual(len(self.df), 130)
        self.assertEqual(self.df['timestamp'].min(), pd.Timestamp('2024-01-01 00:00:00'))
        self.assertEqual(self.df['timestamp'].max(), pd.Timestamp('2024-01-03 01:59:00'))
        self.assertEqual(self.df['user_id'].nunique(), 4)
        self.assertEqual(self.df['action'].nunique(), 3)

    def test_aggregate_daily(self):
        daily = self.processor.aggregate_daily('count')
        self.assertEqual(len(daily), 3)
        self.assertEqual(daily['count'].sum(), 130)
        self.assertEqual(daily['date'].nunique(), 3)
        converted_dates = pd.to_datetime(daily['date'])
        self.assertEqual(converted_dates.min(), pd.Timestamp('2024-01-01'))
        self.assertEqual(converted_dates.max(), pd.Timestamp('2024-01-03'))

    def test_detect_anomalies(self):
        anomalies = self.processor.detect_anomalies(threshold=1.0)
        self.assertEqual(len(anomalies), 3)
        self.assertEqual(int(anomalies['is_anomaly'].sum()), 1)
        self.assertGreater(anomalies['anomaly_score'].max(), 0)

    def test_user_cohort_analysis(self):
        cohorts = self.processor.user_cohort_analysis('M')
        self.assertEqual(len(cohorts), 1)
        self.assertEqual(cohorts.index.nunique(), 1)
        self.assertIn(0, cohorts.columns)
        self.assertEqual(float(cohorts.iloc[0, 0]), 100.0)

    def test_export_results(self):
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        temp_path = Path(tempfile.gettempdir()) / "test_results.csv"
        self.processor.export_results(df, temp_path, format='csv')
        self.assertTrue(temp_path.exists())
        self.assertEqual(len(df), 3)
        self.assertEqual(df['name'].nunique(), 3)
        self.assertEqual(df['age'].nunique(), 3)
        temp_path.unlink(missing_ok=True)




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

    # unit tests
    print("5. Unit tests...")
    unittest.main()
    print()


