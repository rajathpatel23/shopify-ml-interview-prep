"""

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
"""
Quick Design brainstorm

-  do we want a batch operation or a stream operation?
- lets go with a batch operation for now

- what kind of data file do we have - 
    - csv, json, parquet, etc.
    - lets go with csv for now

- what kind of metrics do we want to aggregate?
    - daily, weekly, monthly
    - lets go with daily for now
    - are the metric defined or - we just calculating simple metrics like count, sum, average, etc.


- what kind of anomalies do we want to detect?
    - outliers, missing values, etc.
    - lets go with outliers for now
    - missing value monitors
    - what is our baseline for anomalies?

- what do we use for monitoring?
    - prometheus, grafana, etc.
    - lets go with prometheus for now
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
import traceback
import time
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path



class Metric(BaseModel):
    """Base metric with tags for flexible slicing."""
    model_config = ConfigDict(extra="forbid")
    name: str
    value: Union[int, float]
    unit: str = Field(default="count", description="Unit of the value (e.g., count, ms, rows)")
    step: Optional[str] = Field(default=None, description="Pipeline step emitting the metric")
    pipeline: Optional[str] = Field(default=None, description="Pipeline name")
    run_id: Optional[str] = Field(default=None, description="Unique id for the pipeline run")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = Field(default_factory=dict)


class CounterMetric(Metric):
    """Monotonic counter."""
    unit: str = "count"


class GaugeMetric(Metric):
    """Arbitrary value sampled at a point in time."""
    unit: str = "value"


class HistogramMetric(BaseModel):
    """
    Histogram over observations with tagged buckets.
    Buckets are labeled by an upper-bound string (e.g., '0.1','1','5','+Inf').
    """
    model_config = ConfigDict(extra="forbid")
    name: str
    buckets: Dict[str, int] = Field(default_factory=dict)
    count: int = 0
    sum: float = 0.0
    unit: str = "value"
    step: Optional[str] = None
    pipeline: Optional[str] = None
    run_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = Field(default_factory=dict)


class ErrorRecord(BaseModel):
    """First-class error representation captured during processing."""
    model_config = ConfigDict(extra="forbid")
    step: str
    message: str
    exception_type: Optional[str] = None
    traceback: Optional[str] = None
    count: int = 1
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = Field(default_factory=dict)


class Anomaly(BaseModel):
    """First-class anomaly with score and method metadata."""
    model_config = ConfigDict(extra="forbid")
    name: str
    description: Optional[str] = None
    score: Optional[float] = None
    is_anomaly: bool = True
    method: Optional[str] = None
    threshold: Optional[float] = None
    step: Optional[str] = None
    pipeline: Optional[str] = None
    run_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = Field(default_factory=dict)


class IOConfig(BaseModel):
    """Batch-only I/O configuration."""
    model_config = ConfigDict(extra="forbid")
    input_path: str
    input_format: str = Field(default="csv")
    output_path: Optional[str] = None
    output_format: Optional[str] = Field(default=None)
    required_columns: List[str] = Field(default_factory=list)
    pipeline_name: str = Field(default="DataPipeline")
    chunk_size: Optional[int] = Field(default=None, description="Number of rows per chunk when chunked processing is enabled.")


class PipelineRunReport(BaseModel):
    """Per-run report capturing status, metrics, errors, and anomalies."""
    model_config = ConfigDict(extra="forbid")
    run_id: str
    pipeline: str
    started_at: datetime
    ended_at: datetime
    status: str
    metrics: List[Metric] = Field(default_factory=list)
    histograms: List[HistogramMetric] = Field(default_factory=list)
    errors: List[ErrorRecord] = Field(default_factory=list)
    anomalies: List[Anomaly] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class StepResult:
    """Result of a single pipeline step."""
    data: Optional[pd.DataFrame] = None
    metrics: List[Metric] = field(default_factory=list)
    histograms: List[HistogramMetric] = field(default_factory=list)
    errors: List[ErrorRecord] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)


class DataPipeline(ABC):
    """Abstract multi-step, batch-only data pipeline with per-run reporting."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # Required for batch processing
    @abstractmethod
    def load(self, config: IOConfig) -> StepResult:
        """Load input data. Must be implemented."""
        raise NotImplementedError

    # Optional steps with pass-through defaults
    def validate(self, data: pd.DataFrame, config: IOConfig) -> StepResult:
        return StepResult(data=data)

    def transform(self, data: pd.DataFrame, config: IOConfig) -> StepResult:
        return StepResult(data=data)

    def aggregate(self, data: pd.DataFrame, config: IOConfig) -> StepResult:
        return StepResult(data=data)

    def detect_anomalies(self, data: pd.DataFrame, config: IOConfig) -> StepResult:
        return StepResult(data=data)

    def export(self, data: pd.DataFrame, config: IOConfig) -> StepResult:
        return StepResult(data=data)

    def run(self, config: IOConfig) -> PipelineRunReport:
        """Run the pipeline end-to-end, capturing per-step and overall metrics."""
        run_id = str(uuid4())
        started_at = datetime.now(timezone.utc)

        all_metrics: List[Metric] = []
        all_histograms: List[HistogramMetric] = []
        all_errors: List[ErrorRecord] = []
        all_anomalies: List[Anomaly] = []

        status = "success"
        final_df: Optional[pd.DataFrame] = None

        steps = [
            ("validate", self.validate),
            ("transform", self.transform),
            ("aggregate", self.aggregate),
            ("detect_anomalies", self.detect_anomalies),
            ("export", self.export),
        ]

        def _time_step(step_name: str, fn, *args, **kwargs) -> StepResult:
            nonlocal final_df, status
            start = time.perf_counter()
            try:
                result: StepResult = fn(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000.0
                rows = 0
                if result.data is not None and isinstance(result.data, pd.DataFrame):
                    rows = int(result.data.shape[0])
                    final_df = result.data
                all_metrics.append(GaugeMetric(
                    name="step_duration_ms",
                    value=duration_ms,
                    unit="ms",
                    step=step_name,
                    pipeline=config.pipeline_name,
                    run_id=run_id,
                    tags={}
                ))
                all_metrics.append(GaugeMetric(
                    name="rows_after_step",
                    value=rows,
                    unit="rows",
                    step=step_name,
                    pipeline=config.pipeline_name,
                    run_id=run_id,
                    tags={}
                ))
                all_metrics.extend(self._attach_common_fields(result.metrics, step_name, config.pipeline_name, run_id))
                all_histograms.extend(self._attach_common_fields(result.histograms, step_name, config.pipeline_name, run_id))
                all_anomalies.extend(self._attach_common_fields(result.anomalies, step_name, config.pipeline_name, run_id))
                all_errors.extend(self._attach_common_fields(result.errors, step_name, config.pipeline_name, run_id))
                return result
            except Exception as ex:
                duration_ms = (time.perf_counter() - start) * 1000.0
                status = "failed"
                tb = traceback.format_exc()
                all_metrics.append(GaugeMetric(
                    name="step_duration_ms",
                    value=duration_ms,
                    unit="ms",
                    step=step_name,
                    pipeline=config.pipeline_name,
                    run_id=run_id,
                    tags={"status": "failed"}
                ))
                all_errors.append(ErrorRecord(
                    step=step_name,
                    message=str(ex),
                    exception_type=type(ex).__name__,
                    traceback=tb,
                    tags={}
                ))
                raise

        def _process_steps(current_df: pd.DataFrame) -> pd.DataFrame:
            result_df = current_df
            for step_name, step_fn in steps:
                step_result = _time_step(step_name, step_fn, result_df, config)
                if step_result.data is None:
                    raise ValueError(f"{step_name} step did not return a DataFrame")
                result_df = step_result.data
            return result_df

        processed_any = False
        try:
            for chunk_source in self.chunked_load(config):
                processed_any = True
                load_result = _time_step("load", lambda chunk=chunk_source: chunk)
                chunk_df = load_result.data
                if chunk_df is None:
                    raise ValueError("Load step did not return a DataFrame")
                self._ensure_required_columns(chunk_df, config)
                _process_steps(chunk_df)
            if not processed_any:
                raise ValueError("Load step produced no data.")
        finally:
            ended_at = datetime.now(timezone.utc)
            overall_ms = (ended_at - started_at).total_seconds() * 1000.0
            all_metrics.append(GaugeMetric(
                name="pipeline_duration_ms",
                value=overall_ms,
                unit="ms",
                step=None,
                pipeline=config.pipeline_name,
                run_id=run_id,
                tags={}
            ))
            if final_df is not None:
                all_metrics.append(GaugeMetric(
                    name="rows_final",
                    value=int(final_df.shape[0]),
                    unit="rows",
                    step=None,
                    pipeline=config.pipeline_name,
                    run_id=run_id,
                    tags={}
                ))

            report = PipelineRunReport(
                run_id=run_id,
                pipeline=config.pipeline_name,
                started_at=started_at,
                ended_at=ended_at,
                status=status,
                metrics=all_metrics,
                histograms=all_histograms,
                errors=all_errors,
                anomalies=all_anomalies,
                summary={
                    "num_metrics": len(all_metrics),
                    "num_histograms": len(all_histograms),
                    "num_errors": len(all_errors),
                    "num_anomalies": len(all_anomalies),
                }
            )
            return report

    @staticmethod
    def _attach_common_fields(items: List[Any], step: Optional[str], pipeline: str, run_id: str) -> List[Any]:
        """Attach step/pipeline/run_id if missing."""
        attached: List[Any] = []
        for item in items:
            # Pydantic models are immutable by default unless configured; create updated copies
            if isinstance(item, (Metric, HistogramMetric, Anomaly)):
                item = item.model_copy(update={
                    "step": item.step or step,
                    "pipeline": item.pipeline or pipeline,
                    "run_id": item.run_id or run_id,
                })
            elif isinstance(item, ErrorRecord):
                item = item.model_copy(update={
                    "step": item.step or (step or "unknown"),
                })
            attached.append(item)
        return attached

    def _ensure_required_columns(self, df: pd.DataFrame, config: IOConfig) -> None:
        if config.required_columns:
            missing = set(config.required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns and "date" not in df.columns:
            df["date"] = df["timestamp"].dt.date
        return df

    def chunked_load(self, config: IOConfig) -> Iterable[StepResult]:
        """Yield chunked StepResult objects when chunking is configured."""
        if config.chunk_size and config.input_format == "csv":
            reader = pd.read_csv(
                config.input_path,
                parse_dates=["timestamp"],
                chunksize=config.chunk_size,
            )
            for chunk in reader:
                chunk = self._ensure_date_column(chunk)
                self._ensure_required_columns(chunk, config)
                yield StepResult(
                    data=chunk,
                    metrics=[CounterMetric(name="rows_loaded", value=len(chunk))]
                )
        else:
            yield self.load(config)


# Sample concrete pipeline and helpers


class SampleDataPipeline(DataPipeline):
    """Concrete example showing how a batch pipeline emits metrics and anomalies."""

    def load(self, config: IOConfig) -> StepResult:
        df = pd.read_csv(config.input_path, parse_dates=["timestamp"])
        if "date" not in df.columns:
            df["date"] = df["timestamp"].dt.date
        return StepResult(
            data=df,
            metrics=[CounterMetric(name="rows_loaded", value=len(df))]
        )

    def transform(self, data: pd.DataFrame, config: IOConfig) -> StepResult:
        filtered = data[data["status"] == "success"].copy()
        ratio = len(filtered) / len(data) if len(data) else 0.0
        return StepResult(
            data=filtered,
            metrics=[GaugeMetric(name="success_ratio", value=ratio)]
        )

    def aggregate(self, data: pd.DataFrame, config: IOConfig) -> StepResult:
        aggregated = data.groupby("date", as_index=False).agg(
            total_value=("value", "sum"),
            average_value=("value", "mean"),
            total_events=("value", "size"),
        )
        thresholds = [250, 400, 550]
        buckets: Dict[str, int] = {}
        for threshold in thresholds:
            bucket_key = f"<= {threshold}"
            buckets[bucket_key] = int((aggregated["total_value"] <= threshold).sum())
        buckets["+Inf"] = int((aggregated["total_value"] > thresholds[-1]).sum())
        histogram = HistogramMetric(
            name="daily_value_distribution",
            buckets=buckets,
            count=int(aggregated["total_events"].sum()) if not aggregated.empty else 0,
            sum=float(aggregated["total_value"].sum()) if not aggregated.empty else 0.0,
        )
        metrics = [
            GaugeMetric(name="total_value", value=float(aggregated["total_value"].sum())) if not aggregated.empty else GaugeMetric(name="total_value", value=0.0),
            GaugeMetric(name="avg_value_per_day", value=float(aggregated["average_value"].mean())) if not aggregated.empty else GaugeMetric(name="avg_value_per_day", value=0.0),
            CounterMetric(name="total_days_processed", value=len(aggregated)),
        ]
        return StepResult(data=aggregated, metrics=metrics, histograms=[histogram])

    def detect_anomalies(self, data: pd.DataFrame, config: IOConfig) -> StepResult:
        if data.empty:
            return StepResult(data=data)
        total_value = data["total_value"]
        threshold = float(total_value.mean() + 2 * total_value.std()) if len(total_value) > 1 else float(total_value.mean())
        spikes = data[total_value > threshold]
        anomalies = [
            Anomaly(
                name="daily_value_spike",
                description="Total value exceeded expected threshold",
                score=float(row["total_value"]),
                threshold=threshold,
                step="detect_anomalies",
                tags={"date": str(row["date"])},
            )
            for _, row in spikes.iterrows()
        ]
        return StepResult(
            data=data,
            anomalies=anomalies,
            metrics=[CounterMetric(name="anomaly_rows", value=len(anomalies))]
        )

    def export(self, data: pd.DataFrame, config: IOConfig) -> StepResult:
        if config.output_path:
            output_path = Path(config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(output_path, index=False)
        return StepResult(
            data=data,
            metrics=[CounterMetric(name="rows_exported", value=len(data))]
        )


def write_dummy_logs(path: Path, days: int = 4, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic log rows to feed the sample pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    timestamps = pd.date_range(start, periods=days * 24, freq="H")
    records = []
    for ts in timestamps:
        for _ in range(np.random.randint(6, 12)):
            records.append({
                "timestamp": ts,
                "user_id": f"user_{np.random.randint(1, 200)}",
                "status": np.random.choice(["success", "failure"], p=[0.75, 0.25]),
                "value": float(max(0.0, np.random.normal(300, 80))),
            })
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    sample_dir = Path("data")
    sample_file = sample_dir / "sample_logs.csv"
    aggregated_file = sample_dir / "sample_logs_aggregated.csv"
    write_dummy_logs(sample_file, days=5)

    # pipeline = SampleDataPipeline()
    config = IOConfig(
        input_path="data/sample_logs.csv",
        output_path="data/sample_logs_aggregated.csv",
        chunk_size=100_000,
        pipeline_name="dummy_batch_pipeline",
    )
    report = SampleDataPipeline().run(config)
    print(f"Pipeline run {report.status} ({report.run_id}).")
    print(f"Metrics emitted: {len(report.metrics)}")
    print(f"Histograms emitted: {len(report.histograms)}")
    print(f"Anomalies detected: {len(report.anomalies)}")








