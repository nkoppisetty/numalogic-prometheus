from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    name: str = "VanillaAE"
    family: str = "autoencoder"
    seq_len: int = 10
    n_features: int = 1
    retrain_freq_hr: int = 8
    resume_training: bool = False
    num_epochs: int = 50


@dataclass
class MetricsConfig:
    keys: List[str]
    compare_by: List[str]
    features: List[str]
    scrape_interval: int = 30


@dataclass
class OutputConfig:
    unified_metric_name: str
    unified_metrics: List[str]
    unified_strategy: str = "max"
    unified_weights: List[int] = None
    feature_anomalies: bool = True
