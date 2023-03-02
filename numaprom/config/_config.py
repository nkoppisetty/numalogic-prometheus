from typing import List
from dataclasses import dataclass, field
from omegaconf import MISSING

from numalogic.config import NumalogicConf


@dataclass
class UnifiedConf:
    unified_metric_name: str
    unified_metrics: List[str]
    unified_strategy: str = "max"
    unified_weights: List[int] = field(default_factory=list)


@dataclass
class MetricConf:
    metric: str = "default"
    composite_keys: List[str] = field(default_factory=lambda: ["namespace", "name"])
    static_threshold: int = 3
    scrape_interval: int = 30
    retrain_freq_hr: int = 8
    resume_training: bool = False
    numalogic_conf: NumalogicConf = MISSING


@dataclass
class NamespaceConf:
    namespace: str
    metric_configs: List[MetricConf]
    unified_configs: List[UnifiedConf]


@dataclass
class NumapromConf:
    configs: List[NamespaceConf]
