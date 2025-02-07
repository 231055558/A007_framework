from abc import ABCMeta, abstractmethod
from typing import List, Any

class BaseMetric(metaclass=ABCMeta):
    def __init__(self):
        self.results = []

    @abstractmethod
    def process_batch(self, outputs: Any, targets: Any) -> None:
        pass

    @abstractmethod
    def compute_metric(self) -> Any:
        pass

    def reset(self) -> None:
        self.results = []

    def __call__(self, outputs: Any, targets: Any) -> None:
        self.process_batch(outputs, targets)
