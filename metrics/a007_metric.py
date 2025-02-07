import torch
import numpy as np
from typing import List

from metrics.basemetric import BaseMetric


class A007_Metrics(BaseMetric):
    def __init__(self, thresholds: List[float]):
        super().__init__()
        self.thresholds = thresholds
        self.results = {threshold: [] for threshold in thresholds}

    def process_batch(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        outputs = torch.sigmoid(outputs)
        for threshold in self.thresholds:
            preds = (outputs > threshold).int()
            self.results[threshold].append((preds.cpu().numpy(), targets.cpu().numpy()))

    def compute_metric(self) -> dict:
        metrics = {}
        for threshold, batch_results in self.results.items():
            all_preds = []
            all_targets = []
            for preds, targets in batch_results:
                all_preds.append(preds)
                all_targets.append(targets)

            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            accuracy = self._compute_accuracy(all_preds, all_targets)
            precision, recall = self._compute_precision_recall(all_preds, all_targets)

            metrics[threshold] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall
            }
        return metrics

    def _compute_accuracy(self, preds: np.ndarray, targets: np.ndarray) -> float:
        correct = np.all(preds == targets, axis=1).sum()
        total = targets.shape[0]
        return correct / total

    def _compute_precision_recall(self, preds: np.ndarray, targets: np.ndarray) -> (float, float):
        tp = np.sum((preds == 1) & (targets == 1), axis=0)
        fp = np.sum((preds == 1) & (targets == 0), axis=0)
        fn = np.sum((preds == 0) & (targets == 1), axis=0)

        precisioin = np.mean(tp / (tp + fp + 1e-10))
        recall = np.mean(tp / (tp + fn + 1e-10))
        return precisioin, recall

    def reset(self) -> None:
        self.results = {threshold: [] for threshold in self.thresholds}
