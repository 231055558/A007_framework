import numpy as np
import torch
from typing import List, Dict
from metrics.basemetric import BaseMetric


class A007_Metrics_Sample(BaseMetric):
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


class A007_Metrics_Label(BaseMetric):
    def __init__(self, thresholds: List[float], num_labels=8):
        super().__init__()
        self.thresholds = thresholds
        self.num_labels = num_labels
        self.results = {threshold: [] for threshold in thresholds}

    def process_batch(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        outputs = torch.sigmoid(outputs)  # 将输出转换为概率
        for threshold in self.thresholds:
            preds = (outputs > threshold).int()  # 二值化预测结果
            self.results[threshold].append((
                preds.cpu().numpy(),  # shape: (batch_size, num_labels)
                targets.cpu().numpy()  # shape: (batch_size, num_labels)
            ))

    def compute_metric(self) -> Dict[float, Dict]:
        metrics = {}
        eps = 1e-10  # 防除零小量

        for threshold, batch_results in self.results.items():
            # 合并所有批次的预测和标签
            all_preds = np.concatenate([p for p, t in batch_results], axis=0)
            all_targets = np.concatenate([t for p, t in batch_results], axis=0)

            # 初始化存储结构
            label_metrics = {}
            for label_idx in range(self.num_labels):
                label_metrics[label_idx] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0
                }

            # 计算每个标签的指标
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_correct = 0
            total_samples = all_preds.shape[0]

            for label_idx in range(self.num_labels):
                pred_label = all_preds[:, label_idx]
                target_label = all_targets[:, label_idx]

                # 计算准确率
                correct = (pred_label == target_label).sum()
                accuracy = correct / total_samples

                # 计算 TP/FP/FN
                tp = ((pred_label == 1) & (target_label == 1)).sum()
                fp = ((pred_label == 1) & (target_label == 0)).sum()
                fn = ((pred_label == 0) & (target_label == 1)).sum()

                # 计算精确率和召回率
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)

                # 存储单标签指标
                label_metrics[label_idx]["accuracy"] = float(accuracy)
                label_metrics[label_idx]["precision"] = float(precision)
                label_metrics[label_idx]["recall"] = float(recall)

                # 累计总体统计量
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_correct += correct

            # 计算总体指标
            total_accuracy = total_correct / (total_samples * self.num_labels)  # 总正确标签比例
            total_precision = total_tp / (total_tp + total_fp + eps)  # 微平均精确率
            total_recall = total_tp / (total_tp + total_fn + eps)  # 微平均召回率

            metrics[threshold] = {
                "label_metrics": label_metrics,  # 各标签的指标
                "overall_accuracy": float(total_accuracy),  # 总体准确率
                "overall_precision": float(total_precision),
                "overall_recall": float(total_recall)
            }

        return metrics

    def print_metric(self, metrics: Dict[float, Dict]):
        for threshold, metric_dict in metrics.items():
            print(f"\nThreshold: {threshold:.2f}")
            print("-" * 50)

            # 打印各标签指标
            for label_idx in range(self.num_labels):
                acc = metric_dict["label_metrics"][label_idx]["accuracy"]
                prec = metric_dict["label_metrics"][label_idx]["precision"]
                rec = metric_dict["label_metrics"][label_idx]["recall"]
                print(f"Label {label_idx + 1}:")
                print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

            # 打印总体指标
            print("\nOverall Metrics:")
            print(f"  Accuracy: {metric_dict['overall_accuracy']:.4f}")
            print(f"  Precision: {metric_dict['overall_precision']:.4f}")
            print(f"  Recall: {metric_dict['overall_recall']:.4f}")
            print("-" * 50)

    def reset(self) -> None:
        self.results = {threshold: [] for threshold in self.thresholds}
