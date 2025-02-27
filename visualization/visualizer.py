import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime


class Visualizer:
    # def __init__(self, experiment_name, metrics):
    #     self.log_dir = self._create_log_dir(experiment_name)
    #     self.logger = self._init_logger()
    #     self.loss_history = []
    #     self.metrics_history = {t: {'accuracy':[], 'precision':[], 'recall':[]} for t in metrics.thresholds}
    #     self.label_accuracy = {label:[] for label in ['N', 'A', 'C', 'D', 'G', 'H', 'M', 'O']}

    def __init__(self, experiment_name, metrics):
        self.log_dir = self._create_log_dir(experiment_name)
        self.logger = self._init_logger()
        self.loss_history = []

        # 新指标存储结构
        self.metrics_history = {
            t: {
                'label_metrics': {label_idx: {'accuracy': [], 'precision': [], 'recall': []} for label_idx in range(8)},
                'overall_accuracy': [],
                'overall_precision': [],
                'overall_recall': []
            }
            for t in metrics.thresholds
        }


    def _create_log_dir(self, experiment_name):
        """创建日志文件夹"""
        today = datetime.now().strftime("%Y-%m-%d")
        experiment_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        log_dir = os.path.join("../../logs", today, f"{experiment_time}_{experiment_name}")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _init_logger(self):
        """初始化日志记录器"""
        logger = logging.getLogger('experiment_logger')
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'experiment.log'))
        file_handler.setLevel(logging.INFO)

        fommatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(fommatter)
        file_handler.setFormatter(fommatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger

    def log(self, message):
        """记录日志"""
        self.logger.info(message)

    def update_loss(self, loss):
        self.loss_history.append(loss)
        plt.figure()
        plt.plot(self.loss_history, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'training_loss.png'))
        plt.close()

    # def update_metrics(self, metrics):
    #     for threshold, values in metrics.items():
    #         for metric, value in values.items():
    #             self.metrics_history[threshold][metric].append(value)
    #     plt.figure()
    #     for threshold, values in self.metrics_history.items():
    #         plt.plot(values['accuracy'], label=f'Accuracy (Threshold={threshold})')
    #         plt.xlabel('Epoch')
    #         plt.ylabel('Accuracy')
    #         plt.title('Metrics Over Time')
    #         plt.legend()
    #         plt.savefig(os.path.join(self.log_dir, f'metrics_{threshold}.png'))
    #         plt.close()

    def update_metrics(self, metrics):
        for threshold, thr_metrics in metrics.items():
            # 记录标签级别指标
            for label_idx in range(8):
                label_metric = thr_metrics["label_metrics"][label_idx]
                self.metrics_history[threshold]['label_metrics'][label_idx]['accuracy'].append(label_metric['accuracy'])
                self.metrics_history[threshold]['label_metrics'][label_idx]['precision'].append(
                    label_metric['precision'])
                self.metrics_history[threshold]['label_metrics'][label_idx]['recall'].append(label_metric['recall'])

            # 记录总体指标
            self.metrics_history[threshold]['overall_accuracy'].append(thr_metrics['overall_accuracy'])
            self.metrics_history[threshold]['overall_precision'].append(thr_metrics['overall_precision'])
            self.metrics_history[threshold]['overall_recall'].append(thr_metrics['overall_recall'])

        # 绘制标签级别指标（以 Label 0 的 Accuracy 为例）
        plt.figure(figsize=(12, 6))
        for label_idx in range(8):
            label_acc_history = [self.metrics_history[t]['label_metrics'][label_idx]['accuracy'][-1] for t in
                                 metrics.keys()]
            plt.plot(list(metrics.keys()), label_acc_history, label=f'Label {label_idx}')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Label-wise Accuracy Across Thresholds')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'label_accuracy.png'))
        plt.close()

        # 绘制总体指标（以 Accuracy 为例）
        plt.figure(figsize=(12, 6))
        for threshold in metrics.keys():
            overall_acc_history = self.metrics_history[threshold]['overall_accuracy']
            plt.plot(overall_acc_history, label=f'Threshold {threshold}')
        plt.xlabel('Epoch')
        plt.ylabel('Overall Accuracy')
        plt.title('Overall Accuracy Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'overall_accuracy.png'))
        plt.close()

    def update_label_accuracy(self, label_accuracy):
        for label, accuracy in label_accuracy.items():
            self.label_accuracy[label].append(accuracy)
        plt.figure()
        for label, accuracy in label_accuracy.items():
            plt.plot(accuracy, label=f"Label {label}")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Label Accuracy Over Time')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'label_accuracy.png'))
        plt.close()

    def log_error_samples(self, error_samples, thresholds):
        """记录并保存错误样本"""
        for threshold in thresholds:
            if threshold in error_samples:
                sample = error_samples[threshold]
                img, true_label, pred_label = sample
                plt.figure()
                plt.imshow(img)
                plt.title(f'True: {true_label}, Predicted: {pred_label}')
                plt.savefig(os.path.join(self.log_dir, f'error_sample_threshold_{threshold}.png'))
                plt.close()

    def log_metrics(self, metrics):
        for threshold, thr_metrics in metrics.items():
            self.log(f"Threshold {threshold} Metrics:")
            for label_idx in range(8):
                acc = thr_metrics["label_metrics"][label_idx]["accuracy"]
                prec = thr_metrics["label_metrics"][label_idx]["precision"]
                rec = thr_metrics["label_metrics"][label_idx]["recall"]
                self.log(f"  Label {label_idx}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
            self.log(
                f"  Overall: Acc={thr_metrics['overall_accuracy']:.4f}, Prec={thr_metrics['overall_precision']:.4f}, Rec={thr_metrics['overall_recall']:.4f}")