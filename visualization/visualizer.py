import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime


class Visualizer:
    def __init__(self, experiment_name, metrics):
        self.log_dir = self._create_log_dir(experiment_name)
        self.logger = self._init_logger()
        self.loss_history = []
        self.metrics_history = {t: {'accuracy':[], 'precision':[], 'recall':[]} for t in metrics.thresholds}
        self.label_accuracy = {label:[] for label in ['N', 'A', 'C', 'D', 'G', 'H', 'M', 'O']}

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

    def update_metrics(self, metrics):
        for threshold, values in metrics.items():
            for metric, value in values.items():
                self.metrics_history[threshold][metric].append(value)
        plt.figure()
        for threshold, values in self.metrics_history.items():
            plt.plot(values['accuracy'], label=f'Accuracy (Threshold={threshold})')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Metrics Over Time')
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, f'metrics_{threshold}.png'))
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