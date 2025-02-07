from typing import List

import torch
from tqdm import tqdm
def val_model(
        model,
        val_loader,
        metric,
        device='cuda'
):
    model.to(device)
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            metric(outputs, labels)

    metrics = metric.compute_metric()

    for threshold in metric.thresholds:
        print(f'Threshold: {threshold}')
        print(f'Accuracy: {metrics[threshold]["accuracy"]:.4f}')
        print(f'Precision: {metrics[threshold]["precision"]:.4f}')
        print(f'Recall: {metrics[threshold]["recall"]:.4f}')

    return metrics