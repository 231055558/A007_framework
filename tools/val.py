from typing import List

import torch
from tqdm import tqdm
def val_model(
        model,
        val_loader,
        metric,
        model_name="default_model",
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

    # for threshold in metric.thresholds:
    #     print(f'Threshold: {threshold}')
    #     print(f'Accuracy: {metrics[threshold]["accuracy"]:.4f}')
    #     print(f'Precision: {metrics[threshold]["precision"]:.4f}')
    #     print(f'Recall: {metrics[threshold]["recall"]:.4f}')

    metric.print_metric(metrics)

    return metrics


def val_color_merge_model(
        model,
        val_loader,
        metric,
        model_name="default_model",
        device='cuda'
):
    model.to(device)
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for inputs_l, inputs_r, labels in progress_bar:
            combined_inputs = torch.cat((inputs_l, inputs_r), dim=1)
            inputs = combined_inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            metric(outputs, labels)

    metrics = metric.compute_metric()

    metric.print_metric(metrics)

    return metrics


def val_output_merge_model(
        model,
        val_loader,
        metric,
        model_name="default_model",
        device='cuda'
):
    model.to(device)
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for inputs_l, inputs_r, labels in progress_bar:
            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)

            outputs_l = model(inputs_l)
            outputs_r = model(inputs_r)
            outputs = outputs_l + outputs_r
            metric(outputs, labels)

    metrics = metric.compute_metric()

    metric.print_metric(metrics)

    return metrics


def val_linear_merge_model(
        model,
        val_loader,
        metric,
        model_name="default_model",
        device='cuda'
):
    model.to(device)
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for inputs_l, inputs_r, labels in progress_bar:
            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)

            outputs = model(inputs_l, inputs_r)
            metric(outputs, labels)

    metrics = metric.compute_metric()

    metric.print_metric(metrics)

    return metrics

def val_double_merge_model(
        model_1,
        model_2,
        head,
        val_loader,
        metric,
        model_name="default_model",
        device='cuda'
):
    model_1.to(device)
    model_2.to(device)
    head.to(device)
    model_1.eval()
    model_2.eval()
    head.eval()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for inputs_l, inputs_r, labels in progress_bar:
            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)

            outputs_l = model_1(inputs_l)
            outputs_r = model_2(inputs_r)
            outputs = head(torch.cat((outputs_l, outputs_r), dim=1))
            metric(outputs, labels)

    metrics = metric.compute_metric()

    metric.print_metric(metrics)

    return metrics

