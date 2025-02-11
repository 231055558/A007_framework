import os

import torch
from tqdm import tqdm


def predict_model(
        model,
        test_loader,
        metric,
        model_name="default_model",
        device='cuda',
        output_folder="output"
):
    model.to(device)
    model.eval()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Validation')
        all_labels = []
        all_output = []
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            all_labels.append(labels.cpu())
            all_output.append(outputs.cpu())

        all_labels = torch.cat(all_labels, dim=0)
        all_output = torch.cat(all_output, dim=0)

        for threshold in metric.thresholds:
            file_path = os.path.join(output_folder, f"{threshold}.txt")
            with open(file_path, "w") as f:
                binary_outputs = (all_output >= threshold).int()
                all_labels = all_labels.int()
                for label, output in zip(all_labels, binary_outputs):
                    label_str = ''.join(map(str, label.tolist()))
                    output_str = ''.join(map(str, output.tolist()))

                    f.write(f"{label_str} {output_str}\n")
    print("预测结果已经保存")