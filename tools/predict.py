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


def predict_output_merge_model(
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
        progress_bar = tqdm(test_loader, desc='Predict')
        all_labels = []
        all_output = []
        for inputs_l, inputs_r, labels in progress_bar:
            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)

            outputs_l = model(inputs_l)
            outputs_r = model(inputs_r)
            outputs = outputs_l + outputs_r

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

def predict_net_merge_model(
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
        all_path_l = []
        all_path_r = []
        for inputs_l, inputs_r, labels, path_l, path_r in progress_bar:
            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)
            outputs = model(inputs_l, inputs_r)
            #path_l是一个list，需要把里面的元素依此append到all_path_l中
            for i in range(len(path_l)):
                all_path_l.append(path_l[i])
                all_path_r.append(path_r[i])
            all_labels.append(labels.cpu())
            all_output.append(outputs.cpu())

        all_labels = torch.cat(all_labels, dim=0)
        all_output = torch.cat(all_output, dim=0)

        for threshold in metric.thresholds:
            file_path = os.path.join(output_folder, f"{threshold}.txt")
            with open(file_path, "w") as f:
                binary_outputs = (all_output >= threshold).int()
                all_labels = all_labels.int()
                for label, output, path_l, path_r  in zip(all_labels, binary_outputs, all_path_l, all_path_r):
                    path_l_str = ''.join(map(str, path_l))
                    path_r_str = ''.join(map(str, path_r))
                    label_str = ''.join(map(str, label.tolist()))
                    output_str = ''.join(map(str, output.tolist()))

                    f.write(f"{path_l_str} {path_r_str} {label_str} {output_str}\n")
    print("预测结果已经保存")

async def detect_model(
        model,
        test_loader,
        threshold,
        progress_callback=None,
        device='cuda',
        output_folder="output"
):
    model.to(device)
    model.eval()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with torch.no_grad():
        all_output = []
        all_path_l = []
        all_path_r = []
        total_batches = len(test_loader)
        for batch_idx, (inputs_l, inputs_r, path_l, path_r) in enumerate(test_loader):
            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            outputs = model(inputs_l, inputs_r)
            #path_l是一个list，需要把里面的元素依此append到all_path_l中
            for i in range(len(path_l)):
                all_path_l.append(path_l[i])
                all_path_r.append(path_r[i])
            all_output.append(outputs.cpu())
            # 发送进度更新
            if progress_callback:
                progress = int((batch_idx / total_batches) * 100)
                await progress_callback(progress, f"正在处理第 {batch_idx}/{total_batches} 个batch")

        all_output = torch.cat(all_output, dim=0)

        file_path = os.path.join(output_folder, f"{threshold}.txt")
        with open(file_path, "w") as f:
            binary_outputs = (all_output >= threshold).int()
            for output, path_l, path_r  in zip(binary_outputs, all_path_l, all_path_r):
                path_l_str = ''.join(map(str, path_l))
                path_r_str = ''.join(map(str, path_r))
                output_str = ''.join(map(str, output.tolist()))
                f.write(f"{path_l_str} {path_r_str} {output_str}\n")
    
    if progress_callback:
        await progress_callback(100, "预测完成")
    return '运行结果已经保存'