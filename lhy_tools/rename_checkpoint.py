import torch

def rename_checkpoint(checkpoints_path, output_path):
    checkpoint = torch.load(checkpoints_path, map_location='cpu')
    new_state_dict = {}
    for key, value in checkpoint.items():
        # if 'backbone' in key and not 'layer' in key:
        #     key = key.replace('backbone.', 'stem.0.')
        #     if 'conv1' in key:
        #         key = key.replace('conv1', 'conv')
        #     elif 'bn' in key:
        #         key = key.replace('bn1', 'norm')

        if 'backbone' in key:
            key = key.replace('backbone.', '')
            for i in [1,2,3]:
                if f'conv{i}' in key:
                    key = key.replace(f'conv{i}', f'conv{i}.conv')
                elif f'bn{i}' in key:
                    key = key.replace(f'bn{i}', f'conv{i}.norm')

        new_state_dict[key] = value
        torch.save(new_state_dict, output_path)


if __name__ == "__main__":
    checkpoints_path = "/mnt/mydisk/medical_seg/fwwb_a007/A007_framework/models/deeplabv3+/best_model1.pth"
    output_path = "/mnt/mydisk/medical_seg/fwwb_a007/A007_framework/models/deeplabv3+/best_model2.pth"
    rename_checkpoint(checkpoints_path, output_path)