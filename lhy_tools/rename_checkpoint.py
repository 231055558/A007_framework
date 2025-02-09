import torch

def rename_checkpoint(checkpoints_path, output_path):
    checkpoint = torch.load(checkpoints_path, map_location='cpu')
    new_state_dict = {}
    for key, value in checkpoint.items():
        if "layer" not in key:
            new_key = key.replace("conv1.weight", "stem.0.conv.weight")
            new_key = new_key.replace("bn1.weight", "stem.0.norm.weight")
            new_key = new_key.replace("bn1.bias", "stem.0.norm.bias")
        else:
            if "conv" in key:
                new_key = key
                for i in range(1, 4):
                    new_key = new_key.replace(f'conv{i}', f'conv{i}.conv')
            elif "bn" in key:
                new_key = key
                for i in range(1, 4):
                    new_key = new_key.replace(f'bn{i}', f'conv{i}.norm')
        if new_key is None:
            new_key = key

        new_state_dict[new_key] = value
        torch.save(new_state_dict, output_path)


if __name__ == "__main__":
    checkpoints_path = "../../checkpoints/resnet50-0676ba61.pth"
    output_path = "../../checkpoints/resnet50.pth"
    rename_checkpoint(checkpoints_path, output_path)