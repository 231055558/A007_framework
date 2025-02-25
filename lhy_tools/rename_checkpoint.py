import torch

def rename_checkpoint(checkpoints_path, output_path):
    checkpoint = torch.load(checkpoints_path, map_location='cpu')
    new_state_dict = {}
    for key, value in checkpoint.items():
        if 'backbone' in key:
            new_key = key.replace('backbone.', '')

        if 'projection' in new_key:
            new_key = new_key.replace('projection', 'proj')

        if 'ln' in new_key:
            new_key = new_key.replace('ln', 'norm')

        # if '0.' in new_key:
        #     new_key = new_key.replace('0.', '0.0.')

        if 'ffn' in new_key:
            new_key = new_key.replace('ffn.layers', 'ffn.fcs')

        if '0.0.' in new_key:
            new_key = new_key.replace('0.0.', '0.')


        if new_key is None:
            new_key = key

        new_state_dict[new_key] = value
        torch.save(new_state_dict, output_path)


if __name__ == "__main__":
    checkpoints_path = "../../checkpoints/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth"
    output_path = "../../checkpoints/vit-base.pth"
    rename_checkpoint(checkpoints_path, output_path)