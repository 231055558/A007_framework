import torch


def load_model_weights(model, checkpoint_path):
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path)

    # 提取checkpoint中的state_dict
    checkpoint_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # 获取当前模型的state_dict
    model_state_dict = model.state_dict()

    # 记录模型中没有被赋值的key
    missing_keys = []
    # 记录权重文件中没有找到的key
    unexpected_keys = []

    # 遍历模型的所有层，检查是否能从checkpoint中找到对应的权重
    for key in model_state_dict:
        if key not in checkpoint_state_dict:
            missing_keys.append(key)  # 如果模型中有，但权重文件没有对应的key
        else:
            # 如果存在，检查权重形状是否匹配
            if checkpoint_state_dict[key].shape != model_state_dict[key].shape:
                unexpected_keys.append(key)  # 如果形状不匹配
            else:
                model_state_dict[key] = checkpoint_state_dict[key]  # 否则正常赋值

    # 遍历checkpoint中的所有key，检查是否有多余的key
    for key in checkpoint_state_dict:
        if key not in model_state_dict:
            unexpected_keys.append(key)  # 如果权重文件中有，但模型没有这个key

    # 加载更新后的state_dict到模型
    model.load_state_dict(model_state_dict)

    # 输出未赋值的key和未找到的key
    if missing_keys:
        print("Missing keys in checkpoint:")
        for key in missing_keys:
            print(key)
    else:
        print("No missing keys in checkpoint.")
    print("\n\n---------------------------------------------\n\n")
    if unexpected_keys:
        print("Unexpected keys in checkpoint:")
        for key in unexpected_keys:
            print(key)
    else:
        print("No unexpected keys in checkpoint.")

    return model


if __name__ == "__main__":
    from networks.resnet import ResNet
    model = ResNet(depth=50,
                   num_classes=8)
    checkpoint_path = '../../checkpoints/resnet50.pth'  # 请替换为实际的文件路径
    model = load_model_weights(model, checkpoint_path)