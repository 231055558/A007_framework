import matplotlib.pyplot as plt
def heat(feature_map):
    feature_map = feature_map.cpu().detach().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(feature_map, cmap='hot', interpolation='nearest')
    plt.title('Heatmap of the selected feature map')
    plt.colorbar()
    plt.show()
