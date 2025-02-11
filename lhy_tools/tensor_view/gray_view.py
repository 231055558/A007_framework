import matplotlib.pyplot as plt
def gray(feature_map):
    feature_map = feature_map.cpu().detach().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(feature_map, cmap='gray', interpolation='nearest')
    plt.title('Grayscale of the selected feature map')
    plt.colorbar()
    plt.show()
