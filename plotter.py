import matplotlib.pyplot as plt

def plot_2d_comparison(true_data, model_samples, title):
    """
    绘制二维数据对比图
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(true_data[:, 0], true_data[:, 1], alpha=0.5, label='True Data')
    plt.title(f'True Data - {title}')
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.subplot(1, 2, 2)
    plt.scatter(model_samples[:, 0], model_samples[:, 1], alpha=0.5, c='red', label='GMM Samples')
    plt.title(f'Model Samples - {title}')
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.tight_layout()
    plt.show()