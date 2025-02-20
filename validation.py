import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

def evaluate_high_dim(true_data, model_samples):
    """
    计算高维数据的Pearson和Spearman误差矩阵
    """
    n_dim = true_data.shape[1]
    pearson_errors = np.zeros((n_dim, n_dim))
    spearman_errors = np.zeros_like(pearson_errors)

    for i in range(n_dim):
        for j in range(n_dim):
            if i == j:
                pearson_errors[i, j] = 0  # 对角线设为0
                spearman_errors[i, j] = 0
            else:
                p_true, _ = pearsonr(true_data[:, i], true_data[:, j])
                p_model, _ = pearsonr(model_samples[:, i], model_samples[:, j])
                s_true, _ = spearmanr(true_data[:, i], true_data[:, j])
                s_model, _ = spearmanr(model_samples[:, i], model_samples[:, j])
                pearson_errors[i, j] = np.abs(p_true - p_model)
                spearman_errors[i, j] = np.abs(s_true - s_model)

    return pearson_errors, spearman_errors

def plot_error_heatmap(error_matrix, title, var_labels=None):
    """
    绘制高维误差矩阵热力图
    """
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(
        error_matrix,
        cmap="YlOrRd",
        annot=False,
        mask=np.triu(np.ones_like(error_matrix, dtype=bool)),
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Correlation Error"}
    )

    # 添加变量类型标注
    if var_labels:
        ax.annotate("Load Variables (1-32)",
                   xy=(0.02, 0.95),
                   xycoords="axes fraction",
                   fontsize=12,
                   color="darkblue")
        ax.annotate("Wind Power (33-36)",
                   xy=(0.02, 0.91),
                   xycoords="axes fraction",
                   fontsize=12,
                   color="darkgreen")
        rect = Rectangle(
            (32, 32), 4, 4,
            linewidth=2,
            edgecolor="darkgreen",
            facecolor="none",
            linestyle="--"
        )
        ax.add_patch(rect)

    # 设置坐标轴标签
    ax.set_xticks([16.5, 34.5])
    ax.set_xticklabels(["Load Variables", "Wind Power"],
                      rotation=45, ha="right", fontsize=10)
    ax.set_yticks([16.5, 34.5])
    ax.set_yticklabels(["Load Variables", "Wind Power"],
                      rotation=0, va="center", fontsize=10)

    # 美化设置
    plt.title(f"{title}\n(Upper Triangle Masked)",
             fontsize=14,
             pad=20)
    plt.xlabel("Variable Index", fontsize=12)
    plt.ylabel("Variable Index", fontsize=12)
    plt.tight_layout()
    plt.show()