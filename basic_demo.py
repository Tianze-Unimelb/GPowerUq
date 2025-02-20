import numpy as np
from core.kde_dphem import KDE_DPHEM
from validation import evaluate_high_dim, plot_error_heatmap

# 生成模拟数据
np.random.seed(42)
data = np.concatenate([
    np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 500),
    np.random.multivariate_normal([3, 3], [[1, -0.5], [-0.5, 1]], 500)
])

# 训练模型
model = KDE_DPHEM(n_components=50).fit(data)

# 生成样本并评估
samples, _ = model.sample(1000)
pearson_errors, spearman_errors = evaluate_high_dim(data, samples)

# 绘制热力图
plot_error_heatmap(pearson_errors, title="Pearson Correlation Error (2D Example)")