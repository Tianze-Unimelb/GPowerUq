import numpy as np
from core.hierarchical import HierarchicalModel
from validation import evaluate_high_dim, plot_error_heatmap
from utils.data_loader import load_and_normalize

# 加载数据
data, time_labels = load_and_normalize("data/sample_wind.csv", "data/sample_load.csv")

# 分层建模
model = HierarchicalModel(base_components=1000, final_components=100).fit(data, time_labels)

# 生成样本并评估
samples = model.sample(10000)
pearson_errors, spearman_errors = evaluate_high_dim(data, samples)

# 绘制热力图
plot_error_heatmap(pearson_errors, title="Pearson Correlation Error (36D Example)", var_labels=["Load"]*32 + ["Wind"]*4)