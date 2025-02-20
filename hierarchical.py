import numpy as np
from .kde_dphem import KDE_DPHEM

class HierarchicalModel:
    def __init__(self, base_components=100, final_components=50):
        self.base_components = base_components
        self.final_components = final_components

    def fit(self, X, time_labels):
        unique_times = np.unique(time_labels)
        sub_models = []
        sample_weights = []

        # 子模型训练
        for t in unique_times:
            subset = X[time_labels == t]
            model = KDE_DPHEM(n_components=self.base_components).fit(subset)
            sub_models.append(model)
            sample_weights.append(len(subset) / len(X))

        # 模型聚合与缩减
        all_means = np.concatenate([m.means_ for m in sub_models])
        final_model = KDE_DPHEM(n_components=self.final_components).fit(all_means)
        self.model_ = final_model
        return self

    def sample(self, n_samples=1000):
        return self.model_.sample(n_samples)