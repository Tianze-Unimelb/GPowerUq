import numpy as np
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.utils.extmath import logsumexp
from tqdm import tqdm


class KDE_DPHEM:
    def __init__(self, n_components, max_iter=100, tol=1e-6, m_factor=10):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.m_factor = m_factor

    def fit(self, X):
        # KDE初始化基础GMM
        self.kde_ = gaussian_kde(X.T)
        n_samples, n_features = X.shape
        self.weights_base_ = np.ones(n_samples) / n_samples
        self.means_base_ = X
        self.covs_base_ = [np.diag(self.kde_.covariance)] * n_samples

        # DPHEM初始化
        self.weights_ = np.ones(self.n_components) / self.n_components
        idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[idx]
        self.covariances_ = [np.diag(self.kde_.covariance)] * self.n_components

        # DPHEM迭代优化
        prev_lower_bound = -np.inf
        m = self.m_factor * n_samples

        for _ in tqdm(range(self.max_iter), desc="DPHEM Optimization"):
            # E-Step: 计算变分参数
            log_resp = self._e_step(X, m)

            # M-Step: 更新参数
            self._m_step(X, log_resp)

            # 收敛性检查
            lower_bound = np.sum(log_resp * np.exp(log_resp))
            if np.abs(lower_bound - prev_lower_bound) < self.tol:
                break
            prev_lower_bound = lower_bound

        return self

    def _e_step(self, X, m):
        log_resp = np.zeros((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            for j in range(self.n_components):
                diff = X[i] - self.means_[j]
                cov_inv = np.linalg.inv(self.covariances_[j])
                A_ij = 0.5 * (
                        np.trace(cov_inv @ self.covs_base_[i]) +
                        np.log(np.linalg.det(self.covariances_[j])) +
                        diff.T @ cov_inv @ diff
                )
                log_resp[i, j] = -m * A_ij + np.log(self.weights_[j])
        log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
        return log_resp

    def _m_step(self, X, log_resp):
        resp = np.exp(log_resp)
        Nk = np.sum(resp, axis=0)
        self.weights_ = Nk / X.shape[0]
        self.means_ = (resp.T @ X) / Nk[:, None]
        for j in range(self.n_components):
            diff = X - self.means_[j]
            self.covariances_[j] = (resp[:, j] * (diff.T @ diff)) / Nk[j]

    def sample(self, n_samples=1):
        gmm = GaussianMixture(
            n_components=self.n_components,
            weights_init=self.weights_,
            means_init=self.means_,
            precisions_init=[np.linalg.inv(c) for c in self.covariances_]
        )
        gmm.fit(self.means_)  # Dummy fit to initialize parameters
        return gmm.sample(n_samples)