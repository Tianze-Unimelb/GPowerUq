# GPowerUq: Gaussian Process-Powered Uncertainty Quantification for Power Systems

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

An uncertainty quantification toolkit for power systems using Gaussian Processes and KDE-DPHEM algorithms, featuring hierarchical multi-scale modeling (minute/hour/day) and 36+ dimensional correlation validation with Wasserstein-Copula metrics.

## 1. Introduction 🌟
<!-- 本代码库展示了论文《基于高斯过程的电力系统不确定性建模》中的核心方法：  
- **KDE-DPHEM算法**：通过核密度估计与改进的EM算法提升高斯混合模型精度。  
- **分层建模**：支持时间尺度分层与组合-缩减策略，解决高维数据组合爆炸问题。  
- **高维相关性验证**：提供36维电力系统不确定性建模示例。
-->

### Core Innovations 🚀
- **KDE-DPHEM Algorithm**：Implements high-precision Gaussian Mixture Modeling through Kernel Density Estimation (KDE) and enhanced Expectation-Maximization (EM) algorithm, demonstrating significant performance improvements over traditional methods
- **Multi-scale Hierarchical Architecture:**：
  - **⏳ Temporal Layering**：Supports minute-hour-day multi-timescale joint modeling
  - **🧩 Dimensional Layering**：Solves high-dimensional modeling challenges using combination-reduction strategy, supporting 36+ dimensional systems
- **📊 Verifiable Correlation**：Built-in validation modules with Wasserstein distance and Copula correlation analysis to ensure high-dimensional joint distribution fidelity

---

## 2. Quick Start  ⚡
### **Dependency Installation**
```bash
# Create virtual environment (recommended)
conda create -n gp-uq python=3.8
conda activate gp-uq

# # Install dependencies
pip install -r requirements.txt
```
### **Example Execution**
```bash
# Basic scenario: Wind-load joint modeling (2D)
python examples/basic_demo.py
# 🖼️ Expected outputs:
# - Convergence curve of fitting process
# - 3D visualization of joint distribution
# - Q-Q plots of marginal distributions

# High-dimensional scenario: 36-node provincial grid modeling
python examples/high_dim_demo.py
# 📈 Expected outputs:
# - Time statistics for hierarchical modeling
# - KL divergence distribution boxplots
# - Typical correlation coefficient matrix
```

## 3. Code Architecture 📂
```bash
GPowerUq/
├── core/                  # Core algorithm modules
│   ├── kde_dphem.py       # KDE-DPHEM implementation
│   ├── hierarchical.py    # Hierarchical modeling logic
│   └── validation.py      # Correlation validation & visualization
├── data/                  # Sample datasets
│   ├── sample_wind.csv    # Simulated wind power data
│   └── sample_load.csv    # Simulated load data
├── examples/              # Example scripts
│   ├── basic_demo.py      # Basic modeling demo
│   └── high_dim_demo.py   # High-dim system demo
├── utils/                 # Utility functions
│   ├── data_loader.py     # Data preprocessing
│   └── plotter.py         # Visualization enhancements
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## 4. Data Interface Specification 📊
- Input Format: CSV files with variables as columns (e.g., wind, load) and timestamps as rows
- Preprocessing：See ```bash utils/data_loader.py ``` function in```bashload_and_normalize```

## 5. Academic Citation 📚
This project is licensed under MIT License. For academic use, please cite our original paper.

## 6. Important Tips ⚠️ 
This code repository serves as the demonstration version for the paper **"Gaussian Process-Based Uncertainty Modeling for Power Systems"**, containing core algorithm implementations and basic functional modules. Example codes have been appropriately simplified.



