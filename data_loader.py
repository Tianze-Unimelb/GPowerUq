import numpy as np
import pandas as pd

def load_and_normalize(wind_path, load_path):
    """
    加载风电和负荷数据，并进行归一化处理
    """
    wind_data = pd.read_csv(wind_path).values
    load_data = pd.read_csv(load_path).values

    # 归一化
    wind_data = (wind_data - np.mean(wind_data, axis=0)) / np.std(wind_data, axis=0)
    load_data = (load_data - np.mean(load_data, axis=0)) / np.std(load_data, axis=0)

    # 合并数据
    data = np.hstack([load_data, wind_data])
    time_labels = np.random.randint(1, 13, size=len(data))  # 模拟月份标签

    return data, time_labels