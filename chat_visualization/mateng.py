# coding=GBK
# 作者：马腾
# 日期:2024/11/21
# 程序说明:绘制聚类图。
# 输入参数:

# data:字典，包含若干个方法的若干个指标的评分数据。methods:列表，包含若干个方法的名称。metrics:列表，包含若干个指标的名称。# 输出结果:
# 输出结果:绘制的实验结果图。
# 使用方法:直接运行即可。
# 注意事项:
# 1.程序中，data 的格式为字典，包含若干个方法的若干个指标的评分数据。
# 2.程序中，methods 的格式为列表，包含若干个方法的名称。

import matplotlib.pyplot as plt

# 假设数据
x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nmi_cse = [45, 46, 46.5, 47, 47.5, 47.8, 47.9, 48, 48.1, 48.2]
nmi_lse = [44, 45, 45.5, 46, 46.5, 46.8, 46.9, 47, 47.1, 47.2]

plt.plot(x, nmi_cse, 'r-', label='CSE')
plt.plot(x, nmi_lse, 'b--', label='LSEnet')
plt.xlabel('Number of randomly rooted nodes')
plt.ylabel('NMI (%)')
plt.legend()
plt.title('NMI vs Number of Randomly Rooted Nodes')
plt.show()

# 假设数据
scale = [100, 200, 300, 400, 500]
time_lse = [0.020, 0.028, 0.031, 0.034, 0.042]
time_cse = [0.017, 0.035, 0.043, 0.049, 0.052]

plt.plot(scale, time_lse, 'bo-', label='LSEnet')
plt.plot(scale, time_cse, 'r*-', label='CSE')
plt.xlabel('Data Scale')
plt.ylabel('Time (Seconds)')
plt.legend()
plt.title('Runtime Comparison')
plt.show()


import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# 假设我们有一些高维数据
high_dim_data = np.random.rand(1000, 50)  # 1000个样本，每个样本50维

# 使用t-SNE降维到2维
tsne = TSNE(n_components=2, random_state=42)
low_dim_data = tsne.fit_transform(high_dim_data)

# 应用K-means聚类
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(low_dim_data)

# 绘制图像
plt.scatter(low_dim_data[:, 0], low_dim_data[:, 1], c=clusters, cmap='viridis')
plt.title('Cluster Visualization with t-SNE')
plt.show()


