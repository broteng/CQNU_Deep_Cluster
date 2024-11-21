# coding=GBK
# ���ߣ�����
# ����:2024/11/21
# ����˵��:���ƾ���ͼ��
# �������:

# data:�ֵ䣬�������ɸ����������ɸ�ָ����������ݡ�methods:�б��������ɸ����������ơ�metrics:�б��������ɸ�ָ������ơ�# ������:
# ������:���Ƶ�ʵ����ͼ��
# ʹ�÷���:ֱ�����м��ɡ�
# ע������:
# 1.�����У�data �ĸ�ʽΪ�ֵ䣬�������ɸ����������ɸ�ָ����������ݡ�
# 2.�����У�methods �ĸ�ʽΪ�б��������ɸ����������ơ�

import matplotlib.pyplot as plt

# ��������
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

# ��������
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

# ����������һЩ��ά����
high_dim_data = np.random.rand(1000, 50)  # 1000��������ÿ������50ά

# ʹ��t-SNE��ά��2ά
tsne = TSNE(n_components=2, random_state=42)
low_dim_data = tsne.fit_transform(high_dim_data)

# Ӧ��K-means����
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(low_dim_data)

# ����ͼ��
plt.scatter(low_dim_data[:, 0], low_dim_data[:, 1], c=clusters, cmap='viridis')
plt.title('Cluster Visualization with t-SNE')
plt.show()


