"""
This file visualize the experimental results.
"""
import numpy as np
import matplotlib.pyplot as plt

'''
----------------------
Project Iris data into 2-dim space
'''
path = 'F:\Python\EMO_ELM\demo\experimental_results\X_projection.npz'
p = np.load(path)
X_proj, y = p['X_proj'], p['y']
baseline_names = ['RP', 'PCA', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM-AE']
index_1 = np.nonzero(y == 0)
index_2 = np.nonzero(y == 1)
index_3 = np.nonzero(y == 2)
for i in range(baseline_names.__len__()):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    s1 = ax.scatter(X_proj[i][index_1, 0], X_proj[i][index_1, 1], marker='o')
    s2 = ax.scatter(X_proj[i][index_2, 0], X_proj[i][index_2, 1], marker='^')
    s3 = ax.scatter(X_proj[i][index_3, 0], X_proj[i][index_3, 1], marker='s')
    ax.set_title(baseline_names[i])
    ax.legend((s1, s2, s3), ('Setosa', 'Versicolour', 'Virginica'), loc=4)

print('OK')