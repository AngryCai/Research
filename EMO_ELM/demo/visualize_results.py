"""
This file visualize the experimental results.
"""
import numpy as np
import matplotlib.pyplot as plt

########################################
#  Project Iris data into 2-dim space
########################################

# path = 'F:\Python\EMO_ELM\demo\experimental_results\X_projection.npz'
# p = np.load(path)
# X_proj, y = p['X_proj'], p['y']
# baseline_names = ['RP', 'PCA', 'SPCA', 'NMF', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM-AE']
# index_1 = np.nonzero(y == 0)
# index_2 = np.nonzero(y == 1)
# index_3 = np.nonzero(y == 2)
# for i in range(baseline_names.__len__()):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     s1 = ax.scatter(X_proj[i][index_1, 0], X_proj[i][index_1, 1], marker='o')
#     s2 = ax.scatter(X_proj[i][index_2, 0], X_proj[i][index_2, 1], marker='^')
#     s3 = ax.scatter(X_proj[i][index_3, 0], X_proj[i][index_3, 1], marker='s')
#     # ax.set_title(baseline_names[i])
#     ax.legend((s1, s2, s3), ('Setosa', 'Versicolour', 'Virginica'), loc=1)
#     plt.savefig('./experimental_results/Figures/' + baseline_names[i] + '.eps', format='eps', dpi=1000)
# print('Done')


########################################
#  plot sparsity for different dimension
########################################

path = './experimental_results/sparsity.npz'
p = np.load(path)
sparsity = p['sparsity']
baseline_names = ['SRP', 'PCA', 'SPCA', 'NMF', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM-AE']
# baseline_names = ['ELM-AE', 'SELM-AE', 'EMO-ELM-AE']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', 'D')

dims = range(5, 101, 5)
handles = []
for i in range(baseline_names.__len__()):
    ax, = plt.plot(dims, sparsity[:, i], linestyle=linestyles[i], lw=2,
                   label=baseline_names[i], marker=filled_markers[i], markersize=5)
    handles.append(ax)
plt.xlabel('Number of Features')
plt.ylabel(r'Sparsity ($L_2/L_1$)')
plt.legend(handles=handles, loc=1)
print ('Done')



#########################################
#  plot pareto fronts
########################################
# path = './experimental_results/EMO_results.npz'
# dt = np.load(path)
# obj = dt['obj']
# ax = plt.figure()
# sub = ax.add_subplot(111)
# sub.scatter(obj[:, 0], obj[:, 1])
# print 'done'
