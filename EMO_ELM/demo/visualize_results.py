"""
This file visualize the experimental results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
########################################
#  Project Iris data into 2-dim space
########################################
#
# path = 'F:\Python\EMO_ELM\demo\experimental_results\KSC-X_projection-50hidden-5000iter.npz'
# p = np.load(path)
# X_proj, y = p['X_proj'], p['y']
# baseline_names = ['SRP', 'PCA', 'SPCA', 'NMF', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM-AE']
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
"""
path = './experimental_results/Sparsity-dim-X_proj/SalinasA_corrected-sparsity.npz'
p = np.load(path)
sparsity = p['sparsity']

baseline_names = ['NRP', 'SPCA', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM($f_{1}$)', 'EMO-ELM($f_{2}$)', 'EMO-ELM(best)']

linestyles = ['-', '--', '-.', ':', '--', '-', '-.', ':', '-']
filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', 'D')
# linestyles = ['-', '--', '-.', ':', '--', '-', '-.',  '-']
# filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'D')
# linestyles = ['-', ':', '--', '-', '-.',  '-']
# filled_markers = ('o',  '<', '>', '8', 's', 'D')

dims = range(10, 301, 10)  # KSC=176, SalinasA=204, IndianPine=200

handles = []
for i in range(baseline_names.__len__()):
    ax, = plt.plot(dims, sparsity[:, i], linestyle=linestyles[i], label=baseline_names[i], marker=filled_markers[i])
    handles.append(ax)
plt.xlabel('Number of Features')
plt.ylabel(r'Sparsity ($L_2/L_1$)')
plt.legend(handles=handles, loc='best')
print ('Done')
"""

########################################
#  plot pareto fronts
#########################################
# from sklearn.preprocessing import minmax_scale
#
# path = 'F:\Python\EMO_ELM\demo\experimental_results\EVO_RES-SalinasA_corrected-nh=10-iter=1000-rho=0.05-n_pop=20.npz'
# dt = np.load(path)
# obj = dt['all_obj']
# ax = plt.figure()
# sub = ax.add_subplot(111)
# sub.scatter(obj[:, 0], obj[:, 1])
# plt.xlabel(r'$f_1$')
# plt.ylabel(r'$f_2$')
# print 'done'


#########################################
#  plot Box_Plot for OA, AA, Kappa
#########################################

# path = 'F:\Python\EMO_ELM\demo\experimental_results\Acc_comparison\IndianPines-Box_plot_data.npz'
# npz = np.load(path)
#
# all_aa = npz['aa']
# all_oa = npz['oa']
# all_kappa = npz['kappa']
# baseline_names = ['NRP', 'SPCA', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM($f_1$)', 'EMO-ELM($f_2$)', 'EMO-ELM(best)']
# # baseline_names = ['SRP', 'SPCA', 'NMF', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM']
# # baseline_names = ['RP', 'SPCA', 'ELM-AE', 'SELM-AE', 'SAE', 'EMO-ELM($f_1$)', 'EMO-ELM($f_2$)', 'EMO-ELM(best)']
# n_baseline = baseline_names.__len__()
#
# all_data = all_kappa.transpose()  # # TODO: edit this
# fig, axes = plt.subplots()
#
# # rectangular box plot
# bplot1 = axes.boxplot(all_data, vert=True, patch_artist=True)   # fill with color
#
# # adding horizontal grid lines
# axes.yaxis.grid(True)
# axes.set_xticks([y+1 for y in range(n_baseline)])
# # axes.set_xlabel('Algorithms')
# axes.set_ylabel('Average Accuracy (%)')  # Overall Accuracy (%) # Kappa Coefficient # Average Accuracy (%)
#
# plt.setp(axes.get_xticklabels(), rotation=15, horizontalalignment='right')
# # KSC-kappa-50-0.5-5000
#
# # add x-tick labels
# plt.setp(axes, xticks=[y+1 for y in range(n_baseline)],
#          xticklabels=baseline_names)
#
# plt.show()


'''
-------------------------------------------------------
Plot sparsity+mu effect on accuracy
'''
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
# p = 'F:\Python\EvolutionaryAlgorithm\MOEA_ELM\sparsity_level_mu_effect.npz'
# npz = np.load(p)
# keys = ['Iris', 'Wine', 'car', 'cotton', 'soybean', 'vowel', 'wdbc', 'yeast', 'zoo']
# k = 'Wine'
# test_err = npz['arr_0'][()][k + '_acc'][:, 0]
#
# sparsity_level = np.arange(0., 1.05, 0.1)
# sparsity_level[0] = 0.001
# mu = np.arange(0., 1.05, 0.1)
# mu[0] = 0.001
#
# X, Y = np.meshgrid(sparsity_level, mu,  indexing='ij')
# Z = test_err.reshape((sparsity_level.__len__(), mu.__len__()))
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # Make data.
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap='jet',
#                        linewidth=0, antialiased=False)
# # Customize the z axis.
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.gca().set_xlabel('Number of units')
# fig.gca().set_ylabel('Disconnection probability')
# ax.set_zlabel('Average testing error (%)')
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=10)
# for angle in range(0, 60):
#     ax.view_init(30, angle)
#     plt.draw()
# plt.show()


'''
------------------------------------------------------------
Plot iteration effect on accuracy for ELM....E-ELM,EMO-ELM
'''
# import numpy as np
# import matplotlib.pyplot as plt
# p = 'F:\Python\EvolutionaryAlgorithm\MOEA_ELM\iteration_effect_on_acc.npz'
# m = np.load(p)
# k = 'Iris'
# mean = m['mean'][()][k + '_acc']
# std = m['std'][()][k + '_acc']
#
# x = np.arange(0, 501, 50)
# x[0] = 1
# fig, ax = plt.subplots()
# legend = ['ELM', 'KELM', 'Ada-ELM', 'E-ELM', 'EMO-ELM']
# marker = ['*', '^', 's', 'd', 'o']
# for i in range(5):
#     e = ax.errorbar(x, mean[:, i], marker=marker[i], yerr=std[:, i], label=legend[i])
# ax.set_xlabel('Generation/Iteration')
# ax.set_ylabel('Average testing accuracy (%)')
# ax.legend(loc=1)

'''
------------------------------------------------------------
Plot sparse degree effect on accuracy for EMO-ELM
'''
# import numpy as np
# import matplotlib.pyplot as plt
# p = 'F:\Python\EvolutionaryAlgorithm\MOEA_ELM\sparse_degree_effect_on_acc.npz'
# m = np.load(p)
# k = 'Iris'
# mean = m['mean'][()][k + '_acc']
# std = m['std'][()][k + '_acc']
#
# x = np.arange(0, 1., 0.05)
# x[0] = 0.01
# fig, ax = plt.subplots()
# keys = ['Iris', 'Wine', 'wdbc', 'soybean', 'ecoli', 'diabetes']
# for k in keys:
#     e = ax.errorbar(x, m['mean'][()][k + '_acc'], yerr=m['std'][()][k + '_acc'])
# ax.set_xlabel(r'$\rho$')
# ax.set_ylabel('Average testing accuracy (%)')


'''
------------------------------------------------------------
Plot iteration effect on convergence/accuracy for EMO-ELM
'''
# root = 'F:\Python\EvolutionaryAlgorithm\MOEA_ELM\convergence_acc_pareto_front_'
# gen = [100, 500, 1000, 2000, 5000]
# p_acc = 'F:\Python\EvolutionaryAlgorithm\MOEA_ELM\iteration_effect_on_acc.npz'
# acc_npz = np.load(p_acc)
# k = 'Iris'
# fig, ax = plt.subplots()
# marker = ['*', '^', 's', 'd', 'o']
# for g in range(gen.__len__()):
#     m = np.load(root + str(gen[g]) + '.npz')
#     mean_std = str(acc_npz['mean'][()][k + '_acc'][g]) + r'$\pm$' + str(acc_npz['std'][()][k + '_acc'][g])
#     e = ax.scatter(m['obj'][:, 0], m['obj'][:, 1], marker=marker[g], label='gen=' + str(gen[g]) + ',' + mean_std)
# ax.legend(loc=1)


'''
------------------------------------------------------------
visualize the decision making for final iter
'''
"""
import matplotlib.pyplot as plt
from EMO_ELM.classes.Helper import Helper
from sklearn.preprocessing import minmax_scale
import matplotlib.patches as patches
path = 'F:\Python\EMO_ELM\demo\experimental_results\Iris_scatter\Evo-results-Iris.npz'
dt = np.load(path)
obj = dt['non_obj']
# ax = plt.figure()

norm_x, norm_y = minmax_scale(obj[:, 0]), minmax_scale(obj[:, 1])
original_index = norm_x.argsort()
xx = norm_x[original_index]
yy = norm_y[original_index]
plt.plot(norm_x, norm_y, 'o', label='Normalized PF')
# x = np.linspace(0, 2 * np.pi, 10)
# y = np.sin(x)

cur, (fx, fy), index = Helper.curvature_splines(obj[:, 0], obj[:, 1], s=0.05, k=3)
plt.plot(fx, fy, label='Interpolated PF')
plt.plot(fx, minmax_scale(cur), linestyle='--', label='Curvature of interpolated PF')

plt.plot(norm_x[index], norm_y[index], 'D', color='red')


plt.annotate('', xy=(norm_x[index[0]], norm_y[index[0]]), xytext=(norm_x[index[0]]+0.2, norm_y[index[0]]+0.2),
             bbox=dict(boxstyle="round", fc="0.8"),
             arrowprops=dict(arrowstyle="->"))  # connectionstyle="arc3"
plt.annotate('', xy=(norm_x[index[1]], norm_y[index[1]]), xytext=(norm_x[index[0]]+0.2, norm_y[index[0]]+0.2),
             bbox=dict(boxstyle="round", fc="0.8"),
             arrowprops=dict(arrowstyle="->"))  # connectionstyle="arc3"
plt.annotate('Best compromise', xy=(norm_x[index[2]], norm_y[index[2]]), xytext=(norm_x[index[0]]+0.2, norm_y[index[0]]+0.2),
             bbox=dict(boxstyle="round", fc="0.8"),
             arrowprops=dict(arrowstyle="->"))  # connectionstyle="arc3"

plt.legend(loc='best')
plt.xlabel(r'$f_1$')
plt.ylabel(r'$f_2$')
# plt.savefig('F:\Python\EMO_ELM\demo\experimental_results\Figs\PF\IndianPines-PF-5000-10-s0.01.eps', dpi=10000)
print 'Done'
"""


'''
-----------------------------------------------
visualize the decision making for final iter
-----------------------------------------------
'''
"""
path = 'F:\Python\EMO_ELM\demo\experimental_results\Iris_scatter\X_proj-Iris-nh=2-iter=5000-spar=0.05.npz'
p = np.load(path)
X_proj, y = p['X_proj'], p['y']
baseline_names = ['NRP', 'SPCA', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM($f_1$)', 'EMO-ELM($f_2$)', 'EMO-ELM(best)']
index_1 = np.nonzero(y == 0)
index_2 = np.nonzero(y == 1)
index_3 = np.nonzero(y == 2)
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

for i in range(baseline_names.__len__()):
        index = '33' + str(i + 1)
        plt.subplot(index)
        plt.scatter(X_proj[i][index_1, 0], X_proj[i][index_1, 1], marker='o', label='Setosa')
        plt.scatter(X_proj[i][index_2, 0], X_proj[i][index_2, 1], marker='^', label='Versicolour')
        plt.scatter(X_proj[i][index_3, 0], X_proj[i][index_3, 1], marker='s', label='Virginica')
        # plt.legend(loc=4)
        plt.title(baseline_names[i])
        # plt.legend(loc='best')
print('Done')
"""

'''
-----------------------------------------------
visualize dim-acc(OA, AA, kappa)
-----------------------------------------------
'''
path = './experimental_results/Acc_dim/SalinasA-dim-acc.npz'
p = np.load(path)
oa_mean, oa_std = np.asarray(p['oa'][0]), np.asarray(p['oa'][1])
aa_mean, aa_std = np.asarray(p['aa'][0]), np.asarray(p['aa'][1])
kappa_mean, kappa_std = np.asarray(p['kappa'][0]), np.asarray(p['kappa'][1])

baseline_names = ['NRP', 'SPCA', 'ELM-AE', 'SELM-AE', 'AE', 'SAE', 'EMO-ELM($f_{1}$)', 'EMO-ELM($f_{2}$)', 'EMO-ELM(best)']

linestyles = ['-', '--', '-.', ':', '--', '-', '-.', ':', '-']
filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', 'D')


dims = range(10, 301, 10)  # KSC=176, SalinasA=204, IndianPine=200
handles = []
mean, std = kappa_mean, kappa_std
for i in range(baseline_names.__len__()):
    if i == 1:
        ax = plt.errorbar(dims[:20], mean[:20, i], yerr=std[:20, i], label=baseline_names[i])
    else:
        ax = plt.errorbar(dims, mean[:, i], yerr=std[:, i], label=baseline_names[i], linestyle=linestyles[i], marker=filled_markers[i])
    handles.append(ax)
plt.xlabel('Number of Features')
plt.ylabel('Kappa Coefficient')  # Overall Accuracy (%) # Kappa Coefficient # Average Accuracy (%)
plt.legend(handles=handles, loc='best')
print ('Done')
