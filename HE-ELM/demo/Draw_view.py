
import matplotlib.pyplot as plt
import numpy as np

# '''
# ------------------------------------
# Draw ours train/test error with range of ELM units
# '''
# p = 'F:\Python\ExtremeLearningMachine\MyELM\\n_unit_test_100_hidden.npz'
# npz = np.load(p)
# keys = ['Iris', 'Wine', 'car', 'cotton','soybean', 'vowel', 'wdbc', 'yeast', 'zoo']
# k = 'Iris'
# train_err = npz['arr_0'][()][k + '_train_err_ours']
# test_err = npz['arr_0'][()][k + '_test_err_ours']
# train_err_ad = npz['arr_0'][()][k + '_train_err_adaboost']
# test_err_ad = npz['arr_0'][()][k + '_test_err_adaboost']
#
# x = range(5, 101, 5)
# x.insert(0, 1)
# fig, ax = plt.subplots()
# e1 = ax.errorbar(x, train_err[:, 0], marker='o', yerr=train_err[:, 1], label='train error(Ours)')
# e2 = ax.errorbar(x, test_err[:, 0], marker='s', yerr=test_err[:, 1], label='test error(Ours)')
# e3 = ax.errorbar(x, train_err_ad[:, 0], marker='o', linestyle='--', yerr=train_err_ad[:, 1], label='train error(Adaboost-ELM)')
# e4 = ax.errorbar(x, test_err_ad[:, 0], marker='s', linestyle='--', yerr=test_err_ad[:, 1], label='test error(Adaboost-ELM)')
# ax.set_xlabel('Number of ELM units')
# ax.set_ylabel('Error Rate (%)')
# ax.legend(loc=1)
# plt.show()



'''
------------------------------------
Draw ours/base/adaboost error with range of hidden nodes
'''
# p = 'F:\Python\ExtremeLearningMachine\MyELM\\n_hidden_comparision_20_units.npz'
# npz = np.load(p)
# keys = ['Iris', 'Wine', 'car', 'cotton', 'soybean', 'vowel', 'wdbc', 'yeast', 'zoo']
# k = 'zoo'
# ours = npz['arr_0'][()][k + '_ours_err']
# base = npz['arr_0'][()][k + '_base_err']
# ada = npz['arr_0'][()][k + '_adboost_err']
#
# x = range(10, 201, 10)
# x.insert(0, 5)
# fig, ax = plt.subplots()
# e1 = ax.errorbar(x, ours[:, 0], marker='o', yerr=ours[:, 1], label='Ours')
# e2 = ax.errorbar(x, base[:, 0], marker='^', yerr=base[:, 1], label='ELM')
# e3 = ax.errorbar(x, ada[:, 0], marker='s', yerr=ada[:, 1], label='Adaboost-ELM')
#
# ax.set_xlabel('Number of hidden neurons')
# ax.set_ylabel('Average testing error (%)')
# ax.legend(loc=1)
# plt.show()


'''
------------------------------------
Draw ours/base/adaboost/KEML error with number of training samples
'''
# p = 'F:\Python\ExtremeLearningMachine\MyELM\\n_training_samples_comparision_50_hidden.npz'
# npz = np.load(p)
# keys = ['Iris', 'car', 'cotton','Wine',  'vowel', 'wdbc']#['vowel', 'wdbc', 'yeast', 'zoo', 'Wine', 'car', 'cotton','Iris','soybean']
# k = 'wdbc'
# ours = npz['arr_0'][()][k + '_ours_err']
# base = npz['arr_0'][()][k + '_base_err']
# ada = npz['arr_0'][()][k + '_adboost_err']
# keml = npz['arr_0'][()][k + '_kelm_err']
#
# x = np.arange(0.1, 1.1, 0.1)
# fig, ax = plt.subplots()
# e1 = ax.errorbar(x, ours[:, 0], marker='o', yerr=ours[:, 1], label='Ours')
# e2 = ax.errorbar(x, base[:, 0], marker='^', yerr=base[:, 1], label='ELM')
# e3 = ax.errorbar(x, ada[:, 0], marker='s', yerr=ada[:, 1], label='Adaboost-ELM')
# e4 = ax.errorbar(x, keml[:, 0], marker='d', yerr=keml[:, 1], label='KELM')
#
# ax.set_xlabel('Number of labeled samples')
# ax.set_ylabel('Average testing error (%)')
# ax.legend(loc=1)
# plt.show()


'''
# ------------------------------------
# Draw ours test error with range of ELM units+droprate
# '''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
p = 'F:\Python\ExtremeLearningMachine\MyELM\\n_unit_droprate_test_error_xxx.npz'
npz = np.load(p)
keys = ['Iris', 'Wine', 'car', 'cotton', 'soybean', 'vowel', 'wdbc', 'yeast', 'zoo']
k = 'Wine'
test_err = npz['arr_0'][()][k + '_test_err_ours'][:, 0]
n_units = range(5, 101, 5)
n_units.insert(0, 1)
drop_rate = np.arange(0., 1, 0.05)

X, Y = np.meshgrid(n_units, drop_rate,  indexing='ij')
Z = test_err.reshape((n_units.__len__(), drop_rate.__len__()))
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='jet',
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.gca().set_xlabel('Number of units')
fig.gca().set_ylabel('Disconnection probability')
ax.set_zlabel('Average testing error (%)')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
for angle in range(0, 60):
    ax.view_init(30, angle)
    plt.draw()
plt.show()
