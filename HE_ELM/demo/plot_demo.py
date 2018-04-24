"""
Add resulting results are saved in OneNote at page of Multi-ELM
"""
'''
--------------------
influence of n_learner 
'''
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

data = np.asarray()
data = np.round(data * 100, 2)
range_ = range(5, 101, 10)
markers = ['-o', '-s', '-d', '-*', '-D']
label = ['HE-ELM', 'V-ELM', 'B-ELM', 'Ada-ELM', 'H-ELM-E']
for i in range(5):
    plt.plot(range_, data[:, i], markers[i], label=label[i])
    # plt.errorbar(range_, data[:, i], yerr=err[:, i], fmt=markers[i], label=label[i])
plt.xlabel('Number of component ELMs')
plt.ylabel('Average testing accuracy (%)')
plt.legend()
plt.grid(True)


"""
------------------------
influence of prob/n_hidden
"""
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

Z = np.asarray()
Z = np.round(Z * 100, 2)
X = range(20, 201, 20)
Y = np.arange(0.1, 1., 0.1)[::-1]
X, Y = np.meshgrid(X, Y, indexing='ij')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet',
                       linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=10)
fig.gca().set_xlabel('Number of neurons')
fig.gca().set_ylabel(r'$P$')
ax.set_zlabel('Average testing accuracy (%)')
# Add a color bar which maps values to colors.
for angle in range(0, 45):
    ax.view_init(30, angle)
    plt.draw()



"""
------------------------
influence of n_rsm bar
"""
import numpy as np
import matplotlib.pyplot as plt
import math

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.*height,
                '%s' % height,
                ha='center', va='bottom')

mean = np.asarray()
mean = np.round(mean * 100, 2)
n_groups = len(mean)
index = np.arange(n_groups)
bar_width = 0.5
opacity = 0.5
# rects1 = plt.bar(index, mean, bar_width,  yerr=std, alpha=opacity, capsize=1)
low = min(mean)
high = max(mean)
fig, ax = plt.subplots()

plt.ylim([low-0.5*(high-low), high+0.2*(high-low)])
rects1 = ax.bar(index, mean, bar_width, capsize=1)  #yerr=std,
plt.xlabel('Number of RSMs')
plt.ylabel('Average testing accuracy (%)')
plt.xticks(index, ('5', '10', '15', '20', '25', '30'))
# plt.legend()
plt.tight_layout()
plt.show()
plt.grid(axis=0)
autolabel(rects1)

'''
---------------------
Acc. table processing
'''
str_ = """96.75\pm2.38 
79.44\pm2.15 
98.54\pm1.20 
97.34\pm0.96 
96.62\pm1.26 
92.90\pm1.70 
97.59\pm0.82 
77.40\pm1.60 
88.40\pm1.93
92.32\pm3.84 
96.42\pm0.21 
82.81\pm2.96 
98.95\pm0.07 
99.50\pm0.09 
89.96\pm0.37 
97.08\pm0.58 
99.73\pm0.02
94.22\pm0.90
95.56\pm1.20
97.83\pm0.97
59.17\pm1.35
100.00\pm0.00"""
l1 = str_.split('\n')
acc = []
for x_i in l1:
    acc_, std = x_i.split('\pm')
    acc.append([float(acc_), float(std)])
print('shape=', np.asarray(acc).shape)
print(np.round(np.asarray(acc).mean(axis=0), 2))
k = ['SVM',	'ELM',	'MLP',	'ML-ELM',	'V-ELM',	'B-ELM',	'Ada-ELM',	'HE-ELM(L1)',	'HE-ELM']

from scipy.stats import ttest_ind_from_stats
t = np.load('F:\Python\HE_ELM\demo\Table.npz')['acc']
dt = ['Iris ','SPECTF ','Wine','Car','Chart','Cotton','Dermatology','Diabetes','Ecoli','Glass','Letter','Libras',
 'Optdigits','Pen','Satellite','Segment','shuttle', 'Soybean','Vowel','WDBC', 'Yeast','Zoo']
for i in range(8):
    _, p = ttest_ind_from_stats(t[i, :, 0], t[i, :, 1], 50, t[8, :, 0], t[8, :, 1], 50)
    pp = p <= 0.05
    z_1 = t[8, :, 0] >= t[i, :, 0]
    z_1 = z_1 * 1
    z_2 = t[8, :, 0] < t[i, :, 0]
    # z_2 = z_2 * -1
    res_1 = pp * z_1 * 1
    res_2 = pp * z_2 * -1
    res = res_1 + res_2
    print '%s/%s/%s' % (int(np.nonzero(res == 1)[0].shape[0]), int(np.nonzero(res == 0)[0].shape[0]), int(np.nonzero(res == -1)[0].shape[0]))

    # print zip(dt, res_1 + res_2)
    print('================================')

np.savetxt()