from __future__ import print_function
import scipy.io as sio
import numpy as np
from scipy.stats import ttest_ind_from_stats

path = 'F:\Python\UCIDataset-matlab\UCI_25.mat'
mat = sio.loadmat(path)

keys = mat.keys()
keys.remove('__version__')
keys.remove('__header__')
keys.remove('__globals__')
keys.sort()
save_name = 'result.npz'
results = {}


''' 
----------------------
algorithm comparision
'''
res = np.load('F:\Python\ExtremeLearningMachine\HE_ELM\demo\\result-big_dataset_withMLP.npz')['arr_0'][()]

print ('dataset\t       ours\t      ELM\t       KELM\t      AdaELM\t        ML_ELM\t')
for key in keys:
    # print( 'processing ', key)
    if key == 'abalone' or key == 'letter' or key == 'pen' or key == 'shuttle':
        accs, time = res[key + '_acc'], res[key + '_time']
        round_acc_mean = np.round(accs.mean(axis=0) * 100, 2)
        round_acc_std = np.round(accs.std(axis=0) * 100, 2)
        round_time_mean = np.round(time.mean(axis=0), 3)
        round_time_std = np.round(time.std(axis=0), 3)

        # significance test
        our_mean, our_std = round_acc_mean[0], round_acc_std[0]
        sign_test = []
        for i in range(1, 5):
            statistic, p_value = ttest_ind_from_stats(round_acc_mean[i], round_acc_std[i], 50, our_mean, our_std, 50,
                                                      equal_var=False)
            # print(statistic, p_value)
            if p_value <= 0.05 and round_acc_mean[i] <= our_mean:
                sign_test.append('a')
            elif p_value <= 0.05 and round_acc_mean[i] > our_mean:
                sign_test.append('b')
            else:
                sign_test.append('')
    else: continue
    accs, time = res[key + '_acc'], res[key + '_time']
    round_acc_mean = np.round(accs.mean(axis=0) * 100, 2)
    round_acc_std = np.round(accs.std(axis=0) * 100, 2)
    round_time_mean = np.round(time.mean(axis=0), 3)
    round_time_std = np.round(time.std(axis=0), 3)

    # # significance test
    # our_mean, our_std = round_acc_mean[0], round_acc_std[0]
    # sign_test = []
    # for i in range(1, 4):
    #     statistic, p_value = ttest_ind_from_stats(round_acc_mean[i], round_acc_std[i], 50, our_mean, our_std, 50,
    #                                               equal_var=False)
    #     # print(statistic, p_value)
    #     if p_value <= 0.05 and round_acc_mean[i] <= our_mean:
    #         sign_test.append('a')
    #     elif p_value <= 0.05 and round_acc_mean[i] > our_mean:
    #         sign_test.append('b')
    #     else:
    #         sign_test.append('')

    print(key, round_acc_mean[0], '+-', round_acc_std[0], ' ',
          round_acc_mean[1], '+-', round_acc_std[1],  sign_test[0], ' ',
          round_acc_mean[2], '+-', round_acc_std[2],  sign_test[1], ' ',
          round_acc_mean[3], '+-', round_acc_std[3],  sign_test[2], ' ',
          round_acc_mean[4], '+-', round_acc_std[4],  sign_test[3])
          # round_acc_mean[5], '+-', round_acc_std[5], sign_test[4])

    print ('time:', round_time_mean[0], '+-', round_time_std[0], ' ',
           round_time_mean[1], '+-',  round_time_std[1], ' ',
           round_time_mean[2], '+-',  round_time_std[2],  ' ',
           round_time_mean[3], '+-',  round_time_std[3],  ' ',
           round_time_mean[4], '+-', round_time_std[4])
           # round_time_mean[5], '+-', round_time_std[5])
    print ('-----------------------------------------------------------------------------')
