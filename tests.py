from __future__ import division
from copy import deepcopy
import numpy as np
import pickle
from numpy import matrix
from numpy import linalg
import matplotlib.pyplot as pyplot
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
label_size = 19
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
from numpy import genfromtxt
import scipy.stats
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
from scipy.stats import invgamma
from scipy.stats import beta
from datetime import datetime
import os

import spike_n_slab

def MLE(X, y):
    Theta = np.matrix(X)
    Theta_t = np.matrix.transpose(Theta)
    inv = np.linalg.inv(Theta_t*Theta)
    return inv*Theta_t*matrix.transpose(matrix(y))

def get_data():

    multi_dim = 4
    size = 3
    test_size = 200
    d = 5

    X = np.zeros([size, d])
    X_test = np.zeros([test_size, d])

    for i in range(d):
        X[:, i] = np.random.uniform(size = size)
        X_test[:, i] = np.random.uniform(size = test_size)

    for i in range(d):
        if np.std(X[:, i]) !=0:
            X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])
            X_test[:, i] = (X_test[:, i] -
                            np.mean(X_test[:, i]))/np.std(X_test[:, i])
            
    Y_0 = (0.1*X[:, 1] + 0.1*X[:, 4] +
           np.random.normal(scale = 0.01, size = size))
    Y_1 = (0.3*X[:, 1] - 0.1*X[:, 4] +
           np.random.normal(scale = 0.01, size = size))
    Y_2 = (-0.5*X[:, 1] - 0.5*X[:, 4] +
           np.random.normal(scale = 0.01, size = size))
    Y_3 = (0.2*X[:, 1]  - 0.6*X[:, 4] +
           np.random.normal(scale = 0.01, size = size))

    Y_0_test = (0.1*X_test[:, 1] + 0.1*X_test[:, 4] +
           np.random.normal(scale = 0.01, size = test_size))
    Y_1_test = (0.3*X_test[:, 1] - 0.1*X_test[:, 4] +
           np.random.normal(scale = 0.01, size = test_size))
    Y_2_test = (-0.5*X_test[:, 1] - 0.5*X_test[:, 4] +
           np.random.normal(scale = 0.01, size = test_size))
    Y_3_test = (0.2*X_test[:, 1]  - 0.6*X_test[:, 4] +
           np.random.normal(scale = 0.01, size = test_size))


    tot_len = len(Y_0) + len(Y_1) + len(Y_2) + len(Y_3)
    l0 = len(Y_0)
    l1 = len(Y_1)
    l2 = len(Y_2)
    l3 = len(Y_3)

    tot_len_test = len(Y_0_test) + len(Y_1_test) + len(Y_2_test) + len(Y_3_test)
    l0_test = len(Y_0_test)
    l1_test = len(Y_1_test)
    l2_test = len(Y_2_test)
    l3_test = len(Y_3_test)
    
    X_big = np.zeros([tot_len, d*4])
    X_big_test = np.zeros([tot_len_test, d*4])
    done = 0
    done_test = 0
    for length, length_test, i in zip([l0, l1, l2, l3],
                [l0_test, l1_test, l2_test, l3_test], np.arange(1, 5)):
        new = done + length
        new_test = done_test + length_test
        X_big[done:new, (i - 1)*d:i*d] = X
        X_big_test[done_test:new_test, (i - 1)*d:i*d] = X_test
        done = new
        done_test = new_test

    y_temp1 = np.append(Y_0, Y_1)
    y_temp2 = np.append(Y_2, Y_3)
    y_big = np.append(y_temp1, y_temp2)

    y_temp1 = np.append(Y_0_test, Y_1_test)
    y_temp2 = np.append(Y_2_test, Y_3_test)
    y_big_test = np.append(y_temp1, y_temp2)

    return y_big, X_big, y_big_test, X_big_test

try:
    load_name = 'test_data.pickle'
    data = pickle.load( open( load_name, "rb" ) )
    y_train, X_train, y_test, X_test = data
except:
    y_train, X_train, y_test, X_test = get_data()
    data_CV = [y_train, X_train, y_test, X_test]
    pickle.dump(data_CV, open('test_data.pickle', 'wb'))
    
generate_data = False
generate_weights = False
if generate_data == True:
    y_train, X_train, y_test, X_test = get_data()
if generate_weights == True:
    weights = np.zeros(4*5)
    zs = np.zeros(5)
    test = spike_n_slab.run_MCMC(X_train, y_train, 0.2, 0.2, weights, zs,
                0.5, 4, 100.0, prior = 'spike_n_slab', save_message = 'gest')

weights = np.genfromtxt('spike_slab_results\s_n_s_weights\gest.csv')
zs = np.genfromtxt('spike_slab_results\s_n_s_zs\gest.csv')

n = 5
ws_mean = np.zeros(4*n)
zs_mean = np.zeros(n)
zs_std = np.zeros(n)
ws_std = np.zeros(4*n)
for i in range(n):
    zs_mean[i] = np.mean(zs[1000:, i])
    zs_std[i] = np.std(zs[1000:, i])
for i in range(4*n):
    ws_mean[i] = np.mean(weights[1000:, i])
    ws_std[i] = np.std(weights[1000:, i])

def predict(weights, max_like_est, X, size):
    y_pred = np.zeros(size)
    y_mle = np.zeros(size)
    for i in range(size):
            y_pred[i] = np.sum(ws_mean*X[i, :])
            for j in range(X.shape[1]):
                y_mle[i] = y_mle[i] + float(max_like_est[j])*X[i, j]              
    return y_pred, y_mle

y_pred, y_mle = predict(weights, MLE(X_train, y_train), X_test, len(y_test))
pred_err = np.mean((y_pred - y_test)**2)
mle_err = np.mean((y_mle - y_test)**2)
print("error in spike and slab model is {}\
      and for the MLE is {}".format(pred_err, mle_err))
    
'''
weights = np.zeros(4*5)
zs = np.zeros(5)
test = spike_n_slab.run_MCMC(X_train, y_train, 0.2, 0.2, weights, zs, 0.5, 4,
                        100.0, prior = 'spike_n_slab', save_message = 'gest')

'''

