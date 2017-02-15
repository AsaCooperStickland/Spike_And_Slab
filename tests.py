from __future__ import division
from copy import deepcopy
import numpy as np
import pickle
from numpy import matrix
from numpy import linalg
import matplotlib.pyplot as plt
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import os

import spike_n_slab

class Testing(object):
    def __init__(self, sizes, dim, runs, gen_weights = True):
        self.sizes = sizes
        self.dim = dim
        self.multi_dim = 4
        self.runs = runs
        self.gen_weights = gen_weights
        self.X_train, self.y_train, self.X_test, self.y_test = self.get_data()
        self.number_sizes = len(self.sizes)
        self.mle_out = np.zeros(self.number_sizes)
        self.sns_out = np.zeros(self.number_sizes)
        self.gauss_out = np.zeros(self.number_sizes)
        self.mean_mle_out = np.zeros(self.number_sizes)
        self.mean_sns_out = np.zeros(self.number_sizes)
        self.mean_gauss_out = np.zeros(self.number_sizes)
        self.big_mle_out = np.zeros([self.number_sizes, runs])
        self.big_sns_out = np.zeros([self.number_sizes, runs])
        self.big_gauss_out = np.zeros([self.number_sizes, runs])

    def get_data(self, size = 3):
        
        test_size = 200
        d = self.dim

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

        self.X_train = np.zeros([tot_len, d*4])
        self.X_test = np.zeros([tot_len_test, d*4])
        done = 0
        done_test = 0
        for length, length_test, i in zip([l0, l1, l2, l3],
                [l0_test, l1_test, l2_test, l3_test], np.arange(1, 5)):
            new = done + length
            new_test = done_test + length_test
            self.X_train[done:new, (i - 1)*d:i*d] = X
            self.X_test[done_test:new_test, (i - 1)*d:i*d] = X_test
            done = new
            done_test = new_test

        y_temp1 = np.append(Y_0, Y_1)
        y_temp2 = np.append(Y_2, Y_3)
        self.y_train = np.append(y_temp1, y_temp2)

        y_temp1 = np.append(Y_0_test, Y_1_test)
        y_temp2 = np.append(Y_2_test, Y_3_test)
        self.y_test = np.append(y_temp1, y_temp2)
        return self.X_train, self.y_train, self.X_test, self.y_test

    def test_sizes(self):
        for size, i in zip(self.sizes, np.arange(0, self.number_sizes)):
            self.X_train, self.y_train, self.X_test,\
                          self.y_test = self.get_data(size)
            weights = np.zeros(self.multi_dim*self.dim)
            zs = np.zeros(self.dim)
            if self.gen_weights == True:
                sns = spike_n_slab.run_MCMC(self.X_train, self.y_train, 0.02, 0.2,
                                         weights, zs, 0.5, 4, 100.0,
                                         prior = 'spike_n_slab',
                                     save_message = 'weights' + str(size))
            sns_weights = np.genfromtxt('spike_slab_results\s_n_s_weights\weights'\
                                + str(size) + '.csv')
            if self.gen_weights == True:
                gauss = spike_n_slab.run_MCMC(self.X_train, self.y_train, 0.02,
                                              0.2, weights, zs, 0.5, 4, 100.0,
                                         prior = 'gauss',
                                     save_message = 'gauss_weights' + str(size))
            gauss_weights = np.genfromtxt(
                'spike_slab_results\s_n_s_weights\gauss_weights'\
                                + str(size) + '.csv')
            n = self.multi_dim*self.dim
            sns_mean = np.zeros(n)
            gauss_mean = np.zeros(n)
            for j in range(n):
                sns_mean[j] = np.mean(sns_weights[1000:, j])
                gauss_mean[j] = np.mean(gauss_weights[1000:, j])
            mle_weights = MLE(self.X_train, self.y_train)
            y_sns, y_gauss, y_mle = predict(sns_mean, gauss_mean, mle_weights,
                                self.X_test, len(self.y_test))
            sns_err = np.mean((y_sns - self.y_test)**2)
            gauss_err = np.mean((y_gauss - self.y_test)**2)
            mle_err = np.mean((y_mle - self.y_test)**2)
            self.mle_out[i] = mle_err
            self.sns_out[i] = sns_err
            self.gauss_out[i] = gauss_err

    @staticmethod
    def make_mean(big_arr):
        mean_arr = np.log(big_arr)
        std = np.std(mean_arr, axis = 1)
        mean_arr = np.mean(mean_arr, axis = 1)
        mean = np.mean(mean_arr)
        mean_arr = mean_arr - mean
        return mean_arr, mean, std

    def test_runs(self):
        for i in range(self.runs):
            self.test_sizes()
            self.big_mle_out[:, i] = self.mle_out
            self.big_sns_out[:, i] = self.sns_out
            self.big_gauss_out[:, i] = self.gauss_out

        self.mean_mle_out, mle_mean, mle_std = self.make_mean(self.big_mle_out)
        self.mean_sns_out, sns_mean, sns_std = self.make_mean(self.big_sns_out)
        self.mean_gauss_out, gauss_mean, gauss_std = self.make_mean(self.big_gauss_out)
 
        return mle_std, sns_std, gauss_std, mle_mean, sns_mean, gauss_mean

def MLE(X, y):
    Theta = np.matrix(X)
    Theta_t = np.matrix.transpose(Theta)
    inv = np.linalg.inv(Theta_t*Theta)
    return inv*Theta_t*matrix.transpose(matrix(y))

'''

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
'''
def predict(sns_weights, gauss_weights, max_like_est, X, size):
    y_sns = np.zeros(size)
    y_gauss = np.zeros(size)
    y_mle = np.zeros(size)
    for i in range(size):
            y_sns[i] = np.sum(sns_weights*X[i, :])
            y_gauss[i] = np.sum(gauss_weights*X[i, :])
            for j in range(X.shape[1]):
                y_mle[i] = y_mle[i] + float(max_like_est[j])*X[i, j]              
    return y_sns, y_gauss, y_mle

def fit_GP(xs, ys, std):
    X = np.zeros([len(xs), 2])
    X[:, 0] = 1.0
    X[:, 1] = xs
    gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha = std,
            n_restarts_optimizer=25,
        )
    gp.fit(X, ys)
    x = np.arange(3, 20, 0.005)
    X_pred = np.zeros([len(x), 2])
    X_pred[:, 0] = 1.0
    X_pred[:, 1] = x
    y, std = gp.predict(X_pred, return_std=True)
    y_plus = y + std
    y_minus = y - std
    return y, y_plus, y_minus, x

new_data = False

if new_data == True:
    versus = Testing(sizes = [3, 4, 5, 6, 8, 10, 12, 16, 20], dim = 5, runs = 5)
    mle_std, sns_std, \
             gauss_std, mle_mean, sns_mean, gauss_mean = versus.test_runs()
    errors = [mle_std, sns_std, gauss_std, mle_mean, sns_mean, gauss_mean, versus]
    pickle.dump(errors, open('comparison.pickle', 'wb'))
else:
    load_name = 'comparison.pickle'
    errors = pickle.load( open( load_name, "rb" ) )
    mle_std, sns_std, gauss_std, mle_mean, sns_mean, gauss_mean, versus = errors

mle_out = versus.mean_mle_out
sns_out = versus.mean_sns_out
gauss_out = versus.mean_gauss_out
xs = versus.sizes

y, y_plus, y_minus, x = fit_GP(xs, mle_out, mle_std)
y2, y2_plus, y2_minus, x = fit_GP(xs, sns_out, sns_std)
y3, y3_plus, y3_minus, x = fit_GP(xs, gauss_out, gauss_std)

xs = np.array(xs)
x = np.array(x)
xs *= 4
x *= 4

plt.figure()
plt.scatter(xs, mle_out + mle_mean, s= 30, alpha=0.3,
                    edgecolor='black', facecolor='b', linewidth=0.75)
plt.errorbar(xs, mle_out + mle_mean, mle_std, fmt='b.', markersize=16,
             alpha=0.5, label = 'Maximum likelihood')
plt.scatter(xs, sns_out + sns_mean, s= 30, alpha=0.3,
                    edgecolor='black', facecolor='r', linewidth=0.75 )
plt.errorbar(xs, sns_out + sns_mean, sns_std, fmt='r.', markersize=16,
             alpha=0.5, label = 'Spike and slab prior')
#plt.plot(x, y + mle_mean)
#plt.fill_between(x, y_plus + mle_mean, y_minus + mle_mean, alpha = 0.3)
#plt.plot(x, y2 + sns_mean, color = 'red')
#plt.fill_between(x, y2_plus + sns_mean, y2_minus + sns_mean,
#                 alpha = 0.3, color = 'red')
plt.errorbar(xs, gauss_out + gauss_mean, gauss_std, fmt='g.',
             markersize=16, alpha=0.8, label = 'Gaussian prior')
#plt.plot(x, y3 + gauss_mean, color = 'green')
#plt.fill_between(x, y3_plus + gauss_mean, y3_minus + gauss_mean,
#                 alpha = 0.3, color = 'green')
plt.plot([0, 100], [np.log(0.0001), np.log(0.0001)], 'v--', linewidth = 3.0,
         alpha = 0.6)
plt.legend(loc='upper right', shadow=True)
plt.xlabel('$\mathrm{Number}$' + ' ' + '$\mathrm{of}$' + ' ' + '$x$'+ ' ' +
           '$\mathrm{values}$', fontsize = 20)
plt.ylabel('$\mathrm{log(Error)}$', fontsize = 20)
plt.xlim(10.0, 82.0)
plt.ylim(-11, 25)
plt.tight_layout()
plt.savefig('gps.png')
plt.show()

plt.figure()
plt.scatter(xs, mle_out + mle_mean, s= 30, alpha=0.3,
                    edgecolor='black', facecolor='b', linewidth=0.75)
plt.errorbar(xs, mle_out + mle_mean, mle_std, fmt='b.', markersize=16,
             alpha=0.5, label = 'Maximum likelihood')
plt.scatter(xs, sns_out + sns_mean, s= 30, alpha=0.3,
                    edgecolor='black', facecolor='r', linewidth=0.75 )
plt.errorbar(xs, sns_out + sns_mean, sns_std, fmt='r.', markersize=16,
             alpha=0.5, label = 'Spike and slab prior')
plt.plot(x, y + mle_mean)
plt.fill_between(x, y_plus + mle_mean, y_minus + mle_mean, alpha = 0.3)
plt.plot(x, y2 + sns_mean, color = 'red')
plt.fill_between(x, y2_plus + sns_mean, y2_minus + sns_mean,
                 alpha = 0.3, color = 'red')
plt.plot([0, 100], [np.log(0.0001), np.log(0.0001)], 'v--', linewidth = 3.0,
         alpha = 0.6)
plt.legend(loc='upper right', shadow=True)
plt.xlabel('$\mathrm{Number}$' + ' ' + '$\mathrm{of}$' + ' ' + '$x$'+ ' ' +
           '$\mathrm{values}$', fontsize = 20)
plt.ylabel('$\mathrm{log(Error)}$', fontsize = 20)
plt.xlim(10.0, 82.0)
plt.ylim(-11, 8)
plt.tight_layout()
plt.savefig('gps2.png')
plt.show()

p0 = np.genfromtxt('spike_slab_results\s_n_s_p0\weights4.csv')

plt.figure()
plt.hist(p0[1000:], bins = 50, histtype='stepfilled', normed=True,
            color='b', alpha = 0.7)
plt.ylabel('$\mathrm{Frequency}$', fontsize = 20)
plt.xlabel('$p_0$', fontsize = 20)
plt.tight_layout()
plt.show()

sig2 = np.genfromtxt('spike_slab_results\s_n_s_sigma2\weights4.csv')

plt.figure()
plt.hist(sig2[1000:], bins = 50, histtype='stepfilled', normed=True,
            color='b', alpha = 0.7)
plt.ylabel('$\mathrm{Frequency}$', fontsize = 20)
plt.xlabel('$\sigma^2$', fontsize = 20)
plt.tight_layout()
plt.show()
    
'''
y_pred, y_mle = predict(ws_mean, MLE(X_train, y_train), X_test, len(y_test))
pred_err = np.mean((y_pred - y_test)**2)
mle_err = np.mean((y_mle - y_test)**2)
print("error in spike and slab model is {}\
      and for the MLE is {}".format(pred_err, mle_err))'''
    
'''
weights = np.zeros(4*5)
zs = np.zeros(5)
test = spike_n_slab.run_MCMC(X_train, y_train, 0.2, 0.2, weights, zs, 0.5, 4,
                        100.0, prior = 'spike_n_slab', save_message = 'gest')

'''

