mfrom __future__ import division
from copy import deepcopy
import numpy as np
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
                           
def find_cens(vec, cens_value):
    
    """
    Store the positions in a vector (vec) whose values are larger than
    'cens_value'
    """
    
    cens = np.array([], dtype = 'int')
    for i in range(len(vec)):
        if vec[i] >= cens_value:
            cens = np.append(cens, i)      
    return cens

def find_nan(vec):
    
    """
    Store the positions in a vector (vec) whose values are missing, and
    the ones that aren't in a another array
    """
    
    nans = np.array([], dtype = 'int')
    not_nans = np.array([], dtype = 'int')
    for i in range(len(vec)):
        if vec[i] != vec[i]:
            nans = np.append(nans, i)
        else:
            not_nans = np.append(not_nans, i)
            
    return nans, not_nans

def find_zero(vec):
    
    """
    Store the positions in a vector (vec) whose values are zero, and the
    ones that aren't in a another array
    """
    
    zeros = np.array([], dtype = 'int')
    not_zeros = np.array([], dtype = 'int')
    for i in range(len(vec)):
        if vec[i] == 0.0:
            zeros = np.append(zeros, i)
        else:
            not_zeros = np.append(not_zeros, i)       
    return zeros, not_zeros

def multi_truncnorm(mean, sigma2, cens_value):
    
    """
    Returns vector of truncnormal samples, but they can't be correlated
    """
    
    samples = np.zeros(len(mean))
    for i in range(len(mean)):
        samples[i] = truncnorm.rvs((cens_value - mean[i])/sigma2**0.5,
                                   200.0, loc = mean[i], scale = sigma2**0.5)
    return samples

class Spike_N_Slab(object):

    
    """
    Parameters of a linear regression model with spike-and-slab priors modified
    so that a group of variables is different to zero, rather than just one.
    See appendix of https://link.springer.com/article/10.1007/s10994-014-5475-7
    for more details- I use parameter names based on what they use.
    """

    
    def __init__(self, data, y, sigma2, vs, weights, zs, p0, multi_dim,
                 cens_value, prior = 'spike_n_slab'):

        
        """
        Allocate initial paramters of the model and input data
        
        :type data: numpy array of dimensions #data points by #features, with
        float entries
        :param data: independent variables of each data point

        :type y: numpy array of length #data points, with float entries
        :param y: dependent variable of each data point

        :type size: float
        :param size: #data points 

        :type sigma2: float
        :param sigma2: variability of y values not caused by dependence on
        features

        :type vs: float
        :param vs: variance of 'slab' part of spike-and-slab prior

        :type weights: numpy array of length #features (which sum of #features
        of each group, with float entries
        :param weights: weights given to each feature

        :type zs: numpy array of length #groups, with float entries
        :param zs: array with value 1.0 if that group of features follows
        slab prior or 0.0 if that group of
        feature follows spike prior

        :type g: float
        :param zs: # features in each group (for me it's the same in every
        group)

        type p0_hat: float
        :param p0_hat: a priori probability that a given weight has value zero

        type p0: float
        :param p0: probability that a given weight has value zero
        """

        
        self.data = data
        self.y = y
        self.size = len(y)
        self.sigma2 = sigma2
        self.vs = vs
        self.w = weights
        self.zs = zs
        self.g = len(zs)
        self.p0_hat = p0
        self.p0 = p0
        self.prior = prior
        if prior == 'gauss':
            self.zs = np.ones(self.g)
            self.p0 = 1.0
        self.multi_dim = multi_dim
        self.d = len(weights)
        self.zeros, self.ones = find_zero(zs)
        self.cens_value = cens_value
        self.cens = find_cens(self.y, cens_value)
        if prior == 'gauss':
            self.zeros, self.ones = find_zero(self.zs)
        self.big_zeros = np.zeros(len(self.zeros)*self.multi_dim, dtype = int)
        self.big_ones = np.zeros(len(self.ones)*self.multi_dim, dtype = int)
        self.fill_in_binary()  
        self.mean, self.Eta_z = self.regcalc()
        

    def __validate_params(self):

        """
        Check parameters are in allowed ranges.
        """

        if self.size != self.data.shape[0]:
            raise ValueError("x and y dimensions don't match, got y dim {}"\
                             "and x dim .{}".format(self.size, self.data.shape[0]))
        if self.d != self.data.shape[1]:
            raise ValueError("number of weights and x dimensions don't match,"\
                             "got weight dim {} and x dim .{}".format(self.d, self.data.shape[1]))
        if self.g*self.multi_dim != self.data.shape[1]:
            raise ValueError("number of zs times number of target values and"\
                             "x dimensions don't match, got z dim times target"\
                             "dim {} and x dim {}.".format(self.g*self.multi_dim, self.data.shape[1]))    
        if self.sigma2 < 0:
            raise ValueError("sigma^2 must be > 0, got {}.".format(self.sigma2))
        if self.vs < 0:
            raise ValueError("vs must be > 0, got {}.".format(self.vs))
        if self.p0_hat < 0 or self.p0_hat > 1.0:
            raise ValueError("p0 must be a probability, i.e. between 0.0 and"\
                             "1.0, got {}.".format(self.p0_hat))

    def fill_in_binary(self):

        """
        Update big list of whether a weight is zero or one.
        """

        l_zeros = len(self.zeros)
        l_ones = len(self.ones)
        for j in range(self.multi_dim):
            self.big_zeros[j*l_zeros:(j+1)*l_zeros] = self.zeros + j*self.g
            self.big_ones[j*l_ones :(j+1)*l_ones ] = self.ones + j*self.g

        
    def regcalc(self):
        
        """
        Calculates the estimates for mean and covariance matrix of weights
        given a particular z-vector.
        """
        
        Theta = np.matrix(self.data)
        Theta_t = np.matrix.transpose(Theta)
        mult = np.zeros(self.d)
        mult = mult + 1.0/self.vs
        mult[self.big_zeros] = 10000000.0
        I_vs = mult*np.identity(self.d)
        Eta_z = np.linalg.inv((1.0/self.sigma2)*Theta_t*Theta + I_vs)
        ys = matrix.transpose(matrix(self.y))
        prelim = Theta_t * ys
        M_N = (1/self.sigma2) * Eta_z * prelim

        return M_N, Eta_z
        
    def sample_z_i(self, i):
        
        """
        Samples from distribution of i'th value of z-vector given other
        parameters

        :type i: int
        :param i: number which indicates which feature's z value to sample 
        """
        
        if self.zs[i] == 1.0:
            # This is all on page 48 (Appendix A) of group feature selection
            # paper - pretty hard to explain without looking at that. 
            Theta = matrix(self.data)
            Theta_t = matrix.transpose(Theta)
            mult = np.zeros(self.d)
            mult = mult + self.vs
            mult[self.big_zeros] = 0.0
            for j in range(self.multi_dim):
                mult[i + j*self.g] = 0.0
                
            A_minusg = mult*np.identity(self.d)
            C_minusg = (self.sigma2*np.identity(self.size) +
                        Theta*A_minusg*Theta_t)
            X_g = np.zeros([self.size, self.multi_dim])
            for k in range(self.multi_dim):
                X_g[:, k] = self.data[:, i + k*self.g]
            X_g = matrix(X_g)
            X_g_t = matrix.transpose(X_g)
            C_inv = np.linalg.inv(C_minusg)
            M = X_g_t*C_inv*X_g
            eig_vals, eig_vecs = np.linalg.eigh(M)
            y = matrix(self.y)
            L = 0.0
            for j in range(self.multi_dim):
                s_j = eig_vals[j]
                e_j = eig_vecs[:, j]
                temp = y*C_inv*X_g
                q_j = float(temp*e_j) 
                L = (L + 0.5*(q_j**2.0/((1.0/self.vs) + s_j)
                              - np.log(1.0 + self.vs*s_j)))
            L_tot = np.exp(L)
        else:
            L_tot = 1.0
        test = np.random.uniform()
        
        if (test < (self.p0*L_tot)/(self.p0*L_tot + (1.0 - self.p0))
            or L_tot > 1000000000000.0):
            self.zs[i] = 1.0
        else:
            self.zs[i] = 0.0
        
        self.zeros, self.ones = find_zero(self.zs)
        self.big_zeros = np.zeros(len(self.zeros)*self.multi_dim, dtype = int)
        self.big_ones = np.zeros(len(self.ones)*self.multi_dim, dtype = int)
        self.fill_in_binary() 
 
        return

    def sample_p0(self, k):
        
        """ 
        Samples from distribution of p_0 given other parameters.

        :type k: float
        :param k: variance of prior - lower = broader
        """
        
        s_z = float(len(self.ones))
        g = float(self.g)
        self.p0 = beta.rvs(k*self.p0_hat + s_z, k*(1 - self.p0_hat) + g - s_z)
        return
    
    def sample_sigma2(self, k, sigma2_hat):
        
        """
        Samples from distribution of the error given other parameters.
        """
        
        alpha0 = k/2.0
        beta0 = k*sigma2_hat/2.0
        alpha = alpha0 + self.size/2.0
        data = np.copy(self.data)
        y = np.copy(self.y)
        w = np.copy(self.w)
        data = matrix(data)
        y = matrix.transpose(matrix(y))
        w = matrix.transpose(matrix(w))
        beta = beta0 + 0.5*matrix.transpose(y - data*w)*(y - data*w)
        self.sigma2 = invgamma.rvs(alpha, scale = beta)
        return

    def sample_vs(self, k, vs_hat):
        
        """
        Samples from distribution of the error given other parameters.
        """
        
        alpha0 = k/2.0
        beta0 = k*vs_hat/2.0
        alpha = alpha0 + float(len(self.big_ones))/2.0
        ww = np.sum(self.w**2)
        beta = beta0 + ww
        self.vs = invgamma.rvs(alpha, scale = beta)
        return
        
    def sample_w(self):
        
        """
        Samples from distribution of weights given other parameters.
        """
        
        nothing, eta_rem = self.regcalc()
        eta_rem = np.delete(eta_rem, self.big_zeros, 0)
        eta_rem = np.delete(eta_rem, self.big_zeros, 1)
        X = self.data
        X = np.delete(X, self.big_zeros, 1)
        y = matrix.transpose(matrix(self.y))
        mean_rem = (1.0/(self.sigma2))*eta_rem*matrix.transpose(matrix(X))*y
        mean_simp = np.zeros(len(self.big_ones))
        for i in range(len(self.big_ones)):
            mean_simp[i] = float(mean_rem[i])
        if len(mean_simp) > 0:
            self.w[self.big_ones] = multivariate_normal.rvs(mean_simp, eta_rem)
        self.w[self.big_zeros] = 0.0
        return

    def cens_replace(self):
        
        """
        Replaces values in dependant variables that are censored with
        random samples from a truncated normal distribution with mean and
        variance given by the current values for those
        """
        
        # arrays storing mean and variances of the censored y-values
        cens_mean = np.zeros(len(self.cens))
        cens_sigma2 = np.zeros(len(self.cens))
        # add each weight multiplied by it's dependent variable
        for i in range(self.d):
            cens_mean = cens_mean + float(self.w[i])*self.data[self.cens, i]
   
        self.y[self.cens] = multi_truncnorm(cens_mean, self.sigma2,
                                             self.cens_value)
        return 
    
    def gibbs_chain(self, N, est_p0 = True, censored = True, verbose = True):
        
        """
        Returns a series of Gibbs samples. We sample each parameter in turn. 
        """

        self.__validate_params()
        
        p0_out = np.zeros(N)
        vs_out = np.zeros(N)
        sigma2_out = np.zeros(N)
        w_out = np.zeros([N, self.d])
        zs_out = np.zeros([N, self.g])
        w_out[0] = self.w
        zs_out[0] = self.zs
        
        for i in range(N - 1):
            if censored == True:
                self.cens_replace()
            # We only sample need to sample zs for spike and slab prior
            if self.prior == 'spike_n_slab':
                for k in range(len(self.zs)):
                    self.sample_z_i(k)
            if est_p0 == True:
                self.sample_p0(k = 3.0)
            self.sample_sigma2(k = 3.0, sigma2_hat = 0.0002)
            self.sample_vs(k = 3.0, vs_hat = 0.5)
            self.sample_w()
            
            p0_out[i + 1] = self.p0
            w_out[i + 1] = self.w
            zs_out[i + 1] = self.zs
            vs_out[i + 1] = self.vs
            sigma2_out[i + 1] = self.sigma2
            
            if verbose == True and i % 1000 == 0:
                print("value of weights at iteration {} is: ".format(i))
                print(self.w)
                print("value of zs at iteration {} is: ".format(i))
                print(self.zs)
                print("value of p0 at iteration {} is {}".format(i, self.p0))
                print(self.sigma2)
                
        return w_out, zs_out, p0_out, vs_out, sigma2_out

def run_MCMC(X, y, sigma2, vs, weights, zs, p0, multi_dim,
             cens_value, censored = True, prior = 'spike_n_slab',
             save_message = None, samples = 2000):


    """
    Run a Gibbs sampler for a certain amount of steps with some data, and
    save the results, with the choice of 'spike and slab' or Gaussian priors.
    """

    
    Test = Spike_N_Slab(X, y, sigma2, vs, weights, zs, p0, multi_dim,
                        cens_value, prior)

    assert prior == 'gauss' or 'spike_n_slab', "unrecognized prior"

    if prior == 'gauss':
        ww, zz, pp, vv, ss = Test.gibbs_chain(samples, est_p0 = False,
                                              censored = censored)
    elif prior == 'spike_n_slab':
        ww, zz, pp, vv, ss = Test.gibbs_chain(samples, censored = censored)

    time = datetime.now().strftime(r"%y%m%d_%H%M")
    title = "run_at_time_{}.csv".format(time)

    if save_message == None:
        outdir = "spike_slab_results/s_n_s_weights"
        np.savetxt(os.path.join(outdir, title), ww)
        outdir = "spike_slab_results/s_n_s_zs"
        np.savetxt(os.path.join(outdir, title), zz)
        outdir = "spike_slab_results/s_n_s_p0"
        np.savetxt(os.path.join(outdir, title), pp)
        outdir = "spike_slab_results/s_n_s_sigma2"
        np.savetxt(os.path.join(outdir, title), ss)
        outdir = "spike_slab_results/s_n_s_vs"
        np.savetxt(os.path.join(outdir, title), vv)
        
    else:
        title = save_message + ".csv"
        outdir = "spike_slab_results/s_n_s_weights"
        np.savetxt(os.path.join(outdir, title), ww)
        outdir = "spike_slab_results/s_n_s_zs"
        np.savetxt(os.path.join(outdir, title), zz)
        outdir = "spike_slab_results/s_n_s_p0"
        np.savetxt(os.path.join(outdir, title), pp)
        outdir = "spike_slab_results/s_n_s_sigma2"
        np.savetxt(os.path.join(outdir, title), ss)

    return ww, zz, pp, ss, vv



