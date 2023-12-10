'''
Implementation of the linear shrinkage estimator of the precision matrix as outlined in
Bodnar T., Gupta A. K., Parolya N., 2016, Journal of Multivariate Analysis, 146, 223
'''


import numpy as np
from scipy import linalg

def trace_norm(M):
    return np.trace(linalg.sqrtm(M@M.T))

def Frob_norm_sq(M):
    return np.trace(M@M.T)

def linear_shrinkage(X, pi_0):
    '''
    X: p x n data matrix of p random variables and n realizations.
    pi_0: target precision matrix.
    '''
    p, n = X.shape

    # Shrinkage requires matrix to have zero mean
    x_mean = np.sum(X, axis=1)/n # Find mean of each row
    x_mean_M = np.tile(x_mean, (n, 1)).T # Repeat mean values as columns in a p x n matrix
    Y = X - x_mean_M

    S = 1/(n-1)*Y@Y.T
    Sinv = np.linalg.inv(S)

    alpha = 1 - p/n - (1/n * trace_norm(Sinv)**2 * Frob_norm_sq(pi_0)) / (Frob_norm_sq(Sinv) * Frob_norm_sq(pi_0) - np.trace(Sinv@pi_0)**2)
    beta = np.trace(Sinv@pi_0) / Frob_norm_sq(pi_0) * (1-p/n-alpha)

    Cinv = alpha*Sinv + beta*pi_0

    return Cinv, alpha, beta
