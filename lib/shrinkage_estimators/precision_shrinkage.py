'''
Implementation of the shrinkage estimator of the precision matrix as outlined in
Bodnar T., Gupta A. K., Parolya N., 2016, Journal of Multivariate Analysis, 146, 223
'''

import numpy as np
from scipy import linalg

def trace_norm(M):
    return np.trace(linalg.sqrtm(M@M.T))

def Frob_norm_sq(M):
    return np.trace(M@M.T)

def shrinkage(X, Pi_0):
    '''
    X: p x n data matrix of p random variables and n realizations.
    Pi_0: target precision matrix.
    '''
    p, n = X.shape

    S = np.cov(X)
    Sinv = np.linalg.inv(S)
    
    # Eq 4.4
    alpha = 1 - p/n - (1/n * trace_norm(Sinv)**2 * Frob_norm_sq(Pi_0)) / (Frob_norm_sq(Sinv) * Frob_norm_sq(Pi_0) - np.trace(Sinv@Pi_0)**2)
    # Eq 4.5
    beta = np.trace(Sinv@Pi_0) / Frob_norm_sq(Pi_0) * (1-p/n-alpha)

    Pi = alpha*Sinv + beta*Pi_0

    return Pi, alpha, beta
