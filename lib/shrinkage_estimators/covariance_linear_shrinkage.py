'''
Implementation of the linear shrinkage estimator of the covariance matrix as outlined in
Pope A. C., Szapudi I., 2008, MNRAS, 389, 766
'''

import numpy as np

def linear_shrinkage(P, target=None):
    '''
    P:      p x n data matrix of p random variables and n realizations.
    target: If no target is provided, then diag(S) will be used as target.
            If a target is provided, the assumption is that there is no
            covariance between S and the target.
    '''
    p, n = P.shape
    P_mean = np.sum(P, axis=1)/n # Find mean of each row
    P_mean_matrix = np.tile(P_mean, (n, 1)).T # Repeat mean values as columns in a p x n matrix
    X = P - P_mean_matrix

    W = []
    # Generate W array (which is 3D) of size (n, p, p), order of indices (k, i, j)
    for k in range(n):
        w = X[:,k]
        W.append(np.outer(w, w))
    W_mean = np.sum(W, axis=0)/n

    # Emperically estimated covariance matrix
    S = n / (n-1) * W_mean

    W_mean_rep = np.tile(W_mean, (n, 1, 1))
    V = W - W_mean_rep
    # Compute variance of elements of the covariance matrix
    Var = n / (n-1)**3 * np.sum(V**2, axis=0)

    if target is None:
        # Take as Target the diagonal elements of the sample cov matrix
        M = np.diag(np.diag(Var))
        T = np.diag(np.diag(S))
    else:
        M = np.zeros(Var.shape)
        T = target

    # Compute estimated shrinkage intensity parameter lambda
    lmbda_est = np.sum(Var-M) / np.sum((T-S)**2)

    # Restrict shrinkage intensity to interval [0,1]
    if lmbda_est < 0:
        lmbda_est = 0
    elif lmbda_est > 1:
        lmbda_est = 1

    # Compute shrinkage covariance matrix
    cov_shrinkage = lmbda_est*T + (1-lmbda_est)*S

    return cov_shrinkage, lmbda_est
