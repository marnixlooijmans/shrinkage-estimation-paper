'''
Implementation of the NERCOME estimator of the covariance matrix as outlined in
Joachimi B., 2016, MNRAS: Letters, 466, L83
'''

import numpy as np
import itertools as it

def Q_val(s, X):
    '''
    s: split parameter, i.e. number of data realizations out of N_s in X1.
    X: p x n data matrix of p random variables and n realizations.
    '''
    Nd, Ns = X.shape

    # This is required for the division by s-1 and Ns-s-1.
    # We also require s to be an integer
    assert s >= 2 and s <= (Ns-2) and np.floor(s) == s
    s = s.astype(int) # convert to type int in case s was an integer floating point number

    # Calculate the number of unique combinations of selecting s from Ns columns and compare
    # to the maximum number of averaging that we want to do
    nr_col_combos = np.math.factorial(Ns) // (np.math.factorial(s) * np.math.factorial(Ns-s))
    Nav_max = 1000
    Nav = int(np.amin([nr_col_combos, Nav_max])) # Take the mimimum of the two

    if Nav < nr_col_combos:
        # Fill a matrix with zeros and populate it with random column indices
        # The rows are arrays of column indices that we are going to average over
        # The columns are the number of averaging that we are going to do
        M = np.zeros((Nav,s))
        for i in range(Nav):
            M[i, :] = np.sort(np.random.choice(Ns, s, replace=False)) # Sort required to filter unique duplicates in np.unique
        # Filter out duplicates and cast values as integer type
        # The probability of getting duplicates is very low
        col_combos = np.unique(M.astype(int), axis=0)
    else:
        # If the number of unique combinations is smaller than the maximum averaging number, then we
        # average of the all the possible combinations
        col_combos = np.array(list(it.combinations(range(Ns), s)))

    Z_sum = np.zeros((Nd, Nd))
    S2_sum = np.zeros((Nd, Nd))

    # Go through all the column combinations
    for col_combo in col_combos:
        # Create new array by giving only selecting specific columns
        X1 = X[:, col_combo]
        # Create array which consists of all columns not chosen
        X2 = X[:, np.delete(range(Ns), col_combo)]

        S1 = 1/(s-1) * X1@X1.T
        S2 = 1/(Ns-s-1) * X2@X2.T
        S2_sum += S2

        # Diagonalize S_i = U_i * D_i * U_i^T
        evals1, U1 = np.linalg.eigh(S1)
        D1 = np.diag(evals1)

        # Compute estimator Z = U_1 * diag(U_1^T * S_2 * U_1) * U_1^T
        #Z = np.matmul(np.matmul(U1, np.diag(np.diag(np.matmul(np.matmul(U1.T, S2), U1)))), U1.T)
        Z = U1@np.diag(np.diag(U1.T@S2@U1))@U1.T
        Z_sum += Z

    # Find average of estimated matrices
    Z_avg = Z_sum / len(col_combos)
    S2_avg = S2_sum / len(col_combos)

    # Calculate Frobenius matrix norm, ie mean squared error
    M = Z_avg - S2_avg
    Q_val = np.trace(M@M.T)

    return Q_val, Z_avg


def NERCOME(X, all_splits=False):
    '''
    X:          p x n data matrix of p random variables and n realizations.
    all_splits: If True then try all s in [2,n-2].
                If False then only try a selection as described in both Joachimi (2016) and Lam (2016).
    '''
    Nd, Ns = X.shape

    # NERCOME requires the data vectors to be mean subtracted
    x_mean = np.sum(X, axis=1)/Ns # Find mean of each row
    x_mean_M = np.tile(x_mean, (Ns, 1)).T # Repeat mean values as columns in a Nd x Ns matrix
    Y = X - x_mean_M

    if all_splits == True:
        s = np.arange(Ns-3)+2
    else:
        # Consider following values for s according to paper by Lam in 2016 and Joachimi in 2016
        s_raw = np.unique(np.rint(np.array([
            2*np.sqrt(Ns), Ns-1.5*np.sqrt(Ns), Ns-2.5*np.sqrt(Ns),
            0.1*Ns, 0.15*Ns, 0.2*Ns, 0.25*Ns, 0.3*Ns, 0.35*Ns,
            0.4*Ns, 0.45*Ns, 0.5*Ns, 0.55*Ns, 0.6*Ns, 0.65*Ns,
            0.7*Ns, 0.75*Ns, 0.8*Ns, 0.85*Ns, 0.9*Ns
        ])).astype(int))
        # Restrict s to the interval [2, Ns-2]
        s = np.delete(s_raw, np.concatenate((np.where(s_raw < 2), np.where(s_raw > Ns-2)), axis=None))

    Q = []
    Z = []
    for i in s:
        Q_s, Z_s = Q_val(i, Y)
        Q.append(Q_s)
        Z.append(Z_s)

    # Get value for s corresponding to minimum value for Q
    argmin = np.array(Q).argmin()
    s_min = s[argmin]
    C = Z[argmin]

    return C, s_min
