# Obtain indices for the k-bins corresponding to kmin <= k <= kmax and the selected power spectrum multipoles

import numpy as np

def get_fit_selection(kbins, kmin = 0.0, kmax = 0.4, pole_selection = [True, True, True, True, True]):
    k_fit_selection = np.logical_and(kmin<=kbins,kbins<=kmax)
    pole_fit_selection = np.repeat(pole_selection, len(kbins)/len(pole_selection))
    fit_selection = k_fit_selection * pole_fit_selection

    return fit_selection