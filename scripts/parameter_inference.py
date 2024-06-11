import numpy as np
from nbodykit.lab import cosmology
import pocomc as pc
from multiprocessing import Pool
import os
import sys

module_path = os.path.abspath(os.path.join("../lib"))
if module_path not in sys.path:
    sys.path.append(module_path)

from pk_tools import pk_tools
import pk_fit_selection as fit

class PowerSpectrumLikelihood():

    def __init__(self, mat_type, est_type, nmocks, v):
        datapath = "../data/BOSS_DR12_NGC_z1/"
        datafile = datapath + "ps1D_BOSS_DR12_NGC_z1_COMPnbar_TSC_700_700_700_400_renorm.dat"
        self.W = pk_tools.read_matrix(datapath + "W_BOSS_DR12_NGC_z1_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix")
        self.M = pk_tools.read_matrix(datapath + "M_BOSS_DR12_NGC_z1_V6C_1_1_1_1_1_1200_2000.matrix")

        pk_data_dict = pk_tools.read_power(datafile, combine_bins=10)
        kbins, self.pk_data_vector = pk_tools.dict_to_vec(pk_data_dict)

        pk_model_dict = pk_tools.read_power(datafile, combine_bins=1)
        krange = pk_model_dict["k"]
        cosmo = cosmology.Planck15
        pk_matter = cosmology.LinearPower(cosmo, redshift=0.38, transfer="EisensteinHu")
        self.pk_matter_k = pk_matter(krange)

        pole_selection=[True, False, True, False, False]
        kmin=0.01
        kmax=0.1
        self.fit_selection = fit.get_fit_selection(kbins, kmin=kmin, kmax=kmax, pole_selection=pole_selection)

        mat_path = "../output/BOSS_DR12_NGC_z1/matrices/"
        matrix = pk_tools.read_matrix(mat_path + f"n{nmocks}/{mat_type}_{est_type}/{mat_type}_18_18_{est_type}_{nmocks}_v{v}.matrix")
        if mat_type == "cov":
            # Obtain precision matrix
            Psi = np.linalg.inv(matrix)
            if est_type == "sample":
                # Apply Hartlap correction
                p = np.sum(self.fit_selection)
                H = (nmocks-p-2) / (nmocks-1)
                Psi = H*Psi
        elif mat_type == "pre":
            # The matrix is already the precision matrix
            Psi = matrix
        self.Psi = Psi


    def pk_model(self, theta):
        b = theta[0]
        f = theta[1]

        P0 = (b**2 + 2/3*b*f + 1/5*f**2) * self.pk_matter_k
        P2 = (4/3*b*f + 4/7*f**2) * self.pk_matter_k
        P4 = 8/35*f**2*self.pk_matter_k

        return np.concatenate((P0, P2, P4))


    def loglike(self, theta):
        pk_model_vector = self.pk_model(theta)
        if pk_model_vector is None:
            return -np.inf

        convolved_model = self.W@self.M@pk_model_vector
        diff = self.pk_data_vector - convolved_model
        fit_diff = diff[self.fit_selection]

        return -0.5 * (fit_diff.T@self.Psi@fit_diff)

    def logprior(self, theta):
        b = theta[0]
        f = theta[1]

        if b<0.5 or b>3.5:
            return -np.inf

        if f<0 or f>2:
            return -np.inf

        return 0.0

    def log_posterior(self, theta):
        return self.logprior(theta) + self.loglike(theta)

def run_parameter_inference(mat_type, est_type, nmocks, start, end, ncpus):
    ndim = 2
    nparticles = 1000   # Total of number of samples at the end will be twice this number
                        # because we are adding samples at the end.

    bounds = np.empty((ndim, 2))
    bounds[0] = np.array([0.5, 3.5])
    bounds[1] = np.array([0, 2])

    for v in range(start, end+1):
        print(f"Running version {v}")

        powerlike = PowerSpectrumLikelihood(mat_type=mat_type, est_type=est_type, nmocks=nmocks, v=v)

        prior_samples = np.empty((nparticles, ndim))
        prior_samples[:,0] = np.random.uniform(0.6, 3.4, nparticles)
        prior_samples[:,1] = np.random.uniform(0.1, 1.9, nparticles)

        with Pool(ncpus) as pool:
            sampler = pc.Sampler(
                nparticles,
                ndim,
                log_likelihood=powerlike.loglike,
                log_prior=powerlike.logprior,
                infer_vectorization=False,
                bounds=bounds,
                pool=pool
            )

            sampler.run(prior_samples)
            sampler.add_samples(nparticles)

        results = sampler.results
        np.save(f"../output/BOSS_DR12_NGC_z1/samples/n{nmocks}/{mat_type}_{est_type}/{mat_type}_{est_type}_{nmocks}_results_v{v}", results)

        print(f"Finished version {v}")

if __name__ == "__main__":
    mat_type = str(sys.argv[1])
    est_type = str(sys.argv[2])
    nmocks = int(sys.argv[3])
    start = int(sys.argv[4])
    end = int(sys.argv[5])

    ncpus=2
    run_parameter_inference(mat_type, est_type, nmocks, start, end, ncpus)
