import copy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.sparse import random
from scipy.linalg import block_diag
from sklearn.linear_model import Ridge


class ESN:
    def __init__(self, N_in, N_r, input_scale, average_degree, rho, seed=0):
        self.N_r = N_r
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-input_scale, input_scale, (N_r, N_in))
        num_nonzero = int(N_r/N_in)
        mask = block_diag(*([np.ones((num_nonzero, 1))]*N_in))
        mask = np.pad(mask, pad_width=((0, N_r-mask.shape[0]), (0, 0)), mode='constant', constant_values=0)
        self.W_in *= mask
        self.W_in = csr_matrix(self.W_in)

        def custom_random(size=None, rng=rng):
            return rng.uniform(-1, 1, size)

        p = average_degree / N_r
        A = random(N_r, N_r, density=p, format="csr", random_state=rng, data_rvs=custom_random)
        sp_radius = np.abs(eigs(A, k=1)[0][0])
        self.A = A * (rho / sp_radius) #リザバー隣接行列


    def feed(self, U):
        r = np.zeros(self.N_r) #リザバー状態ベクトル
        self.R = np.zeros((len(U), self.N_r))
        W_inU = (U @ self.W_in.T)
        for i, wu in enumerate(W_inU):
            r = np.tanh(wu + self.A @ r)
            self.R[i, :] = r
        self.r = r

    
    def fit(self, V, beta, trans_steps=0):
        R2 = copy.copy(self.R)
        R2[:, ::2] = R2[:, ::2] ** 2
        ridge = Ridge(alpha=beta, fit_intercept=False)
        ridge.fit(R2[trans_steps:, :], V[trans_steps:, :])
        self.W_out = ridge.coef_
        self.pred = ridge.predict(R2)


    def predict(self, u):
        self.r = np.tanh(self.W_in @ u + self.A @ self.r)
        r2 = copy.copy(self.r)
        r2[::2] = r2[::2] ** 2
        return self.W_out @ r2
    