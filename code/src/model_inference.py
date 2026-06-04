import numpy as np


#**********************************************************
# L I F T I N G   M A T R I X
#**********************************************************


def makeC(eta, dmat):
    W = np.exp(-0.5 * (dmat / eta) ** 2)
    A = np.diag([1 / w.sum() ** 0.5 for w in W])
    return A @ W @ A


def get_basis(eta, dmat):
    n_demes = dmat.shape[0]
    I = np.identity(n_demes)
    C = makeC(eta, dmat)
    L = I - C
    # eigh because L is symmetric positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = eigvals.argsort()
    return eigvals[idx], eigvecs[:, idx]


def simple_basis(eta, dmat, Rmin):
    eigvals, eigvecs = get_basis(eta, dmat)
    varexp = np.cumsum(1 - eigvals) / np.sum(1 - eigvals)
    idx = np.where(varexp > Rmin)[0][0]
    Q = idx + 1
    A = eigvecs[:, :Q]
    return A, int(Q), varexp
