import numpy as np
from scipy.optimize import minimize



def softplus_logsumexp_with_baseline(F):  # F: (N,T,Km1)
    # returns log(1 + sum_k exp(F_k)) stably
    # = logsumexp([0, F_1,...,F_{K-1}])
    # Use max trick:
    mx = np.maximum(0.0, np.max(F, axis=2, keepdims=True))  # (N,T,1)
    return mx[..., 0] + np.log(np.exp(-mx[..., 0]) + np.sum(np.exp(F - mx), axis=2))


def softmax_with_baseline(F):  # F: (N,T,Km1)
    # returns p for Km1 classes and baseline prob pK separately
    mx = np.maximum(0.0, np.max(F, axis=2, keepdims=True))
    e = np.exp(F - mx)                       # (N,T,Km1)
    denom = np.exp(-mx) + np.sum(e, axis=2, keepdims=True)  # (N,T,1)
    p = e / denom                            # (N,T,Km1)
    pK = np.exp(-mx) / denom                 # (N,T,1)
    return p, pK


def objective(z_flat, Ls, Lt, Y):
    # Y: (N,T,K) counts
    N = Ls.shape[0]
    T = Lt.shape[0]
    K = Y.shape[2]
    Km1 = K - 1

    Z = z_flat.reshape(Km1, N, T)            # (Km1,N,T)
    # Build logits F_k = Ls Z_k Lt^T
    F = np.empty((N, T, Km1), dtype=np.float64)
    for k in range(Km1):
        F[:, :, k] = Ls @ Z[k] @ Lt.T

    m = np.sum(Y, axis=2)                    # (N,T)
    logA = softplus_logsumexp_with_baseline(F)  # (N,T)

    nll = np.sum(m * logA - np.sum(Y[:, :, :Km1] * F, axis=2))
    prior = 0.5 * np.sum(Z * Z)
    return nll + prior

def gradient(z_flat, Ls, Lt, Y):
    N = Ls.shape[0]
    T = Lt.shape[0]
    K = Y.shape[2]
    Km1 = K - 1

    Z = z_flat.reshape(Km1, N, T)
    F = np.empty((N, T, Km1), dtype=np.float64)
    for k in range(Km1):
        F[:, :, k] = Ls @ Z[k] @ Lt.T

    m = np.sum(Y, axis=2)                    # (N,T)
    p, _pK = softmax_with_baseline(F)        # p: (N,T,Km1)

    G = np.empty_like(Z)                     # (Km1,N,T)
    for k in range(Km1):
        Gf = m * p[:, :, k] - Y[:, :, k]     # (N,T)  dJ/dF_k
        G[k] = Ls.T @ Gf @ Lt + Z[k]         # (N,T)  dJ/dZ_k

    return G.reshape(-1)

def fit_multinomial_map(Ls, Lt, Y, z0=None, maxiter=200):
    N, T, K = Y.shape
    Km1 = K - 1
    if z0 is None:
        z0 = np.zeros(Km1 * N * T, dtype=np.float64)

    res = minimize(
        objective, z0,
        args=(Ls, Lt, Y),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": maxiter}
    )

    Z_map = res.x.reshape(Km1, N, T)
    F_map = np.stack([Ls @ Z_map[k] @ Lt.T for k in range(Km1)], axis=2)  # (N,T,Km1)
    p_map, pK_map = softmax_with_baseline(F_map)                          # (N,T,Km1), (N,T,1)
    p_full = np.concatenate([p_map, pK_map], axis=2)                      # (N,T,K)

    return res, Z_map, F_map, p_full
