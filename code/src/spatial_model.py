
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


def get_c(pops,dmat,d0,gamma0,gamma1,alpha,rate):
    '''
    pops: population sizes of each deme
    dmat: distance matrix between demes
    d0: distance offset
    gamma0: distance decay parameter
    gamma1: distance decay parameter that depends on distance
    alpha: population size exponent
    rate: overall individual migration rate
    '''
    apops = pops**alpha
    n = dmat.shape[0]
    P = pops.sum()
    d = dmat.copy()
    x = np.log(d0+d)
    W = (d0+d)**(-gamma0-x*gamma1)
    W[np.diag_indices(n)]=0
    return rate*P/(apops@W@apops)




def migration_generator(dmat,pops,d0,gamma0,gamma1,alpha,c):
    '''
    migration generator matrix based on the gravity model of migration.
    pops: population sizes of each deme
    dmat: distance matrix between demes
    d0: distance offset
    gamma0: distance decay parameter
    gamma1: distance decay parameter that depends on distance
    alpha: population size exponent
    c: scaling constant to match overall migration rate
    '''
    N = dmat.shape[0]
    W = np.zeros((N,N))
    for i in range(N):
        ds = dmat[i,:].copy()
        x = np.log(d0+ds)
        w = c*pops[i]**(alpha-1)*pops**alpha/(d0+ds)**(gamma0+x*gamma1)
        w[i]=0
        W[i,:]=w
        W[i,i]=-w.sum()
    return W


def diffusion_generator(dmat,pops,R=100):
    '''
    diffusion generator matrix based on a Gaussian kernel of migration.
    pops: population sizes of each deme
    dmat: distance matrix between demes
    R: characteristic interaction distance (in km)
    '''
    N = dmat.shape[0]
    L = np.zeros((N,N))
    for i in range(N):
        ds = dmat[i,:].copy()
        l = pops*np.exp(-0.5*(ds/R)**2)
        l = l/l.sum()
        L[i,:]=l
        L[i,i] -= 1.0
    return L


def apply_mig_diff_kernels(P,t0,t1,W,L):
    '''
    P: n_demes x n_timepoints x n_variants array of allele frequencies
    t0: starting time point index
    t1: ending time point index
    W: migration generator matrix
    L: diffusion generator matrix
    returns: state vectors, state vector increments, 
    and migration/diffusion generators applied to the state vectors 
    for each time point in the range [t0,t1)
    '''
    X = P.copy()
    Xs = P[:,t0:t1,:]
    dXs = P[:,t0+1:t1+1,:]-P[:,t0:t1,:]
    WXs = np.stack([W@X[:,t,:] for t in range(t0,t1)],axis=1)
    LXs = np.stack([L@X[:,t,:] for t in range(t0,t1)],axis=1)
    return Xs,dXs,WXs,LXs

def dX_model(X,WX,LX,J,s,beta):
    '''
    X: n_demes x n_variants array of allele frequencies at a single time point
    WX: migration generator applied to X    
    LX: diffusion generator applied to X
    J: strength of diffusion
    s: bias term for each variant
    beta: accommodation parameter for frequency dependence of bias
    calculates the expected change in allele frequencies based on the model
    '''
    K=X.shape[1]
    f = s + beta*X
    fbar = (f*X)@np.ones(K)
    return X*(f-fbar[:,None]) + WX + J*LX


def make_objective(P,W,L,t0,t1,J):
    '''
    Creates an objective function for optimizing model parameters.
    '''
    Xs,dXs,WXs,LXs = apply_mig_diff_kernels(P,t0,t1,W,L)
    def obj(args):
        beta=args[0]
        s=args[1:]
        dXms = np.stack([dX_model(Xs[:,t,:],WXs[:,t,:],LXs[:,t,:],J,s,beta) for t in range(Xs.shape[1])], axis=1)
        return np.mean((dXms-dXs)**2)
    return obj


def make_spatial_bias_objective(P,W,L,A,J,t0,t1,reg=1e-4):
    '''
    W is NxN migration laplacian
    L is NxN diffusion laplacian
    A is NxQ lifting matrix
    t0 and t1 are time point indices for training data
    J is diffusion rate
    reg is the L2 regularisation weight on psi (default 1e-4)
    '''
    Xs,dXs,WXs,LXs = apply_mig_diff_kernels(P,t0,t1,W,L)
    N = A.shape[0]
    Q = A.shape[1]
    K = P.shape[2]
    def obj(args):
        beta=args[0]
        #psi is a QxK matrix
        psi=args[1:].reshape((Q,K))
        #s is an NxK matrix
        s = A@psi
        dXms = np.stack([dX_model(Xs[:,t,:],WXs[:,t,:],LXs[:,t,:],J,s,beta) for t in range(Xs.shape[1])], axis=1)
        return np.mean((dXms-dXs)**2)+reg*np.mean(s**2)
    return obj


def kld_time_series(X_true, X_pred, eps=1e-12):
    """
    Returns KL(X_true || X_pred) at each time t.

    X_true, X_pred: (N,T,K)
    eps: numerical stabiliser
    """
    P = np.clip(X_true, eps, 1.0)
    Q = np.clip(X_pred, eps, 1.0)
    
    # Sum over variants to get KL (i,t) matrix
    kl_it = np.sum(P * np.log(P / Q), axis=2)   
    # Return average over space
    return np.mean(kl_it, axis=0)           


class model:
    def __init__(self,W,L,J,s,beta,x0):
        self.A = W + J*L
        #Bias
        self.s = s
        self.beta = beta
        self.x0=x0.copy()
        self.K = x0.shape[1]


    def rhs(self,x_flat,t):
        x = x_flat.reshape(self.x0.shape)
        f = self.s + self.beta*x
        fbar = (f*x)@np.ones(self.K)

        return (x*(f-fbar[:,None]) + (self.A@x)).flatten()
    
    def solve(self,ts):
        xsol = odeint(self.rhs,self.x0.flatten(),ts)
        return xsol.reshape(xsol.shape[0],*self.x0.shape)


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