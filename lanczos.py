import numpy as np
import scipy.sparse as scsp
from time import time
norm = np.linalg.norm


def bestApproxSVDRecon(A, k):
    """Best rank k approximation of matrix A
    
    Truncated SVD of A.
    
    Args:
    A (nuomy array): matrix to be approximated
    k (int): rank of approximation
    
    Returns
    X (numpy array): rank k approximation"""
    U, S, Vh = np.linalg.svd(A)
    m, n = A.shape
    X = np.zeros((m, n))
    # for j in range(0, k):
    #     X = X + S[j]*np.outer(U[:, j],(Vh[j, :]))
    Skdiag = np.diag(S[:k])
    X = U[:, :k]@Skdiag@(Vh[:k, :])
    return X

   
def bestApproxOverTime(ts, A, k, verbose=0):
    """Computes rank k best approximation at various time steps
    
    Args:
    ts (list): list of timesteps
    A (funtion): function of time that returns approximand matrix
    k (int): rank of approximation
    verbose (int/bool): info level printed
    
    Returns:
    Xs (list): list of numpy arrays containing the best approx. at timesteps ts
    tTot (float): total time used
    """
    tStart = time()
    Xs = []
    for t in ts:
        X = bestApproxSVDRecon(A(t), k) 
        Xs.append(X) 
    tTot = time() - tStart
    if verbose: print(f"BestApproxSVD finished in {tTot}s")
    return Xs, tTot

def lanczosSVD(A, k, orth=True):
    """Computes SVD of B from Lanczos alg
    
    Args:
    A (numpy array): approximand
    k (int): rank of aproximation
    orth (bool): to use reorthogonalization or not
    
    Returns:
    U, S, Vh (numpy arrays): SVD decomp of Bk
    """
    Pk, Qk, Bk = LanczosBidiag(A, k, orth=orth)
    U, S, Vh = np.linalg.svd(Bk)
    return U, S, Vh #endret til S fra np.diag(S)

def LanczosOverTime(ts, A, k, orth=True, verbose = 0):
    """Computes rank k Lanczos approximation at various time steps
    
    Args:
    ts (list): list of timesteps
    A (funtion): function of time that returns approximand matrix
    k (int): rank of approximation
    orth (bool): to use reorthogonalization or not
    verbose (int/bool): info level printed
    
    Returns:
    W (list): list of numpy arrays containing the approx. at timesteps ts
    tTot (float): total time used
    """
    tStart = time()
    W = []
    debug=False
    for t in ts:
        At = A(t)
        Pk, Qk, Bk = LanczosBidiag(At, k, orth=orth)
        Aprox = Pk@Bk@(Qk.T)
        if debug:
            _, S, _ = np.linalg.svd(At)
            _, SBk, _ = np.linalg.svd(Bk)
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(range(1, 21), S[:20], label='true')
            plt.plot(range(1, k+1), SBk, label='lanczos')
            plt.legend()
            plt.show()
            pass
        W.append(Aprox)
    tTot = time() - tStart
    if verbose: print(f"Lanczos approx. Finished in {tTot}s")
    return W, tTot
     
def LanczosBidiag(A, k, orth=True):
    """ Computes rank k Lanczos aproximation of A
    
    Args:
    A (funtion): function of time that returns approximand matrix
    k (int): rank of approximation
    orth (bool): to use reorthogonalization or not
    
    Returns:
    Pk, Qk, Bk (numpy arrays): Lanczos approximation matrices
    """
    u, v, alfa, beta = LanczosBidiagMain(A, k, orth)

    Pk = u.T
    Qk = v.T
    Bk = scsp.diags([alfa, beta[1:]], [0, -1]).toarray()
    return Pk, Qk, Bk



def LanczosBidiagMain(A, k, orth):
    """Main computation for Lanczos
    
    Args:
    Args:
    A (funtion): function of time that returns approximand matrix
    k (int): rank of approximation
    orth (bool): to use reorthogonalization or not
    
    Returns:
    u (numpy array): left vectors
    v (numpy arrays); right vectors
    alfa (numpy array): diag of B
    beta (numpy array): off-diag of B
    """
    m, n = A.shape
    v = np.zeros((k, n)) #row k is vk
    u = np.zeros((k, m))
    alfa = np.zeros(k)
    beta = np.zeros(k)
    
    # b = np.random.rand(m) #random start point
    # b=A[:,0]
    b=A[:, np.argmax(np.linalg.norm(A, axis=0))] # start with the largest norm col in A
    
    beta[0] = norm(b)
    u[0] = b/beta[0]
    ATu = A.T@u[0]
    alfa[0] = norm(ATu)
    v[0] = ATu/alfa[0]
    for i in range(0, k-1): # i is python index
        Avi_min_alfaui = A@v[i] - alfa[i]*u[i]
        beta[i+1] = norm(Avi_min_alfaui)
        u[i+1] = Avi_min_alfaui/beta[i+1]
        
        w = A.T@u[i+1] - beta[i+1]*v[i]
        if orth:
            for l in range(0, i):
                w = w - (v[l].T@w)*v[l]
                
        alfa[i+1] = norm(w)
        v[i+1] = w/alfa[i+1]
    return u, v, alfa, beta