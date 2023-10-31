import numpy as np
import scipy.sparse as scsp
from time import time
norm = np.linalg.norm


def bestApproxSVD(A, k):
    U, S, Vh = np.linalg.svd(A)
    m, n = A.shape
    X = np.zeros((m, n))
    # for j in range(0, k):
    #     X = X + S[j]*np.outer(U[:, j],(Vh[j, :]))
    Skdiag = np.diag(S[:k])
    X = U[:, :k]@Skdiag@(Vh[:k, :])
    return X
   
def bestApproxOverTime(ts, A, k, verbose=0):
    tStart = time()
    Xs = []
    for t in ts:
        X = bestApproxSVD(A(t), k) 
        Xs.append(X) 
    tend = time() - tStart
    if verbose: print(f"BestApproxSVD finished in {tend}s")
    return Xs, tend

def lanczosSVD(A, k, b, orth=True):
    Pk, Qk, Bk = LanczosBidiag(A, k, b, orth=orth)
    U, S, Vh = np.linalg.svd(Bk.toarray())
    return U, np.diag(S), Vh

def LanczosOverTime(ts, A, k, b, orth=True, verbose = 0):
    tStart = time()
    W = []
    for t in ts:
        Pk, Qk, Bk = LanczosBidiag(A(t), k, b, orth=orth)

        Aprox = Pk@Bk@(Qk.T)
        W.append(Aprox)
    tend = time() - tStart
    if verbose: print(f"Lanczos approx. Finished in {tend}s")
    return W, tend
     
def LanczosBidiag(A, k, b, orth=True):
    u, v, alfa, beta = LanczosBidiagMain(A, k, b, orth)

    Pk = u.T
    Qk = v.T
    Bk = scsp.diags([alfa, beta[1:]], [0, -1]).toarray()
    return Pk, Qk, Bk


def LanczosBidiagMain(A, k, b, orth):
    m, n = A.shape
    v = np.zeros((k, n)) #row k is vk
    u = np.zeros((k, m))
    alfa = np.zeros(k)
    beta = np.zeros(k)
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