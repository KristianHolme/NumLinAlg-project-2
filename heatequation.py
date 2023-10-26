import numpy as np
from scipy.sparse import diags

def diff(A):
    m, n = A.shape
    assert m == n, "Not square matrix!"
    N = n-1
    k = 1/N
    Ltilde = np.zeros((N+1, N+1))
    offdiag = np.ones(N-1)*-1
    maindiag = np.ones(N)*2
    L = 1/k**2 * diags([offdiag, maindiag, offdiag], [-1, 0, 1])
    
    Ltilde[1:N, 1:N] = L
    dA = Ltilde@A + A@Ltilde
    return A

def initval(g,N):
    x = np.linspace(0,1, N+1)
    y = x
    A = g(x[:, None], y[None, :])
    return A