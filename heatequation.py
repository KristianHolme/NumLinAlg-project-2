import numpy as np
from scipy.sparse import diags

def diff(m, n):
    assert m == n, "Not square matrix!"
    N = n-1
    k = 1/N
    Ltilde = np.zeros((N+1, N+1))
    offdiag = np.ones(N-1)*-1
    maindiag = np.ones(N)*2
    L = 1/k**2 * diags([offdiag, maindiag, offdiag], [-1, 0, 1]).toarray()
    
    Ltilde[1:N+1, 1:N+1] = L
    # dA = Ltilde@A + A@Ltilde
    return Ltilde, Ltilde

def initval(g,N):
    x = np.linspace(0,1, N+1)
    y = x
    A = g(x[:, None], y[None, :])
    return A