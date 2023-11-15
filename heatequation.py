import numpy as np
from scipy.sparse import diags

def diff(m, n):
    """Info to calculate time derivative of heat equation
    
    To be used with the function linMatODEStep, where a general derivative is on the form du = Qu + uR.
    In this case Q = R = L^~
    
    Args:
    m, n: size of matrix to evaluate
    
    Returns
    Ltilde (numpy array): tridiag array for the discrete laplacian
    --------------||----------------------------------------------
    """
    assert m == n, "Not square matrix!"
    N = n-1
    k = 1/N
    Ltilde = np.zeros((N+1, N+1))
    offdiag = np.ones(N-2)*1
    maindiag = np.ones(N-1)*-2
    L = 1/k**2 * diags([offdiag, maindiag, offdiag], [-1, 0, 1]).toarray()
    
    Ltilde[1:N, 1:N] = L
    # dA = Ltilde@A + A@Ltilde
    return Ltilde, Ltilde

def diff2(m, n):
    #used for testing
    Q, R = diff(m, n)
    Q = Q*0
    R = R*0
    return Q, R

def initval(g,N):
    """Get initial value
    Evaluate g into a N+1 x N+1 matrix
    
    Args:
    g (function): function of x, y to evaluate
    N (int): matrix size
    
    Returns:
    A (numpy array): the evaluation of g
    """
    x = np.linspace(0,1, N+1)
    y = x
    A = g(x[:, None], y[None, :])
    return A