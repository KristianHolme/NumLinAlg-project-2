import numpy as np
from scipy.sparse import diags
from scipy.linalg import expm

def precompAll():
    #run all jitted functions on small problems to compile them
    import testing  
    testing.runLanczos(N=2, k=2, usejit=True, orth=True, verbose=False)
    
def makeY(U, S, V):
    if type(U) is list:
        Ylist = [U@S@(V.T) for U, S, V in zip(U, S, V)]
        return Ylist
    else:
        Y = U@S@(V.T)
        return Y

def getSol(u, t, m, n):
    #eval function u(x,y,t) as matrix at time t
    def fn(x, y):
        return u(x, y, t)
    x = np.linspace(0,1,m)
    y = np.linspace(0,1,n)
    
    s = fn(x[:, None], y[None, :])    
    return s

def makeAfuncs(n=10, eps=1e-3):
    ones = np.ones(n**2 - 1)
    T1 = diags([ones, -1*ones], [-1, 1]).toarray()
    T2 = diags([ones[1:], 0.5*ones, -0.5*ones, -1*ones[1:]], [-2, -1, 1, 2]).toarray()
    np.random.seed(42)
    Ipert = np.identity(n) + np.random.rand(n, n)*0.5
    A1 = np.random.rand(n**2, n**2)*eps
    A1[0:n, 0:n] = Ipert
    np.random.seed(99)
    Ipert = np.identity(n) + np.random.rand(n, n)*0.5
    A2 = np.random.rand(n**2, n**2)*eps
    A2[0:n, 0:n] = Ipert
    
    I = np.identity(n**2)
    Q1 = lambda t: expm(T1*t)@I
    Q2 = lambda t: expm(T2*t)@I
    dQ1 = lambda t: T1@(Q1(t))
    dQ2 = lambda t: T2@(Q2(t))
        
    def A(t):
        return Q1(t)@(A1 + np.exp(t)*A2)@(Q2(t).T)
    
    def dA(t):
        return ( dQ1(t)@(A1 + np.exp(t)*A2)@(Q2(t).T) + 
        Q1(t)@( np.exp(t)*A2@(Q2(t).T) + (A1 + np.exp(t)*A2)@(dQ2(t).T) ) )
    return A, dA
    
    
    
    
    
    