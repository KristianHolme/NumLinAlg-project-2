import numpy as np

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