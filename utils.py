import testing
import numpy as np

def precompAll():
    #run all jitted functions on small problems to compile them
    testing.runLanczos(N=2, k=2, usejit=True, orth=True, verbose=False)
    
def makeY(U, S, V):
    if type(U) is list:
        Ylist = [U@S@(V.T) for U, S, V in zip(U, S, V)]
        return Ylist
    else:
        Y = U@S@(V.T)
        return Y

def getSol(u, t, m, n):
    #eval function at time t
    def fn(x, y):
        return u(x, y, t)
    x = np.linspace(1,1,1)
    
    