import numpy as np
# import scipy.sparse as scsp
from numba import jit
from cayley import *
from time import time

def TimeIntegration(t0, tf, h0, U0, S0, V0, dA, stepfunction, cay=cay1,verbose = False,
                    TOL= 1e-5, maxTimeCuts=3):
    tStart = time()
    Ulist = [U0]
    Vlist = [V0]
    Slist = [S0]
    timesteps = [t0]

    t = t0
    h = h0
    tnew = t
    hnew = h
    count = 0
    j = 0
    
    if verbose: print(f"Starting integrating at t:{t0}, with step size {h0}")
    
    while t <= tf:
        if verbose: print(f"Solving step {j+1}. Trying:{t} -> {t+hnew}")
        U, S, V, h, t, hnew = integrateStep(Ulist[-1], Slist[-1], Vlist[-1], dA, hnew, t, count, 
                                            stepfunction, cay, TOL, maxTimeCuts, verbose=verbose)
        if verbose:print(f"Step complete. Now at t={t}")
        Ulist.append(U)
        Slist.append(S)
        Vlist.append(V)
        timesteps.append(t)
        j = j + 1
        
    if t > tf:
        #do the last step again with shorter step length
        t = t- h
        h = tf - t
        U, S, V, sigma = stepfunction(U0, S0, V0, dA, h, t, cay)
        Ulist[-1] = U
        Slist[-1] = S
        Vlist[-1] = V
        timesteps[-1] = tf
    if verbose: print(f"Finished integrating in {j} steps, {time()-tStart}s")
    return Ulist, Slist, Vlist, timesteps

def integrateStep(U0, S0, V0, dA, h, t, count, stepfunction, cay, TOL, maxTimeCuts, verbose=False):
    U, S, V, sigma = stepfunction(U0, S0, V0, dA, h, t, cay)
        
    t = t + h
    #h is the step size taken to get to t
    tnew, hnew = stepcontrol(sigma, TOL, h, t)
    
    if tnew < t and count <= maxTimeCuts:
        if verbose: print(f"Cutting time step. h:{hnew}, count:{count}")
        U, S, V, h, t, hnew = integrateStep(U0, S0, V0, dA, hnew, tnew, count+1, stepfunction, cay, TOL, 
                                            maxTimeCuts, verbose=verbose)
        #h is the step size taken to get to t
    if count > maxTimeCuts and verbose:
        print(f"Maximum time step cuts reached ({maxTimeCuts})")

    return U, S, V, h, t, hnew
    
def USVstep(Uj, Sj, Vj, dA, h, tj, cay):
    m, k = Uj.shape
    n, k1 = Vj.shape
    k2 = Sj.shape
    assert k  == k1 and k2 == k1, "dimension mismatch"

    Ik = np.identity(k)
    Im = np.identity(m)
    In = np.identity(n)
    
    KS1 = h*Uj.T@dA(m, n, tj)@Vj
    Sjint = Sj + 0.5*KS1
    
    FUj = (Im - Uj@(Uj.T))@dA(m, n, tj)@Vj@(np.linalg.inv(Sj))
    Ujint = cay(Uj, h/2*FUj)@Uj
    FVj = (In - Vj@(Vj.T))@(dA(m, n, tj).T)@Uj@(np.linalg.inv(Sj.T))
    Vjint = cay(Vj, h/2*FVj)@Vj
    
    KS2 = h*Ujint.T@dA(m, n, tj + h/2)@Vjint
    Sjp1 = Sj + KS2
    
    FUjint = (Im - Ujint@(Ujint.T))@dA(m, n, tj+h/2)@Vjint@(np.linalg.inv(Sjint))
    Ujp1 = cay(Ujint, h*FUjint)@Uj #not Ujint? no
    FVjint = (In - Vjint@(Vjint.T))@dA(m, n, tj+h/2).T@Ujint@(np.linalg.inv(Sjint.T))
    Vjp1 = cay(Vjint, h*FVjint) 
    
    Sest = Sjint + 0.5*KS1
    Uest = cay(Uj, h*FUj)@Uj
    Vest = cay(Vj, h*FVj)@Vj
    
    sigma = np.linalg.norm(Ujp1@Sjp1@(Vjp1.T) - Uest@Sest@(Vest.T))
    
    return Ujp1, Sjp1, Vjp1, sigma

def stepcontrol(sigma, TOL, h, t):
    if sigma > TOL:
        tnew = t-h
        hnew = h/2
    else:
        if sigma > TOL/2:
            R = (TOL/sigma)**(1/3)
            if R > 0.9 or R < 0.5:
                R = 0.7
        elif sigma > TOL/3:#byttet fra 16 til 3
            R = 1
        else:
            R = 2
        hnew = R*h
        tnew = t
    return tnew, hnew

def linMatODEStep(Uj, Sj, Vj, dA, h, tj, cay):
    m, k = Uj.shape
    n, k1 = Vj.shape
    k2 = Sj.shape[0]
    assert k  == k1 and k2 == k1, "dimension mismatch"
    
    Q, R = dA(m, n)
    
    Ik = np.identity(k)
    Im = np.identity(m)
    In = np.identity(n)
    
    KS1 = h*((Uj.T)@Q@Uj@Sj + Sj@(Vj.T)@R@Vj)#experiment: h utenfor alt
    Sjint = Sj + 0.5*KS1
    
    FUj = (Im - Uj@(Uj.T))@Q@Uj
    Ujint = cay(Uj, h/2*FUj)@Uj
    FVj = (In - Vj@(Vj.T))@(R.T)@Vj
    Vjint = cay(Vj, h/2*FVj)@Vj
    
    KS2 = h*((Ujint.T)@Q@Ujint@Sjint + Sjint@(Vjint.T)@R@Vjint) #eksperiment^^
    Sjp1 = Sj + KS2
    
    FUjint = (Im - Ujint@(Ujint.T))@Q@Ujint
    Ujp1 = cay(Ujint, h*FUjint)@Uj #not Ujint? no
    FVjint = (In - Vjint@(Vjint.T))@(R.T)@Vjint
    Vjp1 = cay(Vjint, h*FVjint)@Vj
    
    Sest = Sj + KS1
    Uest = cay(Uj, h*FUj)@Uj
    Vest = cay(Vj, h*FVj)@Vj
    
    sigma = np.linalg.norm(Ujp1@Sjp1@(Vjp1.T) - Uest@Sest@(Vest.T))
    
    return Ujp1, Sjp1, Vjp1, sigma
    