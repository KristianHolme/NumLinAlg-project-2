import numpy as np
# import scipy.sparse as scsp
from cayley import *
from time import time

def TimeIntegration(t0, tf, h0, U0, S0, V0, dA, stepfunction, cay=cay1,verbose = False,
                    TOL= 1e-5, maxTimeCuts=3):
    """Performs dyamic low rank approximation
    
    Args:
    t0 (float): start time
    tf (float): end time
    h0 (float): initial target step lenght
    U0, S0, V0 (numpy array): initial low rank approximation matrices
    dA (function): functino to obtain the derivative of the approximand function
    stepfunction (function): step function to use. Can be custom made for different problems.
    cay (function): function for computing the cayley map
    verbose (int/bool): info level for printing
    TOL (float): tolerance for time step control
    maxTimeStepCuts (int): max number of time cuts for time step control
    
    Returns:
    Ulist (list): list of numpy arrays containg the matrix U at all the timesteps
    Slist (list): list of numpy arrays containg the matrix S at all the timesteps
    Vlist (list): list of numpy arrays containg the matrix v at all the timesteps
    timesteps (lsit): list of timesteps used
    dUlist (list): list of numpy arrays containg the matrix dU at all the timesteps
    dSlist (list): list of numpy arrays containg the matrix dS at all the timesteps
    dVlist (list): list of numpy arrays containg the matrix dv at all the timesteps
    tRUn (float): total runtime
    """
    tStart = time()
    Ulist = [U0]
    Vlist = [V0]
    Slist = [S0]
    timesteps = [t0]
    dUlist = []
    dSlist = []
    dVlist = []

    t = t0
    h = h0
    tnew = t
    hnew = h
    count = 0
    j = 0
    
    if verbose: print(f"Starting integrating at t:{t0}, with step size {h0}")
    
    while t <= tf:
        if verbose > 1: print(f"Solving step {j+1} from t:{round(t, 4)}. Trying h:{round(hnew, 6)}")
        U, S, V, h, t, hnew, dS, dU, dV = integrateStep(Ulist[-1], Slist[-1], Vlist[-1], dA, hnew, t, count, 
                                            stepfunction, cay, TOL, maxTimeCuts, verbose=verbose)
        if verbose > 1 :print(f"Step complete. Now at t={t}")
        Ulist.append(U)
        Slist.append(S)
        Vlist.append(V)
        dUlist.append(dU)
        dSlist.append(dS)
        dVlist.append(dV)
        timesteps.append(t)
        j = j + 1
        
    if t > tf:
        #do the last step again with shorter step length
        t = t - h
        h = tf - t
        U, S, V, sigma, dS, dU, dV = stepfunction(Ulist[-2], Slist[-2], Vlist[-2], dA, h, t, cay)
        Ulist[-1] = U
        Slist[-1] = S
        Vlist[-1] = V
        dUlist[-1] = (dU)
        dSlist[-1] = (dS)
        dVlist[-1] = (dV)
        timesteps[-1] = tf
    tRUn = time() - tStart
    if verbose: print(f"Finished integrating in {j} steps, {tRUn}s")
    return Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRUn

def integrateStep(U0, S0, V0, dA, h, t, count, stepfunction, cay, TOL, maxTimeCuts, verbose=False):
    """Does one time integration step, with step size control
    
    Uses recursion :O
    
    Args:
    U0, S0, V0 (numpy arrays): approximation matrices at previous time step
    dA (function): function for derivative of approximand passed to stepfunction
    h (float): target step size
    t (float): current time
    count (int): number of time step cuts for current step
    stepfunction: function for calulating stepsize
    cay (functiom): cayley map function
    TOL (float): tolerance for step size control
    maxTimeCuts (int): max number of time step cuts
    verbose (int/bool): level of info printed

    Returns:
    U, S, V (numpy arrays): low rank approximation matrices
    h (float): time final accepted time step
    t (float): current time after step
    hnew (float): next target step size
    dS, dU, dV (numpy arrays): derivatives of S, U, V
    """
    U, S, V, sigma, dS, dU, dV = stepfunction(U0, S0, V0, dA, h, t, cay)
        
    t = t + h
    #h is the step size taken to get to new t 
    tnew, hnew = stepcontrol(sigma, TOL, h, t)
    
    if tnew < t and count <= maxTimeCuts:
        if verbose > 1: print(f"Cutting time step. h:{hnew}, count:{count}")
        U, S, V, h, t, hnew, dS, dU, dV = integrateStep(U0, S0, V0, dA, hnew, tnew, count+1, stepfunction, cay, TOL, 
                                            maxTimeCuts, verbose=verbose)
        #h is the step size taken to get to t
    if count > maxTimeCuts and verbose>1:
        print(f"Maximum time step cuts reached ({maxTimeCuts})")

    return U, S, V, h, t, hnew, dS, dU, dV
    
def USVstep(Uj, Sj, Vj, dA, h, tj, cay):
    """step function for use in integrateStep
    
    Used for problem 4 and 5 when the derivative is time dependent
    
    Args:
    Uj, Sj, Vj (numpy arrays): low rank approx. matrices at step j
    dA (function): function to evaluate derivative of A at time t_j
    h (float): step size
    tj (float): time at step j
    cay (function) cayley map function
    
    Returns:
    Ujp1, Sjp1, Vjp1 (numpy arrays) low rank approx. matrices at step j+1
    sigma (float): error estimate
    dSj, dUj, dVj (numpy arrays) derivaties of low rank approx. matrices at step j+1
    """
    m, k = Uj.shape
    n, k1 = Vj.shape
    k2,k3 = Sj.shape
    assert k  == k1 and k2 == k1, "dimension mismatch"

    Ik = np.identity(k)
    Im = np.identity(m)
    In = np.identity(n)
    
    KS1 = h*Uj.T@dA(tj)@Vj
    Sjint = Sj + 0.5*KS1
    
    FUj = (Im - Uj@(Uj.T))@dA(tj)@Vj@(np.linalg.inv(Sj))
    Ujint = cay(Uj, h/2*FUj)@Uj
    FVj = (In - Vj@(Vj.T))@(dA(tj).T)@Uj@(np.linalg.inv(Sj.T))
    Vjint = cay(Vj, h/2*FVj)@Vj
    
    KS2 = h*Ujint.T@dA(tj + h/2)@Vjint
    Sjp1 = Sj + KS2
    
    FUjint = (Im - Ujint@(Ujint.T))@dA(tj+h/2)@Vjint@(np.linalg.inv(Sjint))
    Ujp1 = cay(Ujint, h*FUjint)@Uj #not Ujint? no
    FVjint = (In - Vjint@(Vjint.T))@dA(tj+h/2).T@Ujint@(np.linalg.inv(Sjint.T))
    Vjp1 = cay(Vjint, h*FVjint)@Vj
    
    Sest = Sjint + 0.5*KS1
    Uest = cay(Uj, h*FUj)@Uj
    Vest = cay(Vj, h*FVj)@Vj
    
    sigma = np.linalg.norm(Ujp1@Sjp1@(Vjp1.T) - Uest@Sest@(Vest.T))
    
    dSj = (Uj.T)@dA(tj)@Vj
    dUj = (FUj@(Uj.T)- Uj@(FUj.T))@Uj
    dVj = (FVj@(Vj.T)- Vj@(FVj.T))@Vj
    
    
    return Ujp1, Sjp1, Vjp1, sigma, dSj, dUj, dVj

def stepcontrol(sigma, TOL, h, t):
    """Step control function
    
    Args:
    sigma (float): approximation of error
    TOL (float): error tolerance
    h (float): step size for current step
    t (float): current time
    
    Returns:
    tnew, hnew (floats): new current time and new step size target
    """
    if sigma > TOL:
        tnew = t-h
        hnew = h/2
    else:
        if sigma > TOL/2:
            R = (TOL/sigma)**(1/3)
            if R > 0.9 or R < 0.5:
                R = 0.7
        elif sigma > TOL/16:#byttet fra 16 til 3
            R = 1
        else:
            R = 2
        hnew = R*h
        tnew = t
    return tnew, hnew

def linMatODEStep(Uj, Sj, Vj, dA, h, tj, cay):
    """Step function for linear ODE problems
    
    Args:
    Uj, Sj, Vj (numpy arrays): low rank approx. matrices at step j
    dA (function): function to get derivative matrices Q, R (indep. of time)
    h (float): step size
    tj (float): time at step j (not used, but here to have same args as USVstep)
    cay (function) cayley map function
    
    Returns:
    Ujp1, Sjp1, Vjp1 (numpy arrays) low rank approx. matrices at step j+1
    sigma (float): error estimate
    dSj, dUj, dVj (numpy arrays) derivaties of low rank approx. matrices at step j+1
    """
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
    
    dSj = (Uj.T)@Q@Uj@Sj + Sj@(Vj.T)@R@Vj
    dUj = (FUj@(Uj.T)- Uj@(FUj.T))@Uj
    dVj = (FVj@(Vj.T)- Vj@(FVj.T))@Vj
    
    
    return Ujp1, Sjp1, Vjp1, sigma, dSj, dUj, dVj
    