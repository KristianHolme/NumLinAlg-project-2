import numpy as np
from scipy.sparse import diags
from scipy.linalg import expm
from lanczos import *
from cayley import *
from heatequation import *
from plotting import *
from timeIntegration import *
import pandas as pd
norm = np.linalg.norm

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
def makedY(U, S, V, dU, dS, dV):
    assert len(U) == (len(dU)+1)
    if type(dU) is list:
        dYlist = [dU@S@(V.T) + U@dS@(V.T) + U@S@(dV.T) for U, S, V, dU, dS, dV in zip(U, S, V, dU, dS, dV)]
        return dYlist
    else:
        dY = dU@S@(V.T) + U@dS@(V.T) + U@S@(dV.T)
        return dY

def getSol(u, t, m, n):
    #eval function u(x,y,t) as matrix at time t
    def fn(x, y):
        return u(x, y, t)
    x = np.linspace(0,1,m)
    y = np.linspace(0,1,n)
    
    s = fn(x[:, None], y[None, :])    
    return s

def g(x, y):
    return np.sin(np.pi*x)*np.sin(2*np.pi*y)
def u(x, y, t):
    return np.exp(5*np.pi**2*t)*np.sin(np.pi*x)*np.sin(2*np.pi*y)

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
    
def GetTimeIntegrationResults(n=10, k=10, verbose=1, epss=[1e-3],
                              cay=cay1, h0=0.01, TOL=1e-3, maxTimeCuts=10, t0=0, tf=1):
    # epss = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    resultsByEps = {}
    for i, eps in enumerate(epss):
        A, dA = makeAfuncs(n, eps=eps)
        # N = 99
        #run timeintegration
        A0 = A(t0)
        b = np.random.rand(A0.shape[0])
        Pk, Qk, Bk = LanczosBidiag(A0, k, b)
        U0, S0, V0 = Pk, Bk, Qk
        
        Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRun = TimeIntegration(
        t0, tf, h0, U0, S0, V0, dA, USVstep, cay=cay, verbose = verbose,
        TOL= TOL, maxTimeCuts=maxTimeCuts)
        Ylist = makeY(Ulist, Slist, Vlist)
        dYlist = makedY(Ulist, Slist, Vlist, dUlist, dSlist, dVlist)
        
        ts = timesteps
        # ts = np.arange(t0, tf, h0)
        W, Wtime = LanczosOverTime(ts, A, k, b, verbose=verbose)
        X, Xtime = bestApproxOverTime(ts, A, k, verbose=verbose)
        
        err = lambda f, H, tlist: [  norm(f(t)-H[i]) for i, t in enumerate(tlist)]
        
        Werr = err(A, W, ts)
        Xerr = err(A, X, ts)
        Yerr = err(A, Ylist, timesteps)
        dYerr = err(dA, dYlist, timesteps[0:-1])
        XYerr = [norm(X[i]-Ylist[i]) for i in range(len(X))]
        
        resultsByEps[eps] = {'Werr':Werr, 'Xerr':Xerr, 'Yerr':Yerr, 'dYerr':dYerr,
                        'Wtime':Wtime, 'Xtime':Xtime, 'Ytime':tRun, 'timesteps':timesteps,
                        'XYerr':XYerr}
        
    return resultsByEps
    
def makeTables(resultsByKByEps):
    tables_by_k = {}
    errorNames = ["$||W(t)-A(t)||$","$||X(t)-A(t)||$","$||Y(t)-A(t)||$", "$||X(t)-Y(t)||$", "$||\dot{Y}(t)-\dot{A}(t)||$"]
    errorKeys = ['Werr', 'Xerr', 'Yerr', 'XYerr', 'dYerr']
    # Iterate through the nested dictionary to get the last time step errors for each (k, epsilon) pair
    for k, eps_dict in resultsByKByEps.items():
        table_rows = []
        for eps, err_dict in eps_dict.items():
            row = {'epsilon': eps}
            for err_key, err_name in zip(errorKeys, errorNames):
                row[err_name] = err_dict[err_key][-1]  # Get the last time step error
            table_rows.append(row)

        # Create a DataFrame to display the table for each k
        df_k = pd.DataFrame(table_rows)
        
        # Sort by epsilon for easier reading
        df_k = df_k.sort_values(by=['epsilon'])
        
        # Store the table in the dictionary
        tables_by_k[k] = df_k
    return tables_by_k
    
    