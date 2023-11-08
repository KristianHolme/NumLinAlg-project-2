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

def getU0S0V0(A, k):
    U, S, Vh = np.linalg.svd(A)
    U = U[:,:k]
    S = np.diag(S)[:k, :k]
    V = Vh[:k, :].T
    return U, S, V

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

def get_true_solutions(u, timesteps, m, n):
    """
    Generates a list of true solution matrices evaluated at given timesteps.

    Parameters:
    - u: The function that returns the true solution when passed (x, y, t).
    - timesteps: A list or array of time values at which to evaluate the true solution.
    - m: The number of points in the x-dimension.
    - n: The number of points in the y-dimension.

    Returns:
    - A list of 2D numpy arrays containing the true solutions.
    """
    true_solutions = []

    for t in timesteps:
        true_solution_matrix = getSol(u, t, m, n)
        true_solutions.append(true_solution_matrix)

    return true_solutions

def g(x, y):
    return np.sin(np.pi*x)*np.sin(2*np.pi*y)
def u(x, y, t):
    return np.exp(-5*np.pi**2*t)*np.sin(np.pi*x)*np.sin(2*np.pi*y)

def makeAfuncs(n=10, eps=1e-3, cosMult=False):
    ones = np.ones(n**2 - 1)
    T1 = diags([ones, -1*ones], [-1, 1]).toarray()
    T2 = diags([ones[1:], 0.5*ones, -0.5*ones, -1*ones[1:]], [-2, -1, 1, 2]).toarray()
    #make A1
    np.random.seed(42)
    Ipert = np.identity(n) + np.random.rand(n, n)*0.5
    A1 = np.random.rand(n**2, n**2)*eps
    A1[0:n, 0:n] = Ipert
    #Make A2
    np.random.seed(99)
    Ipert = np.identity(n) + np.random.rand(n, n)*0.5
    A2 = np.random.rand(n**2, n**2)*eps
    A2[0:n, 0:n] = Ipert
    
    I = np.identity(n**2)
    def Q1(t): return expm(T1*t)@I
    def Q2(t): return expm(T2*t)@I
    def dQ1(t): return T1@(Q1(t))
    def dQ2(t): return T2@(Q2(t))
    
    if not cosMult:
        def A(t):
            return Q1(t)@(A1 + np.exp(t)*A2)@(Q2(t).T)
        
        def dA(t):
            Q1_t = Q1(t)
            Q2_t = Q2(t)
            dQ1_t = dQ1(t)
            dQ2_t = dQ2(t)
            
            exp_t_A2 = np.exp(t) * A2
            term1 = dQ1_t @ (A1 + exp_t_A2) @ Q2_t.T
            term2 = exp_t_A2 @ Q2_t.T
            term3 = (A1 + exp_t_A2) @ dQ2_t.T
                
            return term1 + Q1_t @ (term2 + term3)
            # return ( dQ1(t)@(A1 + np.exp(t)*A2)@(Q2(t).T) + 
            # Q1(t)@( np.exp(t)*A2@(Q2(t).T) + (A1 + np.exp(t)*A2)@(dQ2(t).T) ) )
        return A, dA
    else:
        def A(t):
            return Q1(t)@(A1 + np.cos(t)*A2)@(Q2(t).T)
        
        def dA(t):
            Q1_t = Q1(t)
            Q2_t = Q2(t)
            dQ1_t = dQ1(t)
            dQ2_t = dQ2(t)
            
            msin_t_A2 = -1*np.sin(t) * A2
            term1 = dQ1_t @ (A1 + msin_t_A2) @ Q2_t.T
            term2 = msin_t_A2 @ Q2_t.T
            term3 = (A1 + msin_t_A2) @ dQ2_t.T
                
            return term1 + Q1_t @ (term2 + term3)
            # return ( dQ1(t)@(A1 + np.exp(t)*A2)@(Q2(t).T) + 
            # Q1(t)@( np.exp(t)*A2@(Q2(t).T) + (A1 + np.exp(t)*A2)@(dQ2(t).T) ) )
        return A, dA
    
def GetTimeIntegrationResults(n=10, k=10, verbose=1, epss=[1e-3],
                              cay=cay1, h0=0.01, TOL=1e-3, maxTimeCuts=10,
                              t0=0, tf=1, cosMult=False):
    # epss = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    resultsByEps = {}
    for i, eps in enumerate(epss):
        A, dA = makeAfuncs(n, eps=eps, cosMult=cosMult)
        # N = 99
        #run timeintegration
        A0 = A(t0)
        U0, S0, V0 = getU0S0V0(A0, k)
        
        Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRun = TimeIntegration(
        t0, tf, h0, U0, S0, V0, dA, USVstep, cay=cay, verbose = verbose,
        TOL= TOL, maxTimeCuts=maxTimeCuts)
        Ylist = makeY(Ulist, Slist, Vlist)
        dYlist = makedY(Ulist, Slist, Vlist, dUlist, dSlist, dVlist)
        
        ts = timesteps
        # ts = np.arange(t0, tf, h0)
        b = np.random.rand(A0.shape[0])
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
    errorNames = [r"$||W(t)-A(t)||$",r"$||X(t)-A(t)||$",r"$||Y(t)-A(t)||$", r"$||X(t)-Y(t)||$", r"$||\dot{Y}(t)-\dot{A}(t)||$"]
    errorKeys = ['Werr', 'Xerr', 'Yerr', 'XYerr', 'dYerr']
    # Iterate through the nested dictionary to get the last time step errors for each (k, epsilon) pair
    for k, eps_dict in resultsByKByEps.items():
        table_rows = []
        for eps, err_dict in eps_dict.items():
            row = {r'$\epsilon$': eps}
            for err_key, err_name in zip(errorKeys, errorNames):
                row[err_name] = err_dict[err_key][-1]  # Get the last time step error
            table_rows.append(row)

        # Create a DataFrame to display the table for each k
        df_k = pd.DataFrame(table_rows)
        
        # Sort by epsilon for easier reading
        df_k = df_k.sort_values(by=[r'$\epsilon$'])
        
        # Store the table in the dictionary
        tables_by_k[k] = df_k
    return tables_by_k
    
def CompareRankkApproximation(ns = [10, 100, 1000], res=3):
    from plotting import PlotSVDTest
    for n in ns:
        k = n
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        _, singvals, _ = np.linalg.svd(A)
        WErr = np.zeros(n)
        WNErr = np.zeros(n)
        bestAppErr = np.zeros(n)
        POrthErr = np.zeros(n)
        QOrthErr = np.zeros(n)
        nonPOrthErr = np.zeros(n)
        nonQOrthErr = np.zeros(n)
        for k in range(1, n+1):
            I = np.identity(k)
            PkOrth, QkOrth, BkOrth= LanczosBidiag(A, k, b, orth=True)
            Worth = PkOrth@BkOrth@(QkOrth.T)
            PkNonOrth, QkNonOrth, BkNonOrth = LanczosBidiag(A, k, b, orth=False)
            WNonOrth = PkNonOrth@BkNonOrth@(QkNonOrth.T)
            X = bestApproxSVDRecon(A, k)
            WErr[k-1] = norm(Worth - A)
            WNErr[k-1] = norm(WNonOrth - A)
            bestAppErr[k-1] = norm(X-A)
            POrthErr[k-1] = norm(PkOrth.T@PkOrth -I)
            QOrthErr[k-1] = norm(QkOrth.T@QkOrth - I)
            nonPOrthErr[k-1] = norm(PkNonOrth.T@PkNonOrth - I)
            nonQOrthErr[k-1] = norm(QkNonOrth.T@QkNonOrth - I)
            
        PlotSVDTest(n, singvals, WErr, WNErr, bestAppErr, POrthErr, QOrthErr,
                    nonPOrthErr, nonQOrthErr, res=res)

def calculate_differences(approx_matrices, timesteps, u, m, n):
    """
    Given a list of approximated matrices and corresponding timesteps,
    computes the list of matrices representing the difference from the true solution.

    Parameters:
    - approx_matrices: List of 2D numpy arrays representing the approximated solution.
    - timesteps: List of times at which the approximated solutions are computed.
    - u: Function that provides the exact solution when called with x, y, t.
    - m: Number of points in the x dimension for the exact solution grid.
    - n: Number of points in the y dimension for the exact solution grid.

    Returns:
    - List of 2D numpy arrays, each representing the difference between the
      approximation and the true solution at a given timestep.
    """

    # Check for equal number of matrices and timesteps
    if len(approx_matrices) != len(timesteps):
        raise ValueError("Number of matrices and timesteps must match.")

    differences = []
    for i, approx_matrix in enumerate(approx_matrices):
        t = timesteps[i]
        true_matrix = getSol(u, t, m, n)
        difference = approx_matrix - true_matrix
        differences.append(difference)
    
    return differences
    