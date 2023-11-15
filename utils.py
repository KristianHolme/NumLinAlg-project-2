import numpy as np
from scipy.sparse import diags
from scipy.linalg import expm
from lanczos import *
from cayley import *
from heatequation import *
from plotting import *
from timeIntegration import *
import pandas as pd
from cayley import *
norm = np.linalg.norm

def getU0S0V0(A, k):
    """ Computes Initial values of U, S and V for time integration
    Computes the initial values using a best approximation SVD of the initial value of the approximand A
    
    Args:
    A (numpy array): A(0) when A(t) is the approximand for the dynamic approximation
    k (int): rank of approximation
    
    returns:
    U, S, V (numpy array): Matrices such that USV^T is the best rank k approximation of A
    """
    U, S, Vh = np.linalg.svd(A)
    U = U[:,:k]
    S = np.diag(S)[:k, :k]
    V = Vh[:k, :].T
    return U, S, V

def makeY(U, S, V):
    """Assembles U, S V into Y
    Computes Y = USV^T 
    
    Args:
    U, S, V ((lists of) numpy arrays)
    
    Returns:
    Y ((list of) numpy array(s))
    
    """
    if type(U) is list:
        Ylist = [U@S@(V.T) for U, S, V in zip(U, S, V)]
        return Ylist
    else:
        Y = U@S@(V.T)
        return Y

def makedY(U, S, V, dU, dS, dV):
    """Computes the derivative of Y
    Assembles U, S V and their derivatives into the derivative of Y
    
    Args:
    U, S, V ((lists of) numpy arrays)
    dU, dS, dV ((lists of) numpy arrays)
    
    Returns:
    dY ((list of) numpy array(s))
    
    """
    assert len(U) == (len(dU)+1)
    if type(dU) is list:
        dYlist = [dU@S@(V.T) + U@dS@(V.T) + U@S@(dV.T) for U, S, V, dU, dS, dV in zip(U, S, V, dU, dS, dV)]
        return dYlist
    else:
        dY = dU@S@(V.T) + U@dS@(V.T) + U@S@(dV.T)
        return dY

def getSol(u, t, m, n):
    """Evaluate the function u(x, y, t) at time t on [0,1]x[0,1]
    
    Args: 
    u (function: args: x, y, t):
    t (float): time at which to evaluate u
    m, n (int): dimensions of matrix
    
    Returns
    S (numpy array: matrix evaluation of u)
    """
    #eval function u(x,y,t) as matrix at time t
    def fn(x, y):
        return u(x, y, t)
    x = np.linspace(0,1,m)
    y = np.linspace(0,1,n)
    
    s = fn(x[:, None], y[None, :])    
    return s

def runCay(ks, average=False):
    """
    Run three different algorithms for computing the cayley map, and collects
    runtimes
    
    Args:
    ks: (list of ints) the random matrices U, F used for computation are
    3k x k matrices
    average (bool): if true runs the computations 200 times to get a representative average. Use when k is low.
    
    Returns:
    dirtime (numpy array): direct alg computation times for each value of k
    c1time (numpy array): method 2 alg computation times for each value of k
    cqrtime (numpy array): QR alg computation times for each value of k
    """
    
    
    ms = [3*k for k in ks]
    l = len(ks)    
    dirtime = np.zeros(l)
    c1time = np.zeros(l)
    cqrtime = np.zeros(l)
    if average:
        r = 200
    else:
        r = 1
    for i, k in enumerate(ks):
        for j in range(1, r+1):
    
            m = ms[i]
            A = np.random.rand(m, m+k)

            # perform qr factorization to get an orthogonal matrix q
            Q, _ = np.linalg.qr(A, mode='complete')
            F = Q[:, :k]
            U = Q[:, k:2*k]
            t0 = time()
            cdir = cayDirect(U, F, verbose=False)
            t1 = time()
            dirtime[i] += (t1 - t0)/r
            c1 = cay1(U, F, verbose=False)
            t2 = time()
            c1time[i] += (t2-t1)/r
            cqr = cayQR(U, F, verbose=False)
            cqrtime[i] += (time() - t2)/r
    return dirtime, c1time, cqrtime
        
def get_true_solutions(u, timesteps, m, n):
    """
    generates a list of true solution matrices evaluated at given timesteps.

    Args:
    u (function): the function that returns the true solution when passed (x, y, t).
    timesteps (list): a list or array of time values at which to evaluate the true solution.
    m (int): the number of points in the x-dimension.
    n (int): the number of points in the y-dimension.

    returns:
    true_solutinos (list of numpy arrays) the true solutions.
    """
    true_solutions = []

    for t in timesteps:
        true_solution_matrix = getSol(u, t, m, n)
        true_solutions.append(true_solution_matrix)

    return true_solutions

def g(x, y):
    """Initial value for the heat equation problem"""
    return np.sin(np.pi*x)*np.sin(2*np.pi*y)

def u(x, y, t):
    """Exact solution for the hat equation problem"""
    return np.exp(-5*np.pi**2*t)*np.sin(np.pi*x)*np.sin(2*np.pi*y)

def makeAfuncs(n=10, eps=1e-3, cosMult=False):
    """ Computes A and its derivative for dynamic approximation
    Builds A as described in the project desription and in the paper by Koch and Lubich
    
    Args:
    n (int): A will be a n^2 x n^2 matrix
    eps (float): noise parameter
    cosMult (bool): if True, will compute A and the derivative with cos(t) in place of e^t
    
    Returns:
    A (function: arg: (t)) function that evaluates A at time given time, as n^2 x n^2 numpy array
    dA (function: arg: (t)) function that evaluates the derivative of A at time given time, as n^2 x n^2 numpy array
    """
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
    def Q1(t): return expm(T1*t)
    def Q2(t): return expm(T2*t)
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
            cos_t_A2 = np.cos(t)*A2
            term1 = dQ1_t @ (A1 + cos_t_A2) @ Q2_t.T
            term2 = msin_t_A2 @ Q2_t.T
            term3 = (A1 + cos_t_A2) @ dQ2_t.T
                
            return term1 + Q1_t @ (term2 + term3)
            # return ( dQ1(t)@(A1 + np.exp(t)*A2)@(Q2(t).T) + 
            # Q1(t)@( np.exp(t)*A2@(Q2(t).T) + (A1 + np.exp(t)*A2)@(dQ2(t).T) ) )
        return A, dA
    
def GetResults(n=10, k=10, verbose=1, epss=[1e-3],
                              cay=cay1, h0=0.01, TOL=1e-3, maxTimeCuts=10,
                              t0=0, tf=1, cosMult=False):
    """Runs dynamic-, Lanczos- and best approximation algorithms and compiles results and errors
    
    Args:
    n (int): square root of problem size
    k (int): rank of approximations
    verbose (int/bool) level of information displayed
    epss (list) list of different epsilons used to create A and dA
    cay (function): function that computes the cayley transformation from matries U and F
    h0 (float): initial step length target
    TOL (float): tolerance for time step control
    maxTimeCuts (int): maximum number of allowed time step cuts in the dynamic algorithm
    t0 (float): starting time
    tf (float): end time
    cosMult (bool): True if we want cosine version of A
    
    Returns:
    resultsByEps (dict): dictionary of dictionaries (one for each epsilon) 
    containing errors, runtimes and timesteps for dynamic approximation
    """
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
        W, Wtime = LanczosOverTime(ts, A, k, verbose=verbose)
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
    """Make tables of errors
    
    Args:
    resultsByKByeps (dict): dict of results from GetResults, one result dict for each k value desired
    
    Returns:
    tables_by_k (dict: int -> pandas dataframe) a pandas dataframe of epsilons and errors for each k
    """
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
    """Compare Lanczos with and without reorth. and the best approximation
    
    Runs computations and plots singular values for the random approximand A, 
    approximation errors and orthogonality errors.
    
    Args:
    ns (list): list of different values of matrix sizes n to approximate
    res (float): resolution for plotting
    """
    from plotting import PlotSVDTest
    for n in ns:
        k = n
        A = np.random.rand(n, n)
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
            PkOrth, QkOrth, BkOrth= LanczosBidiag(A, k, orth=True)
            Worth = PkOrth@BkOrth@(QkOrth.T)
            PkNonOrth, QkNonOrth, BkNonOrth = LanczosBidiag(A, k, orth=False)
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
    approx_matrices: List of 2D numpy arrays representing the approximated solution.
    timesteps: List of times at which the approximated solutions are computed.
    u: Function that provides the exact solution when called with x, y, t.
    m: Number of points in the x dimension for the exact solution grid.
    n: Number of points in the y dimension for the exact solution grid.

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

def RunSVComparison(n=10, ks=[10], numPoints=30, TOL=1e-1, maxcuts=5, verbose=1,
                    eps=1e-1, cosMult=True, tf=10):
    """
    Runs dynamic approximation, calculates the singular values of the approximatino and plots the 
    approximated singular values against the true singular values.
    
    Args: 
    n (int): square root of matrix size
    ks (list): different values of k(approximation rank)
    numPoints (int): number of timepoints to plot the SV's
    TOL (float):, tolerance for time step control
    maxcuts (int): maximum number of timestep cuts in time step control
    verbose (int/bool): level of info printed
    eps (float): parameter for constructing A and dA
    cosMult (bool): wheter or not to make the cosine version of A and dA
    tf (float): end time
    """
    from plotting import plotSVDComparison
    for k in ks:
        A, dA = makeAfuncs(n, eps=eps, cosMult=cosMult)
        # N = 99
        t0 = 0
        h0 = 0.1
        A0 = A(t0)
        U0, S0, V0 = getU0S0V0(A0, k)
        
        Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRun = TimeIntegration(
            t0, tf, h0, U0, S0, V0, dA, USVstep, cay=cay1, verbose = verbose,
            TOL= TOL, maxTimeCuts=maxcuts)
        Ylist = makeY(Ulist, Slist, Vlist)
        plotSVDComparison(Ylist, k, A, timesteps, numPoints=numPoints)
    