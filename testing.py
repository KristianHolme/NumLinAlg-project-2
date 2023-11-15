from lanczos import *
import numpy as np
import time
from functools import wraps
from cayley import *
from heatequation import *
from timeIntegration import *
from utils import *
from plotting import *
import matplotlib.pyplot as plt
from scipy.linalg import expm

norm = np.linalg.norm

def plsgofast(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

@plsgofast
def runLanczos(M=2, N=2, k=2, usejit=True, orth=True, verbose=False):
    A = np.random.rand(M, N)
    Pk, Qk, Bk = LanczosBidiag(A, k, usejit=usejit, orth=orth)
    Aprox = Pk@Bk@Qk.T
    err = np.linalg.norm(Aprox - A)
    if verbose:print(f"error:{err}")
    pass
def testBestSVD(M=2, N=2, k=2, verbose=1):
    t0 = time()
    A = np.random.rand(M, N)
    X = bestApproxSVDRecon(A, k)
    err = np.linalg.norm(X - A)
    tend = time() - t0
    if verbose:print(f"error:{err}, time:{tend}")
    pass

def testLanczos(N=5, k=3, orth=True):
    #small case to compile
    runLanczos(N=2, k=2, orth=orth)

    #without jit
    runLanczos(M = 1500, N=2000, k=400, usejit=False, orth=orth)

    #test with jit
    runLanczos(M=1500, N=2000, k=400, orth=orth)

def testCay(m=5, k=2):
    # k must be under half of m
    # Generate a random m x (m+k) matrix
    A = np.random.rand(m, m+k)

    # Perform QR factorization to get an orthogonal matrix Q
    Q, _ = np.linalg.qr(A, mode='complete')

    # Partition Q into F and U
    F = Q[:, :k]
    U = Q[:, k:2*k]
    I = np.identity(k)
    # Verify that U^T U = I and F^T U = 0
    # print("U^T U - I norm=")
    # print(np.linalg.norm((U.T @ U)-I))
    # print("F^T U norm:")
    # print(np.linalg.norm(F.T @ U))
    
    CDir = cayDirect(U, F, verbose=True)
    C1 = cay1(U, F, verbose=True)
    CQR = cayQR(U, F, verbose=True)
    
    C1err = norm(C1-CDir)
    CQRerr = norm(CQR - CDir)
    print(f"C1err: {C1err}")
    print(f"CQR err: {CQRerr}")
    
    pass

def testLanczos2(N=2, k=2, usejit=False, orth=True, verbose=True, loop=False):
    A = np.random.rand(N+1, N+1)
    A = initval(g, N)
    if verbose:print(f"Rank A:{np.linalg.matrix_rank(A)}")
    if loop:
        err = np.zeros(N+1)
        for k in range(1, N+1 +1):
            Pk, Qk, Bk = LanczosBidiag(A, k, usejit=usejit, orth=orth)
            Aprox = Pk@Bk@Qk.T
            if verbose: print(f"k:{k}, approx rank:{np.linalg.matrix_rank(Aprox)}")
            err[k-1] = np.linalg.norm(Aprox - A)
            if verbose:print(f"error:{err[k-1]}")
    
        plt.plot(err)
        plt.show()
    else:
        Pk, Qk, Bk = LanczosBidiag(A, k, usejit=usejit, orth=orth)
        Aprox = Pk@Bk@Qk.T
        if verbose: print(f"k:{k}, approx rank:{np.linalg.matrix_rank(Aprox)}")
        err = np.linalg.norm(Aprox - A)
        if verbose:print(f"error:{err}")
        
    pass



def testODEsolver(N = 32, k=1):
    t0 = 0
    tf = 0.2
    h0 = 0.0005
    A0 = initval(g, N)
    U0, S0, V0 = getU0S0V0(A0, k)
    TOL = 1e-5
    maxcuts = 10
    m = n = N+1
    
    Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRUn= TimeIntegration(t0, tf, h0, U0, S0, V0, 
                    diff, linMatODEStep, 
                    cay=cay1, verbose = 3,
                    TOL= TOL, maxTimeCuts=maxcuts)
    Ylist = makeY(Ulist, Slist, Vlist)
    plotRankApproxError(Ylist, u, timesteps, k)
    animtime = 2
    animate_matrices(Ylist, timesteps, animtime)
    diffs = calculate_differences(Ylist, timesteps, u, m, n)
    animate_matrices(diffs, timesteps, animtime)
    truesol = get_true_solutions(u, timesteps, m, n)
    animate_matrices(truesol, timesteps, animtime, updateCbar=False)
    pass

def testAnim():
    A, dA = makeAfuncs()
    animateMatrix(A)
    pass

def testTimeInt(n = 10, k=10, eps=1e-3, TOL=1e-2, maxCuts=10, cosMult=False,
                tf=1, h0=0.1):
    A, dA = makeAfuncs(n, eps=eps, cosMult=cosMult)
    N = 99
    t0 = 0
    A0 = A(t0)
    U0, S0, V0 = getU0S0V0(A0, k)
    
    Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRUn = TimeIntegration(
        t0, tf, h0, U0, S0, V0, dA, USVstep, cay=cay1, verbose = 2,
        TOL= TOL, maxTimeCuts=maxCuts)
    Ylist = makeY(Ulist, Slist, Vlist)
    plotRankApproxError(Ylist, A, timesteps, k, needGetSol=False)
    pass

def runTimeIntegrationex4(n=10, k=10, eps=1e-3, cay=cay1):
    A, dA = makeAfuncs(n, eps=eps)
    # N = 99
    t0 = 0
    tf = 1
    h0 = 0.1
    A0 = A(t0)
    U0, S0, V0 = getU0S0V0(A0, k)
    
    Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRun = TimeIntegration(
        t0, tf, h0, U0, S0, V0, dA, USVstep, cay=cay, verbose = 2,
        TOL= 1e-2, maxTimeCuts=10)
    Ylist = makeY(Ulist, Slist, Vlist)
    dYlist = makedY(Ulist, Slist, Vlist, dUlist, dSlist, dVlist)
    return Ylist, dYlist, timesteps, tRun
    
def ex4(n=10, k=10):
    epss = [1e-3, 1e-1]
    resultsByEps = GetResults(n=n, k=k, epss=epss)
    
    fig, axs = plt.subplots(1, 2)
    axs = axs.flatten()
    for i, eps in enumerate(epss):
        plotErrorsToAx(axs[i], resultsByEps[eps], f"$\epsilon$={eps}")
    plt.tight_layout()
    plt.show()
    pass
    
def ex5(n=10, ks=[10], numPoints=30, TOL=1e-1, maxcuts=5, verbose=1, eps=1e-1):
    n= 10
    for k in ks:
        A, dA = makeAfuncs(n, eps=eps, cosMult=True)
        # N = 99
        t0 = 0
        tf = 10
        h0 = 0.1
        A0 = A(t0)
        U0, S0, V0 = getU0S0V0(A0, k)
        
        Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRun = TimeIntegration(
            t0, tf, h0, U0, S0, V0, dA, USVstep, cay=cay1, verbose = verbose,
            TOL= 1e-1, maxTimeCuts=maxcuts)
        Ylist = makeY(Ulist, Slist, Vlist)
        plotSVDComparison(Ylist, k, A, timesteps, numPoints=numPoints)
        pass
        
def testODEsolverSimple(N = 32, k=33):
    t0 = 0
    tf = 0.2
    h0 = 0.01
    TOL = 1e-5
    maxcuts = 10
    m = n = N+1
    A = np.random.rand(N+1, N+1)
    A0 = np.random.rand(N+1, N+1)
    f = lambda m, n, t: expm(-A*t)@A0
    df = lambda m, n: (-A, 0*A)
    U0, S0, V0 = getU0S0V0(A0, k)
    
    Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRUn= TimeIntegration(t0, tf, h0, U0, S0, V0, 
                    df, linMatODEStep, 
                    cay=cay1, verbose = 3,
                    TOL= TOL, maxTimeCuts=maxcuts)
    Ylist = makeY(Ulist, Slist, Vlist)
    plotRankApproxError(Ylist, f, timesteps, k)
    animtime = 2
    animate_matrices(Ylist, timesteps, animtime)
    diffs = calculate_differences(Ylist, timesteps, f, m, n)
    animate_matrices(diffs, timesteps, animtime)
    truesol = get_true_solutions(f, timesteps, m, n)
    animate_matrices(truesol, timesteps, animtime, updateCbar=False)
    pass
# testLanczos()
# testCay(m=30*100, k=29*37)
# testLanczos2(N=32, k=2, loop=False, orth=True)
# testODEsolver(N=32, k=1)
# testAnim()
L = 5
# testTimeInt(n=L, k=L**2, TOL=1e-2, maxCuts=10, eps=1e-3)
# testBestSVD(M = 800, N = 1200, k=230)
# runTimeIntegrationex4(n=4, k=4)
# CompareRankkApproximation(ns=[8])
# ex5(n=4, ks=[5])
# testODEsolv[erSimple()
# testODEsolver(k = 1)
# testTimeInt(n=10, k=10, eps=1e-1, TOL=1e-1, maxCuts=3, cosMult=True, tf=10, h0=0.1)
# ex4(k=10)
# ks = [2**n for n in range(5, 10 + 1)]
# dirtime, C1time, CQRtime, C1err, CQRerr = runCay(ks)
# plotCayComp(dirtime, C1time, CQRtime, ks)
ks = list(range(3, 110, 10))
dirtime, C1time, CQRtime, C1err, CQRerr = runCay(ks)
plotCayComp(dirtime, C1time, CQRtime, ks)
pass
