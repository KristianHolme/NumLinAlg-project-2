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
    b = np.random.rand(M)
    Pk, Qk, Bk = LanczosBidiag(A, k, b, usejit=usejit, orth=orth)
    Aprox = Pk@Bk@Qk.T
    err = np.linalg.norm(Aprox - A)
    if verbose:print(f"error:{err}")
    pass
def testBestSVD(M=2, N=2, k=2, verbose=1):
    t0 = time()
    A = np.random.rand(M, N)
    b = np.random.rand(M)
    X = bestApproxSVD(A, k)
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

def testCay(m=2, k=2):
  # Generate a random m x (m+k) matrix
    A = np.random.rand(m, m+k)

    # Perform QR factorization to get an orthogonal matrix Q
    Q, _ = np.linalg.qr(A)

    # Partition Q into F and U
    F = Q[:, :k]
    U = Q[:, k:2*k]
    I = np.identity(k)
    # Verify that U^T U = I and F^T U = 0
    print("U^T U - I norm=")
    print(np.linalg.norm((U.T @ U)-I))
    print("F^T U norm:")
    print(np.linalg.norm(F.T @ U))
    
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
    b = np.random.rand(N+1)
    if loop:
        err = np.zeros(N+1)
        for k in range(1, N+1 +1):
            Pk, Qk, Bk = LanczosBidiag(A, k, b, usejit=usejit, orth=orth)
            Aprox = Pk@Bk@Qk.T
            if verbose: print(f"k:{k}, approx rank:{np.linalg.matrix_rank(Aprox)}")
            err[k-1] = np.linalg.norm(Aprox - A)
            if verbose:print(f"error:{err[k-1]}")
    
        plt.plot(err)
        plt.show()
    else:
        Pk, Qk, Bk = LanczosBidiag(A, k, b, usejit=usejit, orth=orth)
        Aprox = Pk@Bk@Qk.T
        if verbose: print(f"k:{k}, approx rank:{np.linalg.matrix_rank(Aprox)}")
        err = np.linalg.norm(Aprox - A)
        if verbose:print(f"error:{err}")
        
    pass



def testODEsolver(N = 32, k=2):
    t0 = 0
    tf = 0.2
    h0 = 0.001
    A0 = initval(g, N)
    b = np.random.rand(A0.shape[0])
    Pk, Qk, Bk = LanczosBidiag(A0, k, b)
    U0, S0, V0 = Pk, Bk, Qk
    
    Ulist, Slist, Vlist, timesteps = TimeIntegration(t0, tf, h0, U0, S0, V0, 
                    diff2, linMatODEStep, 
                    cay=cay1,verbose = True,
                    TOL= 1e-3, maxTimeCuts=3)
    Ylist = makeY(Ulist, Slist, Vlist)
    plotRankApproxError(Ylist, u, timesteps, k)
    pass

def testAnim():
    A, dA = makeAfuncs()
    animateMatrix(A)
    pass
def testTimeInt(n = 10, k=10, eps=1e-3, TOL=1e-2, maxCuts=10):
    A, dA = makeAfuncs(n, eps=eps)
    N = 99
    t0 = 0
    tf = 1
    h0 = 0.1
    A0 = A(t0)
    b = np.random.rand(A0.shape[0])
    Pk, Qk, Bk = LanczosBidiag(A0, k, b)
    U0, S0, V0 = Pk, Bk, Qk
    
    Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRUn = TimeIntegration(
        t0, tf, h0, U0, S0, V0, dA, USVstep, cay=cay1, verbose = 2,
        TOL= TOL, maxTimeCuts=maxCuts)
    Ylist = makeY(Ulist, Slist, Vlist)
    plotRankApproxError(Ylist, A, timesteps, k, needGetSol=False)
    pass
def runTimeIntegrationex4(n=10, k=10, eps=1e-3):
    A, dA = makeAfuncs(n, eps=eps)
    # N = 99
    t0 = 0
    tf = 1
    h0 = 0.1
    A0 = A(t0)
    b = np.random.rand(A0.shape[0])
    Pk, Qk, Bk = LanczosBidiag(A0, k, b)
    U0, S0, V0 = Pk, Bk, Qk
    
    Ulist, Slist, Vlist, timesteps, dUlist, dSlist, dVlist, tRun = TimeIntegration(
        t0, tf, h0, U0, S0, V0, dA, USVstep, cay=cay1, verbose = 2,
        TOL= 1e-2, maxTimeCuts=10)
    Ylist = makeY(Ulist, Slist, Vlist)
    dYlist = makedY(Ulist, Slist, Vlist, dUlist, dSlist, dVlist)
    return Ylist, dYlist, timesteps, tRun
    


def ex4():
    epss = [1e-3, 1e-4]
    resultsByEps = GetTimeIntegrationResults(epss=epss)
    
    fig, axs = plt.subplots(1, 2)
    axs = axs.flatten()
    for i, eps in enumerate(epss):
        plotTimeIntegrationResults(axs[i], resultsByEps[eps], f"$\epsilon$={eps}")
    plt.tight_layout()
    plt.show()
    pass
    
# testLanczos()
# testCay(m=7*1000, k=5*370)
# testLanczos2(N=32, k=2, loop=False, orth=True)
# testODEsolver(N=32, k=1)
# testAnim()
testTimeInt(n=10, k=10, TOL=1e-3, maxCuts=10, eps=1e-6)
# testBestSVD(M = 800, N = 1200, k=230)
# ex4()
