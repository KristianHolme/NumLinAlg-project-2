from lanczos import *
import numpy as np
import time
from functools import wraps


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

def testLanczos(N=5, k=3, orth=True):
    #small case to compile
    runLanczos(N=2, k=2, orth=orth)

    #without jit
    runLanczos(M = 1500, N=2000, k=400, usejit=False, orth=orth)

    #test with jit
    runLanczos(M=1500, N=2000, k=400, orth=orth)

testLanczos()