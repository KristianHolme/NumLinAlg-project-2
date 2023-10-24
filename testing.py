from lanczosBidiagonalization import *
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
def runLanczos(N=5, k=3, usejit=True):
    A = np.random.rand(N, N)
    b = np.random.rand(N)
    Pk, Qk, Bk = LanczosBidiag(A, k, b, usejit=usejit)
    Aprox = Pk@Bk@Qk.T
    err = np.linalg.norm(Aprox - A)
    print(f"error:{err}")
    pass

def testLanczos(N=5, k=3, usejit=True):
    #small case to compile
    testLanczos(N=2, k=2)

    #without jit
    testLanczos(N=2000, k=400, usejit=False)

    #test with jit
    testLanczos(N=2000, k=400)
