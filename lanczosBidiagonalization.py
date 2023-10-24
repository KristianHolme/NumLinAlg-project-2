import numpy as np
import scipy.sparse as scsp
norm = np.linalg.norm
from numba import jit


def LanczosBidiag(A, k, b, usejit=True):
    if usejit:
        u, v, alfa, beta = LanczosBidiagMain(A, k, b)
    else:
        u, v, alfa, beta = LanczosBidiagMain.py_func(A, k, b)
    Pk = u.T
    Qk = v.T
    Bk = scsp.diags([alfa, beta[1:]], [0, -1])
    return Pk, Qk, Bk

@jit(nopython=True)
def LanczosBidiagMain(A, k, b):
    n = np.shape(A)[0]
    v = np.zeros((k, n)) #row k is vk
    u = np.zeros((k, n))
    alfa = np.zeros(k)
    beta = np.zeros(k)
    beta[0] = norm(b)
    u[0] = b/beta[0]
    ATu = A.T@u[0]
    alfa[0] = norm(ATu)
    v[0] = ATu/alfa[0]
    for i in range(0, k-1): # i is python index
        Avi_min_alfaui = A@v[i] - alfa[i]*u[i]
        beta[i+1] = norm(Avi_min_alfaui)
        u[i+1] = Avi_min_alfaui/beta[i+1]
        
        w = A.T@u[i+1] - beta[i+1]*v[i]
        for l in range(0, i):
            w = w - (v[l].T@w)*v[l]
            
        alfa[i+1] = norm(w)
        v[i+1] = w/alfa[i+1]
    return u, v, alfa, beta