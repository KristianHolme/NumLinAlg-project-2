import numpy as np
from time import time

def cayDirect(U, F, verbose=False):
    t0 = time()
    C = np.block([F, -U])
    D = np.block([U, F])
    B = C@(D.T)
    m, n = B.shape
    assert m==n
    I = np.identity(n)
    t1 = time()
    inv = np.linalg.inv((I - 0.5*B))
    invTime = time() - t1
    if verbose: print(f"Inverting matrix of size {m} took {invTime}s.")
    cay =  inv@(I+0.5*B)
    endTime = time() - t0
    if verbose: print(f"CayDirect with m:{m} took {endTime}s\n")
    return cay


def cay1(U, F, verbose = False):
    t0 = time()
    m, k = U.shape
    C = np.block([F, -U])
    D = np.block([U, F])
    assert C.shape == D.shape, "C and D have different dimensions!"
    I2k = np.identity(2*k)
    Im = np.identity(m)
    t1 = time()
    inverseTerm = np.linalg.inv(I2k-0.5*(D.T)@C)
    invTime = time() - t1
    if verbose: print(f"Inverting matrix of size {2*k} took {invTime}s.")
    
    cay = Im + C@inverseTerm@(D.T)
    endTime = time() - t0
    if verbose: print(f"Cay1 with m:{m} took {endTime}s\n")
    return cay

def cayQR(U, F, verbose = False, customInvert=True):
    t0 = time()
    m, k = U.shape
    assert  U.shape == F.shape
    p = 2*k
    Im = np.identity(m)
    I2k = np.identity(2*k)
    Q, R = np.linalg.qr(F)
    assert Q.shape == (m, k)
    assert R.shape == (k, k)
    RBlock = np.block([[np.zeros_like(R), -R.T], [R, np.zeros_like(R)]])
    t1 = time()#to time inversion
    

    if not customInvert:
        invertedBlock = np.linalg.inv(I2k - 0.5*RBlock)
    else:
        # Compute (I + 0.25 R R^T)^{-1}
        inv_term = np.linalg.inv(np.eye(R.shape[1]) + 0.25 * R.dot(R.T))

        # Compute M^{-1} using the simplified formula
        invertedBlock = np.block([
        [np.eye(R.shape[0]) + 0.5 * R.T.dot(inv_term).dot(-0.5 * R), -0.5 * R.T.dot(inv_term)],
        [-inv_term.dot(-0.5 * R), inv_term]
        ])
    invTime = time() - t1
    G = RBlock@invertedBlock
    if verbose: print(f"Inverting matrix of shape {G.shape} took {invTime}s.")
    UQ = np.block([U, Q])
    cay = Im + UQ@G@UQ.T
    
    endTime = time() - t0
    if verbose: print(f"CayQR with m, k:{m, k} took {endTime}s\n")
    return cay
    
    