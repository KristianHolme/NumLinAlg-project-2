import matplotlib.pyplot as plt
import numpy as np
from utils import *
norm = np.linalg.norm

def plotRankApproxError(Ylist, u, times, k):
    errors = np.zeros_like(times)
    m, n = (Ylist[0]).shape
    for i, t in enumerate(times):
        errors[i] = norm(Ylist[i]-getSol(u, t, m, n))
    
    plt.semilogy(times, errors)
    plt.title(f"Rank {k} approximation error norm")
    plt.xlabel(f"Time [s]")
    plt.ylabel(f"Error norm")
    plt.show()

"""
import matplotlib.pyplot as plt
plt.imshow(U0@S0@(V0.T))
"""
    