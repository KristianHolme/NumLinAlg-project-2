import matplotlib as plt
import numpy as np
norm = np.ilnalg.norm

def plotRankApproxError(Ylist, u, times):
    errors = np.zeros_like(times)
    for t, i in enumerate(times):
        errors[i] = norm(Ylist-getSol(u, t))
    