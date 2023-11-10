import matplotlib.pyplot as plt
import numpy as np
from utils import *
from matplotlib.animation import FuncAnimation
norm = np.linalg.norm

def plotRankApproxError(Ylist, u, times, k, needGetSol=True):
    from utils import getSol
    errors = np.zeros_like(times)
    m, n = (Ylist[0]).shape
    for i, t in enumerate(times):
        if needGetSol:
            sol = getSol(u, t, m, n)
        else:
            sol = u(t)
        errors[i] = norm(Ylist[i]-sol)
    
    plt.semilogy(times, errors)
    plt.title(f"Rank {k} approximation error norm")
    plt.xlabel(f"Time [s]")
    plt.ylabel(f"Error norm")
    plt.grid()
    plt.show()

def plotGrid3D(A):  
    """
    Plots 3D surface plot of data in the 2D Matrix A
    """
    A = np.flipud(np.rot90(A)) #because grid in project  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid with normalized coordinates
    x, y = np.meshgrid(np.linspace(0, 1, A.shape[0]), np.linspace(0, 1, A.shape[1]))

    # Plot the surface
    surf = ax.plot_surface(x, y, A, cmap='viridis')

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(np.min(A), np.max(A))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(surf)
    plt.show()
    
def plotGrid2D(A, title=""):
    """
    Plots a 2D image of the matrix A
    Parameters: 
    A: 2D matrix to be plotted
    title: plot title
    """
    fig = plt.figure()
    A = np.flipud(np.rot90(A)) #because of grid in project
    #flipud to match surface plot orientation
    plt.imshow(np.flipud(A), cmap='viridis', interpolation='nearest', extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title) if title else None
    plt.show()
    
def animateMatrix(A):
    fig, ax = plt.subplots()
    im = ax.imshow(A(0), animated=True, cmap='viridis')

    # Update function for animation
    def update(frame):
        im.set_array(A(frame))
        return [im]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), blit=True)
    ani.save("animated_matrix.gif", writer="imagemagick")
    plt.show()
    
def plotTimeIntegrationResults(ax, results, title):
    Werr = results['Werr']
    Xerr = results['Xerr']
    XYerr = results['XYerr']
    Yerr = results['Yerr']
    dYerr = results['dYerr']
    timesteps = results['timesteps']
    
    ax.plot(timesteps, Werr, linestyle='-', label="$||W_{\\perp}(t)-A(t)||$")
    ax.plot(timesteps, Xerr, linestyle='-', label="$||X(t)-A(t)||$")
    ax.plot(timesteps, Yerr, linestyle='-', label="$||Y(t)-A(t)||$")
    ax.plot(timesteps, XYerr, linestyle=':', label="$||X(t)-Y(t)||$")
    ax.plot(timesteps[0:-1], dYerr, linestyle='-.', label="$||\dot{Y}(t)-\dot{A}(t)||$")
    
    ax.legend()
    ax.set_xlabel(f"time [s]")
    ax.set_title(title)
    
def plotErrors(resultsByKByEps, ks, epss, ksToPlot, epssToPlot, res = 6):
    fig = plt.figure(constrained_layout=True, figsize=(ksToPlot*res, epssToPlot*res))
    fig.suptitle(f"Title")
    subfigs = fig.subfigures(nrows=1, ncols=ksToPlot)
    for row, subfig in enumerate(subfigs):
        k = ks[row]
        subfig.suptitle(f"k={k}", fontsize=14)
        axs = subfig.subplots(nrows=epssToPlot, ncols=1)
        for col, ax in enumerate(axs):
            eps = epss[col]
            plotTimeIntegrationResults(ax, resultsByKByEps[k][eps], f"$\epsilon$={eps}")
    # plt.tight_layout()
    plt.show()
    
def PlotSVDTest(n, singvals, WErr, WNerr, bestAppErr, POrthErr,
                QOrthErr, nonPOrthErr, nonQOrthErr,
                res=3):
    fig, axs = plt.subplots(1, 3, figsize=(3*res, 1*res))
    x = range(1,n+1)
    axs[0].semilogy(x, singvals, '.')
    axs[0].set_title(f"Singular values of ${n}\\times {n}$ test matrix")
    axs[0].set_xlabel("Singular value #")
    axs[0].grid()
    
    axs[1].plot(x, WErr, label=r"$||W_{\perp}-A||_F$")
    axs[1].plot(x, WNerr, label=r"$||W-A||_F$")
    axs[1].plot(x, bestAppErr, label=r"$||X-A||_F$")
    axs[1].legend()
    axs[1].set_title(f"Rank k approximation error")
    axs[1].set_xlabel("k")
    axs[1].grid()
    
    axs[2].plot(x, nonPOrthErr, label=r"$||P_k^T P_k - I||_F$")
    axs[2].plot(x, nonQOrthErr, label=r"$||Q_k^T Q_k - I||_F$")
    axs[2].plot(x, POrthErr, label=r"$||P_{\perp k}^T P_{\perp k} - I||_F$")
    axs[2].plot(x, QOrthErr, label=r"$||Q_{\perp k}^T Q_{\perp k} - I||_F$")
    axs[2].set_title(f"Non-orthogonality error for $P_k$ and $Q_k$")
    axs[2].set_xlabel("k")
    axs[2].legend()
    axs[2].grid()
    
    plt.tight_layout()
    plt.show()
    
def plotSVDComparison(Ylist, k, A, timesteps, skipparam=20):
    ts = timesteps[::skipparam]
    trueSing = np.zeros((k, len(ts)))
    YSing = np.zeros((k, len(ts)))
    for i, t in enumerate(ts):
        At = A(t)
        _, AS, _ = np.linalg.svd(At)
        trueSing[:, i] = AS[:k]
        j = timesteps.index(t)
        Y = Ylist[j]
        _, YS, _ = np.linalg.svd(Y)
        YSing[:, i] = YS[:k]
    for i in range(k):
        plt.plot(ts, trueSing[i, :])
        plt.plot(ts, YSing[i,:], 'o')
    plt.title(f'Singular values, k={k}')
    plt.xlabel("time [s]")
    plt.show()
    pass
    
    

"""
import matplotlib.pyplot as plt
plotGrid2D(U0@S0@(V0.T))
C
"""
    