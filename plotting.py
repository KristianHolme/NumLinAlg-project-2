import matplotlib.pyplot as plt
import numpy as np
from utils import *
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.lines as mlines
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
    
def animateMatrix(A, t0=0, tf=1, length=2, fps=60, name="animated_matrix"):
    fig, ax = plt.subplots(figsize=(5,5))
    plt.tight_layout()
    im = ax.imshow(A(0), animated=True, cmap='viridis')
    ax.axis('off')
    # Update function for animation
    def update(frame):
        im.set_array(A(frame))
        return [im]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.linspace(t0, tf, length*fps), blit=True)
    ani.save(name + ".gif", writer="Pillow", fps=fps)
    # plt.show()
    plt.close(fig)
    return name + ".gif"
    
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
    
def plotErrors(resultsByKByEps, ks, epss, ksToPlot, epssToPlot, res = 6, title=''):
    fig = plt.figure(constrained_layout=True, figsize=(ksToPlot*res, epssToPlot*res))
    fig.suptitle(title)
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
    
def plotSVDComparison(Ylist, k, A, timesteps, numPoints=50):
    skipparam = round(len(timesteps)/numPoints)
    ts = timesteps[::skipparam]
    trueSing = np.zeros((k, len(ts)))
    YSing = np.zeros((k, len(ts)))
    colors = iter(plt.cm.viridis(np.linspace(0, 1, k))) 
    for i, t in enumerate(ts):
        At = A(t)
        _, AS, _ = np.linalg.svd(At)
        trueSing[:, i] = AS[:k]
        j = timesteps.index(t)
        Y = Ylist[j]
        _, YS, _ = np.linalg.svd(Y)
        YSing[:, i] = YS[:k]
    for i in range(k):
        color = next(colors)
        plt.plot(ts, trueSing[i, :], color=color, lw=1)
        plt.plot(ts, YSing[i,:], 'o', color=color, markersize=2)
    
    true_sv_legend = mlines.Line2D([], [], color='black', marker='_', markersize=10, label='True SV')
    approx_sv_legend = mlines.Line2D([], [], color='black', marker='o',linestyle='None', markersize=4, label='Approx SV')
    plt.legend(handles=[true_sv_legend, approx_sv_legend])
     
    plt.title(f'Singular values, k={k}')
    plt.xlabel("time [s]")
    plt.show()
    pass
    
def animate_matrices(matrices, timesteps, total_anim_time, updateCbar = True):
    """
    Animates a list of matrices as images with a dynamic colorbar and a timestamp.

    Parameters:
    - matrices: A list of 2D numpy arrays.
    - timesteps: A list of time values corresponding to each matrix.
    - total_anim_time: Total animation time in seconds.
    """
    
    if not all(isinstance(matrix, np.ndarray) for matrix in matrices):
        raise ValueError('All items in the list must be numpy arrays.')
    if len(matrices) != len(timesteps):
        raise ValueError('The length of timesteps must match the number of matrices.')

    # Calculate the evenly distributed times for the animation frames
    animation_times = np.linspace(timesteps[0], timesteps[-1], int(60 * total_anim_time))
    
    # Find the indices in `timesteps` that are closest to the target `animation_times`
    indices_to_use = np.searchsorted(timesteps, animation_times)
    indices_to_use = np.clip(indices_to_use, 0, len(timesteps) - 1)  # Ensure indices are within bounds
    
    # Calculate the interval
    interval = (total_anim_time * 1000) / len(animation_times)

    fig, ax = plt.subplots()
    im = ax.imshow(matrices[0], animated=True)
    cbar = plt.colorbar(im, ax=ax)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')

    def update_fig(frame_index):
        # Use the precomputed index to ensure the frames are displayed at the correct time
        idx = indices_to_use[frame_index]
        data = matrices[idx]
        im.set_array(data)
        time_text.set_text(f'Time: {timesteps[idx]:.5f}/{timesteps[-1]:.5f}')
        
        # Update colorbar
        if updateCbar:
            im.set_clim(data.min(), data.max())
            cbar.update_normal(im)
        

        return im, time_text

    ani = FuncAnimation(fig, update_fig, frames=len(animation_times), interval=interval, blit=False)
    plt.show()
    return ani

def plotCols(U, V, t, res=3):
    fig, axs = plt.subplots(1, 2, figsize=(2*res, 2*res))
    axs[0].imshow(U, aspect='auto')
    axs[0].set_title(f"U")
    axs[0].set_xticks([0], range(1, U.shape[1]+1))
    axs[0].set_xlabel("i")
    axs[0].set_yticks([])
    
    axs[1].imshow(V, aspect='auto')
    axs[1].set_xticks([0], range(1, V.shape[1]+1))
    axs[1].set_xlabel("i")
    axs[1].set_yticks([])
    axs[1].set_title(f"V")
    
    fig.suptitle(f"Columns of U and V at t:{t:.5f}")
    
    plt.tight_layout()
    
    pass
"""
import matplotlib.pyplot as plt
plotGrid2D(U0@S0@(V0.T))
C
"""
    