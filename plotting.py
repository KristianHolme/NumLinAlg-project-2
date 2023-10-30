import matplotlib.pyplot as plt
import numpy as np
from utils import *
from matplotlib.animation import FuncAnimation
norm = np.linalg.norm

def plotRankApproxError(Ylist, u, times, k, needGetSol=True):
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

"""
import matplotlib.pyplot as plt
plotGrid2D(U0@S0@(V0.T))
C
"""
    