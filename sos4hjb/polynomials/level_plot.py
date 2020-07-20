import numpy as np
import matplotlib.pyplot as plt

def level_plot(y, x_min, x_max, label=None, title=None, file_name=None):
    
    # Grid state space.
    n = 101
    x1 = np.linspace(x_min[0], x_max[0], n)
    x2 = np.linspace(x_min[1], x_max[1], n)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Evaluate function on grid.
    Y = np.empty(X1.shape)
    for i in range(n):
        for j in range(n):
            xij = np.array([X1[i, j], X2[i, j]])
            Y[i, j] = y(dict(zip(y.variables(), xij)))
            
    # Plot the function.
    contours = plt.contour(X1, X2, Y, colors='black')
    plt.clabel(contours, fontsize=8)
    extent = np.vstack((x_min, x_max)).T.flatten()
    plt.imshow(np.flip(Y, axis=1), extent=extent, cmap='RdGy', alpha=.5)

    # Color bar.
    cbar = plt.colorbar()
    if label is not None:
        cbar.set_label(label)
    
    # Miscelaneous options.
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title)

    # Save image if requested.
    if file_name is not None:
        plt.savefig(file_name + '.pdf', bbox_inches='tight')
