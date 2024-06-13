import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def embed_velocity(X, velocity, embed_fn):
    dX = X + velocity
    V_emb = embed_fn(dX)
    X_embedding = embed_fn(X)
    dX_embed = V_emb - X_embedding

    return dX_embed

def arrow_grid(velos, data, pca, labels, cell_colors='black'):
    # If we pass in just a single data set of velocities, convert it to a list
    # so that all the comparison based code works
    if type(data) is not list and type(data) is not tuple:
        velos = [velos]
        data = [data]
    if type(velos) is not list and type(velos) is not tuple:
        velos = [velos]
        data = [data]
    X = data
    num_comparisons = len(velos)
    num_cells = velos[0].shape[1]
    if num_comparisons != len(labels):
        raise ValueError(f'Number of labels ({len(labels)}) must match number of comparisons ({num_comparisons})')
    
    # Compute the projection of gene expression onto the first two principal components
    proj = [np.array(pca.transform(x))[:,0:2] for x in X]
    # Find the extents of the projection
    minX = min([np.min(p[:,0]) for p in proj])
    maxX = max([np.max(p[:,0]) for p in proj])
    minY = min([np.min(p[:,1]) for p in proj])
    maxY = max([np.max(p[:,1]) for p in proj])
    # Add some buffer to the sides of the plot
    xbuf = (maxX - minX) * 0.05
    ybuf = (maxY - minY) * 0.05
    # Set the granularity of the grid, i.e. how many grids to divide the space into
    # in both the x and y directions
    n_points = 20
    x_grid_points = np.linspace(minX, maxX, n_points)
    y_grid_points = np.linspace(minY, maxY, n_points)
    # Find the width and height of each grid cell
    x_spacing = x_grid_points[1] - x_grid_points[0]
    y_spacing = y_grid_points[1] - y_grid_points[0]
    # This creates a sequential list of points defining the upper left corner of each grid cell
    grid = np.array(np.meshgrid(x_grid_points, y_grid_points)).T.reshape(-1,2)
    # Set up a list of velocities for each grid cell
    velocity_grid = np.zeros_like(grid)
    # This is nan, rather than zero so that we can distinguish between
    # grid cells with zero velocity and grid cells with 
    # no points inside, which wil be (nan)
    velocity_grid[:] = np.nan

    # Find points inside each grid cell
    mean_velocities = np.zeros((num_comparisons, len(grid), num_cells))
    variances = np.zeros((num_comparisons, len(grid), num_cells))
    mean_X = np.zeros((num_comparisons, len(grid), num_cells))

    for i,(x,y) in enumerate(grid):
        for j in range(num_comparisons):
            # Find the points inside the grid cell
            idx = np.where((proj[j][:,0] > x) & (proj[j][:,0] < x+x_spacing) & 
                           (proj[j][:,1] > y) & (proj[j][:,1] < y+y_spacing))[0]
            # If there are any points inside the grid cell
            if len(idx) > 0:
                # Get the average velocity vector for the points 
                # inside the grid cell
                velo = velos[j][idx,:]
                # var = vars[j][idx,:]
                # Compute the mean velocity vector of the points inside the grid cell
                mean_velocities[j,i] = velo.mean(axis=0).reshape(-1)
                # variances[j,i] = var.mean(axis=0).reshape(-1)
                mean_X[j,i] = X[j][idx,:].mean(axis=0)
            
    # variances = util.tonp(variances)
    pca_embed = lambda x: np.array(pca.transform(x)[:,0:2])
    velocity_grid = [embed_velocity(x, v, pca_embed) for x,v in zip(mean_X, mean_velocities)]
    
    fig, ax = plt.subplots(1,1, figsize=(10,10))

    ax.set_xlim(minX-xbuf, maxX+xbuf)
    ax.set_ylim(minY-ybuf, maxY+ybuf)
    ax.set_facecolor('#edf2f4')
    # Add lines to show the grid
    for x in x_grid_points:
        ax.axvline(x, color='white', alpha=.7)
    for y in y_grid_points:
        ax.axhline(y, color='white', alpha=.7)
    #Plot the velocity vectors
    ax.scatter(proj[0][:,0], proj[0][:,1], c=cell_colors, s=.7, alpha=.2)
    for j in range(num_comparisons):
        for i in range(len(grid)):
            velo = velocity_grid[j][i,:]
            # If the velocity is zero, don't plot it
            if np.abs(velo).sum() == 0:
                continue
            x,y = grid[i,:]
            # Plot an arrow in the center of the grid cell, 
            # pointing in the direction of the velocity
            # TODO change color from sequence from tab20c
            arrow = ax.arrow(x + x_spacing/2, y + y_spacing/2, velo[0], velo[1], 
                             width=0.01, head_width=0.07, head_length=0.05, 
                             color='black', alpha=.1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    # Manually specify handles and labels for the legend
    ax.legend([matplotlib.patches.Arrow(0,0,0,0, color='black', width=.1) 
               for i in range(num_comparisons)],
              labels)
    ax.set_title(f'{" vs ".join([label.capitalize() for label in labels])}', fontsize=14);
