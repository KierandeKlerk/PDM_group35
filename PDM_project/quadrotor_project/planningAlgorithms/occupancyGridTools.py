import numpy as np
import trimesh
import matplotlib.pyplot as plt

def generateOccupancyGrid(pathTo3DFile, file_type = None, pitch=0.05):
    '''
    Function that generates an occupancy grid from a mesh file (e.g. ) file

    Inputs
    ------
        - pathTo3DFile: the path to the obj file ('str' or file object)
        - file_type: the type of the mesh file, e.g. stl or obj (str)
        - pitch: the size of a grid box (float)
    
    Outputs
    -------
        - grid: (m,n,o) 3-dimensional ndarray containing the obstacles as 1's and free space as 0's
        - offsets: (3,) ndarray containing the offset of the object relative to the origin (in float)
    '''    
    mesh = trimesh.load_mesh(pathTo3DFile, file_type)
    angelVoxel = mesh.voxelized(pitch)
    points = np.array(angelVoxel.points)
    offsets = points.min(axis=0)
    points -= offsets

    grid = np.zeros(shape=(int(np.max(points[:,0])/pitch), int(np.max(points[:,1])/pitch) , int(np.max(points[:,2])/pitch)))
    points *= (np.asarray(grid.shape)-1)/points.max(axis=0)
    for x,y,z in points:
        grid[round(x), round(y), round(z)] = 1
    
    return grid, offsets

def checkneighbours2D(grid, x, y, xsize, ysize):
    '''
    Function that checks if a point in 2D has any neighbours that are obstacles

    Inputs:
    ------
        - grid: 2D numpy array of the occupancy grid (m,n)
        - x,y : current coordinate to check (int, int)
        - xsize,ysize: size of the grid in x,y direction (int, int)
    Output:
    -------
        - bool : True if a neighbouring cell contain an obstacle , False otherwise (bool)
'''  
    if x-1 > 0 and grid[x-1][y] == 1: return True
    if y-1 > 0 and grid[x][y-1] == 1: return True
    if x+1 < xsize and grid[x+1][y] == 1: return True
    if y+1 < ysize and grid[x][y+1] == 1: return True
    return False

def checkneighbours3D(grid, x, y, z, xsize, ysize, zsize):
    '''
    Function that checks if a point in 3D has any neighbours that are obstacles

    Inputs:
    ------
        - grid: 3D numpy array of the occupancy grid (m, n, o)
        - x,y,z : current coordinate to check (int, int, int)
        - xsize,ysize,zsize: size of the grid in x,y,z direction (int, int, int)
    Output:
    -------
        - True if a neighbouring cell contain an obstacle , False otherwise (bool)
    '''  
    if x-1 > 0 and grid[x-1][y][z] == 1: return True
    if y-1 > 0 and grid[x][y-1][z] == 1: return True
    if x+1 < xsize and grid[x+1][y][z] == 1: return True
    if y+1 < ysize and grid[x][y+1][z] == 1: return True
    if z-1 > 0 and grid[x][y][z-1] == 1: return True
    if z+1 < zsize and grid[x][y][z+1] == 1: return True
    return False

def marginise_grid2D(grid):
    '''
    Function that expands the obstacle region in 2D occupancy grid by adding cells that have obstacle neighbours

    Inputs:
    ------
        - grid: 2D numpy array of the occupancy grid

    Output:
    -------
        - newgrid: 2D numpy array of the occupancy grid with expanded obstacle region
    '''
    newgrid =np.zeros(grid.shape, dtype=np.int8)
    xsize, ysize = grid.shape
    count = 0
    for x in range(xsize):
        for y in range(ysize):
            if checkneighbours2D(grid, x, y, xsize, ysize) or grid[x][y] == 1:
                newgrid[x][y] = 1
                count+=1
    print(f"Found {count} points as margins")
    return newgrid

def marginise_grid3D(grid):
    '''
    Function that expands the obstacle region in 3D occupancy grid by adding cells that have obstacle neighbours

    Inputs:
    ------
        - grid: 3D numpy array of the occupancy grid

    Output:
    -------
        - newgrid: 3D numpy array of the occupancy grid with expanded obstacle region
    '''
    newgrid =np.zeros(grid.shape, dtype=np.int8)
    xsize, ysize, zsize = grid.shape
    count = 0
    for x in range(xsize):
        for y in range(ysize):
            for z in range(zsize):
                if checkneighbours3D(grid, x, y, z, xsize, ysize, zsize) or grid[x][y][z] == 1:
                    newgrid[x][y][z] = 1
                    count+=1
    print(f"Found {count} points as margins")
    return newgrid

def marginWithDepth(grid, desiredMarginDepthinMeters=0.1, pitchInMeters=0.05):
    '''
    Function that expands the obstacle region in the occupancy grid by a certain depth (in meters) by repeatedly calling
    marginise_grid2D or marginise_grid3D function depending on the dimension of the input grid.

    Inputs:
    ------
        - grid: 2D/3D numpy array of the occupancy grid
        - desiredMarginDepthinMeters : the desired additional obstacle region to add in meters (default = 0.1)
        - pitchInMeters : the size of a grid box in meters (default = 0.05)

    Output:
    -------
        - newgrid: 2D/3D numpy array of the occupancy grid with expanded obstacle region
    '''
    #newGrid = np.zeros(grid.shape, dtype=np.int8)
    iterations = int(np.rint(desiredMarginDepthinMeters/pitchInMeters))
    newGrid = grid.copy()
    if grid.ndim == 2:
        for i in range(iterations):
            newGrid = marginise_grid2D(newGrid)
        return newGrid
    elif grid.ndim == 3:
        for i in range(iterations):
            newGrid = marginise_grid3D(newGrid)
        return newGrid
    else:
        raise Exception("Grid dimension {} is not 2 or 3".format(grid.ndim))
        
def plotgrid3D(grid):
    '''
    Function that plots the 3D occupancy grid
    Inputs:
    ------
        - grid: 3D numpy array of the occupancy grid
    '''
    occupied_points = np.argwhere(grid == 1)
    x = occupied_points[:, 0]
    y = occupied_points[:, 1]
    z = occupied_points[:, 2]
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c='b', alpha =0.1)
    ax.set_aspect('equal')
    plt.show()

def plotgrid3Dwithmargin(grid, margin_grid):
    '''
    Function that plots the 3D occupancy grid and its expanded obstacle region
    Inputs:
    ------
        - grid: 3D numpy array of the occupancy grid
        - margin_grid: 3D numpy array of the occupancy grid with expanded obstacle region
    '''
    occupied_points = np.argwhere(grid == 1)
    x = occupied_points[:, 0]
    y = occupied_points[:, 1]
    z = occupied_points[:, 2]
    margin_occupied_points = np.argwhere(margin_grid == 1)
    margin_x = margin_occupied_points[:, 0]
    margin_y = margin_occupied_points[:, 1]
    margin_z = margin_occupied_points[:, 2]
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(margin_x, margin_y, margin_z, c='r', alpha =0.1)
    ax.scatter(x, y, z, c='b', alpha =0.1)
    
    ax.set_aspect('equal')
    plt.show()

def plotgrid2D(grid):
    '''
    Function that plots the 2D occupancy grid
    Inputs:
    ------
        - grid: 3D numpy array of the occupancy grid
    '''
    occupied_points = np.argwhere(grid == 1)
    x = occupied_points[:, 0]
    y = occupied_points[:, 1]
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.scatter(x, y, c='b', alpha =0.5)
    ax.set_aspect('equal')
    plt.show()

def plotgrid2Dwithmargin(grid, margin_grid):
    '''
    Function that plots the 2D occupancy grid and its expanded obstacle region
    Inputs:
    ------
        - grid: 3D numpy array of the occupancy grid
        - margin_grid: 3D numpy array of the occupancy grid with expanded obstacle region
    '''
    occupied_points = np.argwhere(grid == 1)
    x = occupied_points[:, 0]
    y = occupied_points[:, 1]
    margin_occupied_points = np.argwhere(margin_grid == 1)
    margin_x = margin_occupied_points[:, 0]
    margin_y = margin_occupied_points[:, 1]
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.scatter(margin_x, margin_y, c='r', alpha =0.5)
    ax.scatter(x, y, c='b', alpha =0.5)
    ax.set_aspect('equal')
    plt.show()
