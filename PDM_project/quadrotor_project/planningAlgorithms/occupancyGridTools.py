import numpy as np
import trimesh

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
    angelVoxel = mesh.voxelize(pitch)
    points = np.array(angelVoxel.points)
    offsets = points.min(axis=0)
    points -= offsets

    grid = np.zeros(shape=(len(np.unique(points[:,0])),len(np.unique(points[:,1])) , len(np.unique(points[:,2]))))
    points *= (np.asarray(grid.shape)-1)/points.max(axis=0)
    for x,y,z in points:
        grid[round(x), round(y), round(z)] = 1
    
    return grid, offsets
