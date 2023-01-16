from tqdm import tqdm
import time
import numpy as np
from quadrotor_project.planningAlgorithms.RRT import GridRRT3D, GridRRTstar3D
import quadrotor_project.planningAlgorithms.occupancyGridTools as GT
import os
import pkg_resources

####### THIS IS A TEST FILE ########

occupancy_grid, _ = GT.generateOccupancyGrid(pkg_resources.resource_filename("quadrotor_project", "assets/track2.obj"))

####EXAMPLE GRIDRRT2D
#(uncomment)
# start = np.array([10, 10])
# goal = np.array([40, 100])
# dgoal = 5
# dsearch = 10
# dcheaper = 15
# max_iter = 8000

# #grid = np.load('PDM_group35/PDM_project/RRT/occupancygrid.npy')
# grid2D = grid[:, :, 0]
# #m_grid2D = GT.marginise_grid2D(GT.marginise_grid2D(GT.marginise_grid2D(grid2D)))
# m_grid2D = GT.marginWithDepth(grid2D, 0.2, 0.05)
# graph = GridRRTstar2D(start, goal, dgoal, dsearch, dcheaper, grid2D, m_grid2D, max_iter)
# graph.makemap()
# starttime = time.time()
# pbar = tqdm(total = max_iter)

# while graph.iterations():
#     graph.expand()
#     pbar.update(1)

# endtime = time.time()
# pbar.close()
# print("Time elapsed: ", endtime - starttime)
# graph.makemap()


####EXAMPLE GRIDRRT3D
#(uncomment)
#Parcour2
# start = np.array([10, 10, 10])
# goal = np.array([80, 10, 10])

#Parcour1
start = np.array([20, 10, 10])
goal = np.array([110, 40, 40])
#goal = np.array([10, 70, 10])
dgoal = 5
dsearch = 15
dcheaper = 15
max_iter = 40000

#grid = np.load('PDM_group35/PDM_project/RRT/occupancygrid.npy')
grid = occupancy_grid
m_grid3D = GT.marginWithDepth(grid, 0.10)
graph = GridRRTstar3D(start, goal, dgoal, dsearch, dcheaper, grid, m_grid3D, max_iter)
graph.makemap()
starttime = time.time()
pbar = tqdm(total = max_iter)
while graph.iterations():
    graph.expand()
    pbar.update(1)
endtime = time.time()
pbar.close()
print("Time elapsed: ", endtime - starttime)
graph.makemap(showpath=True, showsimplegraph=True, showspline = False )

graph.plotpathlengths()






####OLD RRT TEST
# # Choose test map:
# #   1: easy
# #   2: normal
# #   3: hard
# # Others are empty

# map_n = 3
# start, goal, mapdim, dgoal, dsearch, dcheaper, obstacles, obsmargin, max_iter = map(map_n)

# # Set map manually:
# start = np.array([15, 15])
# goal = np.array([183, 70])
# mapdim = (200, 100)
# dgoal = 10
# dsearch = 15
# dcheaper = 25
# max_iter = 6000
# obsmargin = 3
# obstacles = []
# obstacles.append(np.array([40, 0, 50, 50], dtype = object)) #Obstacles: [x1, y1, x2, y2]

# #Computng map and path
# graph = RRTstar2D(start, goal, mapdim, dgoal, dsearch, dcheaper, obstacles, obsmargin, max_iter)
# graph.makemap()
# starttime = time.time()
# pbar = tqdm(total = max_iter)

# while graph.iterations():
#     graph.expand()
#     pbar.update(1)

# endtime = time.time()
# pbar.close()
# print("Time elapsed: ", endtime - starttime)

# graph.makemap(showpath = False, showrest = False, shownodes = False)