from Testmaps import map
#from RRT_star import RRTstar
from tqdm import tqdm
import time
import numpy as np
from quadrotor_project.planningAlgorithms.RRT import RRTstar2D, GridRRTstar2D, GridRRTstar3D
import quadrotor_project.planningAlgorithms.occupancyGridTools as GT
import os

####EXAMPLE GRIDRRT2D
#(uncomment)
start = np.array([10, 10])
goal = np.array([40, 100])
dgoal = 5
dsearch = 10
dcheaper = 15
max_iter = 4000

grid = np.load('PDM_group35/PDM_project/RRT/occupancygrid.npy')
grid2D = grid[:, :, 0]
m_grid2D = GT.marginise_grid2D(GT.marginise_grid2D(grid2D))
graph = GridRRTstar2D(start, goal, dgoal, dsearch, dcheaper, grid2D, m_grid2D, max_iter)
graph.makemap()
starttime = time.time()
pbar = tqdm(total = max_iter)

while graph.iterations():
    graph.expand()
    pbar.update(1)

endtime = time.time()
pbar.close()
print("Time elapsed: ", endtime - starttime)
graph.makemap()


####EXAMPLE GRIDRRT3D
#(uncomment)
# start = np.array([10, 10, 20])
# goal = np.array([40, 100, 0])
# dgoal = 5
# dsearch = 10
# dcheaper = 15
# max_iter = 10000

# grid = np.load('PDM_group35/PDM_project/RRT/occupancygrid.npy')
# m_grid3D = GT.marginise_grid3D(GT.marginise_grid3D(GT.marginise_grid3D(grid)))
# graph = GridRRTstar3D(start, goal, dgoal, dsearch, dcheaper, grid, m_grid3D, max_iter)
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