from Testmaps import map
#from RRT_star import RRTstar
from tqdm import tqdm
import time
import numpy as np
from quadrotor_project.planningAlgorithms.RRT import RRTstar
import os

# Choose test map:
#   1: easy
#   2: normal
#   3: hard
# Others are empty

map_n = 4
start, goal, mapdim, dgoal, dsearch, dcheaper, obstacles, obsmargin, max_iter = map(map_n)

# Set map manually:
start = np.array([15, 15])
goal = np.array([183, 70])
mapdim = (200, 100)
dgoal = 10
dsearch = 15
dcheaper = 25
max_iter = 6000
obsmargin = 3
obstacles = []
obstacles.append(np.array([40, 0, 50, 50], dtype = object)) #Obstacles: [x1, y1, x2, y2]


#Computng map and path
graph = RRTstar(start, goal, mapdim, dgoal, dsearch, dcheaper, obstacles, obsmargin, max_iter)
graph.makemap()
starttime = time.time()
pbar = tqdm(total = max_iter)

while graph.iterations():
    graph.expand()
    pbar.update(1)

endtime = time.time()
pbar.close()
print("Time elapsed: ", endtime - starttime)

graph.makemap(showpath = False, showrest = False, shownodes = False)