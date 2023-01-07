import numpy as np
import os
def map(n):
    print("Loading map ", n, "...")
    if n == 1:
        start = np.array([15, 15])
        goal = np.array([183, 70])
        mapdim = (200, 100)
        dgoal = 10
        dsearch = 15
        dcheaper = 25
        max_iter = 6000
        obsmargin = 3
        obstacles = []
        obstacles.append(np.array([40, 0, 50, 50], dtype = object))
        obstacles.append(np.array([110, 50, 120, 100], dtype = object))
        print("Loaded map 1 (easy)")
    
    elif n == 2:        
        start = np.array([15, 15])
        goal = np.array([180, 80])
        mapdim = (200, 100)
        dgoal = 10
        dsearch = 15
        dcheaper = 25
        max_iter = 10000
        obsmargin = 3
        obstacles = []
        obstacles.append(np.array([30, 0, 35, 70], dtype = object))
        obstacles.append(np.array([30, 70, 110, 75], dtype = object))
        obstacles.append(np.array([105, 50, 110, 70], dtype = object))
        obstacles.append(np.array([135, 25, 140, 110], dtype = object))
        obstacles.append(np.array([75, 20, 175, 25], dtype = object))
        obstacles.append(np.array([75, 20, 80, 50], dtype = object))
        obstacles.append(np.array([160, 50, 200, 55], dtype = object))
        obstacles.append(np.array([160, 55, 165, 80], dtype = object))
        print("Loaded map 2 (normal)")
    
    elif n == 3:
        start = np.array([15, 15])
        goal = np.array([187, 15])
        mapdim = (200, 145)
        dgoal = 10
        dsearch = 15
        dcheaper = 25
        max_iter = 20000
        obsmargin = 3
        obstacles = []
        obstacles.append(np.array([25, 0, 30, 25], dtype = object))
        obstacles.append(np.array([0, 55, 60, 60], dtype = object))
        obstacles.append(np.array([55, 20, 60, 60], dtype = object))
        obstacles.append(np.array([110, 25, 145, 30], dtype = object))
        obstacles.append(np.array([85, 0, 90, 125], dtype = object))
        obstacles.append(np.array([25, 85, 90, 90], dtype = object))
        obstacles.append(np.array([25, 90, 30, 125], dtype = object))
        obstacles.append(np.array([145, 55, 175, 60], dtype = object))
        obstacles.append(np.array([55, 110, 60, 145], dtype = object))
        obstacles.append(np.array([110, 25, 115, 145], dtype = object))
        obstacles.append(np.array([145, 60, 150, 125], dtype = object))
        obstacles.append(np.array([170, 0, 175, 55], dtype = object))
        obstacles.append(np.array([150, 120, 180, 125], dtype = object))
        obstacles.append(np.array([170, 95, 200, 100], dtype = object))
        print("Loaded map 3 (hard)")
    
    elif n == 4:   # Map using grids
        start = np.array([5, 5])
        goal = np.array([22, 50])
        occupancy_grid = np.load("./occupancygrid.npy")
        mapdim = (occupancy_grid.shape[0], occupancy_grid.shape[1])
        dgoal = 10
        dsearch = 15
        dcheaper = 25
        max_iter = 10000
        return start, goal, mapdim, dgoal, dsearch, dcheaper, occupancy_grid, max_iter
    else:
        start = np.array([15, 15])
        goal = np.array([180, 80])
        mapdim = (200, 100)
        dgoal = 10
        dsearch = 15
        dcheaper = 25
        max_iter = 4000
        obsmargin = 3
        obstacles = []
        print("Loaded empty map")

    return start, goal, mapdim, dgoal, dsearch, dcheaper, obstacles, obsmargin, max_iter
