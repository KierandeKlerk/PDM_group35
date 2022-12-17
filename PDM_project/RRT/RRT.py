import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import interpolate

class Map:
    def __init__(self, startpos, goalpos, mapdim):
        #Initializing self
        self.start = startpos
        self.goal = goalpos
        self.mapdim = mapdim
        self.mapw, self.maph = self.mapdim


    def makemap(self, obstacles):
        ##Obstacles of form: [(leftx, lefty), Height, width]
        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.start[0], self.start[1])
        ax.scatter(self.goal[0], self.goal[1], marker = 'o', s = 500, alpha = 0.5)
        #ax.add_patch(Rectangle(obstacles[0], obstacles[1], obstacles[2]))
        plt.xlim([0, self.mapw])
        plt.ylim([0, self.maph])
        plt.show()
    
start = np.array([50, 150])
goal = np.array([50, 50])
mapdim = (100, 200)
obstacles = np.array([(10, 10), 30, 40], dtype = object)
map = Map(start, goal, mapdim)
map.makemap(obstacles)


        
