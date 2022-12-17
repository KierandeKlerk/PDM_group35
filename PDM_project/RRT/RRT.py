import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import interpolate

class Map:
    #Initializing self
    def __init__(self, startpos, goalpos, mapdim):
        self.start = startpos
        self.goal = goalpos
        self.mapdim = mapdim
        self.mapw, self.maph = self.mapdim

    #Creating the map
    def makemap(self, obstacles):    #Obstacles of form: [leftx, bottomy, Height, width]
        fig, ax = plt.subplots(1, 1)

        #Adding start and goal points to the map
        ax.scatter(self.start[0], self.start[1])
        ax.scatter(self.goal[0], self.goal[1], marker = 'o', s = 500, alpha = 0.5)
        
        #Adding obstacles to the map
        obs = []
        for i in range(len(obstacles)):
            ax.add_patch(Rectangle((obstacles[i][0], obstacles[i][1]), obstacles[i][2], obstacles[i][3], color = '#454545'))
        
        #Setting map size
        plt.xlim([0, self.mapw])
        plt.ylim([0, self.maph])

        #Showing map
        plt.show()
    

#Tests
start = np.array([50, 150])
goal = np.array([50, 50])
mapdim = (100, 200)
obstacles = []
obstacles.append(np.array([10, 10, 30, 40], dtype = object))
obstacles.append(np.array([60, 60, 10, 10], dtype = object))

map = Map(start, goal, mapdim)
map.makemap(obstacles)


        
