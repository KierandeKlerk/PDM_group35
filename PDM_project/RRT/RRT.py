import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import interpolate
import time

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
        for i in range(len(obstacles)):
            ax.add_patch(Rectangle((obstacles[i][0], obstacles[i][1]), obstacles[i][2], obstacles[i][3], color = '#454545'))

        #Setting map size
        plt.xlim([0, self.mapw])
        plt.ylim([0, self.maph])

        #Showing map
        plt.show()    

class RRT:
    #initializing self
    def __init__(self, startpos, goalpos, mapdim, d_goal, d_search):
        self.start = startpos
        self.goal = goalpos
        self.mapdim = mapdim
        self.mapw, self.maph = self.mapdim
        self.parent = []
        self.x = [startpos[0]]
        self.y = [startpos[1]]
        self.obsregion = []
        self.edge = []
        self.path = []
        self.goalfound = False
        self.d_goal = d_goal
        self.d_search = d_search
        

    def addobst(self, obstacles):
        #Adding obstacle coordinates to obsregion
        for i in range(len(obstacles)):           
            for x in range(obstacles[i][2]):
                for y in range(obstacles[i][3]):
                    self.obsregion.append((obstacles[i][0] + x, obstacles[i][1] + y))       

    def addnode(self, x, y):
        #print('Add node (', x, ", ", y, ")")
        self.x.append(x)
        self.y.append(y)
    
    def removenode(self, n):
        #print('remove node ', n)
        self.x.pop(n)
        self.y.pop(n)

    def addedge(self, nparent, nchild):
        #print('Add edge between', nparent, ' and ', nchild)
        self.edge.append(nparent)
        
    def removeedge(self, n):
        #print('remove edge ', n)
        self.edge.pop(n)
    
    def distance(self, n1, n2):
        dx = (self.x[n1] - self.x[n2])
        dy = (self.y[n1] - self.y[n2])
        distance = np.sqrt(dx**2 + dy**2)
        #print('Distance between ', n1, " and ", n2, ": ", distance)
        return distance

    def goaldistance(self, n):
        dx = (self.x[n] - self.goal[0])
        dy = (self.y[n] - self.goal[1])
        distance = np.sqrt(dx**2 + dy**2)
        #print('Distance between ', n, " and the goal: ", distance)
        return distance

    def nearestnode(self, n):
        distances = np.zeros(n)
        for i in range(n):
            distances[i] = self.distance(i, n)
        #print("Node nearest to ", n, ": ", np.argmin(distances))
        return np.argmin(distances)

    def nodefree(self, n):
        #print('Check node ', n)
        if ((self.x[n], self.y[n]) in self.obsregion):
            return False
        else:
            return True  

    def edgefree(self, n1, n2):
        #print('Check edge between ', n1, "and", n2)
        steps = np.linspace(0, 1, 100)
        for step in steps:
            x = self.x[n1] * step + self.x[n2] * (1-step)
            y = self.y[n1] * step + self.y[n2] * (1-step)
            if  (x, y) in self.obsregion:
                return False
        return True
        
    def randomsample(self):
        #print('Take random sample')
        x = np.random.randint(0, self.mapw)
        y = np.random.randint(0, self.maph)
        return (x, y)

    def expand(self):
        #print('Expand')
        (x, y) = self.randomsample()
        self.addnode(x, y)
        n = len(self.x) - 1
        nnear = self.nearestnode(n)
        if self.nodefree(n) and (self.distance(n, nnear) <= self.d_search) and self.edgefree(nnear, n):
            self.addedge(nnear, n)
            if self.goaldistance(n) <= self.d_goal:
                self.goalfound = True
        else: 
            self.removenode(n)

    
    def pathfound(self):
        if self.goalfound: print("Path found in", len(self.x), 'steps')
        return self.goalfound
    
    #Creating the map
    def makemap(self, obstacles):    #Obstacles of form: [leftx, bottomy, Height, width]
        fig, ax = plt.subplots(1, 1)

        #Adding start and goal points to the map
        ax.scatter(self.start[0], self.start[1])
        ax.scatter(self.goal[0], self.goal[1], marker = 'o', s = 500, alpha = 0.5)
        
        #Adding obstacles to the map
        for i in range(len(obstacles)):
            ax.add_patch(Rectangle((obstacles[i][0], obstacles[i][1]), obstacles[i][2], obstacles[i][3], color = '#454545'))

        #Adding nodes
        for i in range(1, len(self.x)):
            ax.scatter(self.x[i], self.y[i], color = '#FF0000')
            #ax.text(self.x[i], self.y[i], i)

        #Adding edges
        for i in range(len(self.edge)):
            x_values = [self.x[i+1], self.x[self.edge[i]]]
            y_values = [self.y[i+1], self.y[self.edge[i]]]
            ax.plot(x_values, y_values, color = '000000')

        #Setting map size
        plt.xlim([0, self.mapw])
        plt.ylim([0, self.maph])

        #Showing map
        plt.show()   

    
#Tests
start = np.array([50, 10])
goal = np.array([170, 80])
mapdim = (200, 100)
dgoal = 10
dsearch = 30
obstacles = []
obstacles.append(np.array([10, 10, 20, 20], dtype = object))
obstacles.append(np.array([60, 60, 30, 10], dtype = object))
obstacles.append(np.array([75, 0, 20, 60], dtype = object))

map = Map(start, goal, mapdim)
map.makemap(obstacles)

graph = RRT(start, goal, mapdim, dgoal, dsearch)
graph.addobst(obstacles)
t1 = time.time()
while not graph.pathfound():
    time.sleep(0.005)
    elapsed = time.time() -t1
    t1 = time.time()
    graph.expand()
graph.makemap(obstacles)
