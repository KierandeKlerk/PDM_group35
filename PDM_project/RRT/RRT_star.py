import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy import interpolate
import time

class RRTstar:
    #initializing self
    def __init__(self, startpos, goalpos, mapdim, d_goal, d_search, d_cheaper, obstacles, obsmargin, max_iter):
        #Initializing map parameters
        self.start = startpos
        self.goal = goalpos
        self.mapw, self.maph = mapdim
        #Initializing obstacle parameters
        self.obstacles = obstacles
        self.obsmargin = obsmargin
        #Initializing node coordinates
        self.x = [startpos[0]]
        self.y = [startpos[1]]
        #Initializing edge parameters
        self.parent = [0] #Gives nodenumber of parent
        self.path = [] #Gives nodenumbers of final path to goal
        #Initializing search parameters
        self.d_goal = d_goal
        self.d_search = d_search
        self.d_cheaper = d_cheaper #Parameter for RRT*
        #Initializing iterations
        self.iters= 0
        self.max_iter = max_iter
        #Initializing goalfound parameter
        self.goalfound = False        

    #Function to check if a point is free of obstacles
    def isfree(self, x, y):
        free = True
        i = 0
        nobst = len(self.obstacles)
        while free and i < nobst:
            if ((self.obstacles[i][0] - self.obsmargin) <= x <= (self.obstacles[i][2] + self.obsmargin)) and ((self.obstacles[i][1]- self.obsmargin) <= y <= (self.obstacles[i][3] + self.obsmargin)):
                free = False
            i += 1
        return free
        
    #Function to add a node to the list
    def addnode(self, x, y):
        #print('Add node (', x, ", ", y, ")")
        self.x.append(x)
        self.y.append(y)
    
    #Function to remove the nth node from the list
    def removenode(self, n):
        #print('remove node ', n)
        self.x.pop(n)
        self.y.pop(n)

    #Function to add an edge between the parent and the child node
    def addedge(self, nparent):
        #print("Added edge")
        self.parent.append(nparent)
    
    #Function to remove an edge
    def removeedge(self, n):
        #print('remove edge ', n)
        self.parent.pop(n)
    
    #Function to calculate the distance between nodes n1 and n2
    def distance(self, n1, n2):
        dx = (self.x[n1] - self.x[n2])
        dy = (self.y[n1] - self.y[n2])
        distance = np.sqrt(dx**2 + dy**2)
        #print('Distance between ', n1, " and ", n2, ": ", distance)
        return distance

    #Function to calculate the distance to the endgoal
    def goaldistance(self, n):
        dx = (self.x[n] - self.goal[0])
        dy = (self.y[n] - self.goal[1])
        distance = np.sqrt(dx**2 + dy**2)
        #print('Distance between ', n, " and the goal: ", distance)
        return distance

    #Function to calculate distances to each node
    def distances(self, n):
        distances = np.zeros(n)
        for i in range(n):
            distances[i] = self.distance(i, n)
        #print("Node nearest to ", n, ": ", np.argmin(distances))
        return distances

    #Function to check if the node is in the free space
    def nodefree(self, n):
        #print('Check node ', n)
        return self.isfree(self.x[n], self.y[n])

    #Function to check if the edge is in the free space
    def edgefree(self, n1, n2):
        #print('Check edge between ', n1, "and", n2)
        steps = np.linspace(0, 1, np.int64(self.distance(n1, n2)))
        for step in steps:
            x = np.int64(self.x[n1] * step + self.x[n2] * (1-step))
            y = np.int64(self.y[n1] * step + self.y[n2] * (1-step))
            if self.isfree(x, y) == False:
                return False
        return True
    
    #Function to take a random sample in the map
    def randomsample(self):
        #print('Take random sample')
        x = np.random.randint(0, self.mapw)
        y = np.random.randint(0, self.maph)
        return (x, y)
    
    #Function to expand the graph
    def expand(self):
        #print('Expand')
        (x, y) = self.randomsample()
        self.addnode(x, y)
        n = len(self.x) - 1
        distances = self.distances(n)
        nnear = np.argmin(distances)
        if self.edgefree(nnear, n) and (self.distance(n, nnear) <= self.d_search):
            self.addedge(nnear)
            cheapernode = self.cheapernodes(n)
            if cheapernode != nnear:
                #print("found cheaper node: ", cheapernode)
                self.removeedge(n)
                self.addedge(cheapernode)
            if self.goaldistance(n) <= self.d_goal:
                self.addnode(self.goal[0], self.goal[1])
                self.addedge(n)
                cheapernode = self.cheapernodes(n+1)
                if cheapernode != n:
                    self.removeedge(n+1)
                    self.addedge(cheapernode)
                self.goalfound = True
        else: 
            self.removenode(n)
    
    #Function to search for cheaper nodes to connect to
    def cheapernodes(self, n):
        cheapestparent = self.parent[n]
        lowestcost = self.reachcost(n)
        for i in range(n-1):
            if self.distance(n, i) < self.d_cheaper:
                if self.edgefree(n, i):
                    cost = self.reachcost(i) + self.distance(n, i)
                    if cost < lowestcost:
                        cheapestparent = i
                        lowestcost = cost
        return cheapestparent

    #Function to calculate the cost to reach a node from the start
    def reachcost(self, n):
        nparent = self.parent[n]
        cost = 0
        while n != 0:
            cost += self.distance(n, nparent)
            n = nparent
            nparent = self.parent[n]
        return cost   
    
    #Function to check if the goal is found, and to collect the nodes on the path
    def pathfound(self):
        if self.goalfound: 
            self.path = []
            ngoal = len(self.x)-1
            self.path.append(ngoal)
            nparent = self.parent[ngoal]
            while nparent != 0:
                self.path.append(nparent)
                nparent = self.parent[nparent]
            self.path.append(0)
            print("Goal found with", len(self.x), 'nodes after ', self.iters, " iterations.")
            print("Cost to reach goal: ", self.reachcost(len(self.x)-1))
        return self.goalfound

    #Function to count the iterations
    def iterations(self):
        self.iters += 1
        # if self.iters % 50 == 0:
        #     print("iteration ", self.iters)
        if self.iters < self.max_iter:
            return True
        else: 
            print("No goal found after ", self.iters, " iterations :(")
            return False
    
    #Function to get the path to the goal
    def getpath(self):
        path = []
        for i in self.path:
            path.append((self.x[i], self.y[i]))
        return path

    #Function to get a smooth path to the goal
    def getsmoothpath(self):
        self.smoothpath = []
        if self.goalfound:
            u = np.linspace(0, 1, 400)
            x = []
            y = []
            coords = self.getpath()
            for i in range(len(self.getpath())):
                x.append(coords[i][0])
                y.append(coords[i][1])
            tck, _ = interpolate.splprep([x, y])
            self.smoothpath = interpolate.splev(u, tck)

    #Function to visualize the map and graph
    def makemap(self):   
        fig, ax = plt.subplots(1, 1)

        #Adding start and goal points to the map
        ax.scatter(self.start[0], self.start[1], s = 100)
        ax.add_patch(plt.Circle(self.goal, self.d_goal, ec = '#FFA500', fill = False, lw = 5))
        #Adding obstacles to the map
        for i in range(len(self.obstacles)):
            ax.add_patch(Rectangle((self.obstacles[i][0], self.obstacles[i][1]), (self.obstacles[i][2] - self.obstacles[i][0]), (self.obstacles[i][3] - self.obstacles[i][1]), color = '#454545'))

        # #Adding nodes to the map
        # for i in range(1, len(self.x)):
        #     if i in self.path:
        #         ax.scatter(self.x[i], self.y[i], color = '#FF0000')
        #     else: 
        #         ax.scatter(self.x[i], self.y[i], color = '#000000')

        #Adding edges to the map
        for i in range(1, len(self.parent)):
            x_values = [self.x[i], self.x[self.parent[i]]]
            y_values = [self.y[i], self.y[self.parent[i]]]
            if i in self.path:
                ax.plot(x_values, y_values, color = '#FF0000')
            else:
                ax.plot(x_values, y_values, color = '#000000')

        #Adding smooth path to the map
        self.getsmoothpath()
        coords = self.smoothpath
        if len(coords) != 0:
            for i in range(len(coords[0])):
                ax.scatter(coords[0][i], coords[1][i], color = '#00FF00')

        #Setting map size
        plt.xlim([0, self.mapw])
        plt.ylim([0, self.maph])

        #Showing map
        plt.show()   
    
#Tests
starttime = time.time()
start = np.array([10, 10])
goal = np.array([187, 15])
mapdim = (200, 100) #145
dgoal = 10
dsearch = 15
dcheaper = 25
max_iter = 10000
obsmargin = 3
obstacles = []

##Map 3
# obstacles.append(np.array([25, 0, 30, 25], dtype = object))
# obstacles.append(np.array([0, 55, 60, 60], dtype = object))
# obstacles.append(np.array([55, 20, 60, 60], dtype = object))
# obstacles.append(np.array([110, 25, 145, 30], dtype = object))
# obstacles.append(np.array([85, 0, 90, 125], dtype = object))
# obstacles.append(np.array([25, 85, 90, 90], dtype = object))
# obstacles.append(np.array([25, 90, 30, 125], dtype = object))
# obstacles.append(np.array([145, 55, 175, 60], dtype = object))
# obstacles.append(np.array([55, 110, 60, 145], dtype = object))
# obstacles.append(np.array([110, 25, 115, 145], dtype = object))
# obstacles.append(np.array([145, 60, 150, 125], dtype = object))
# obstacles.append(np.array([170, 0, 175, 55], dtype = object))
# obstacles.append(np.array([150, 120, 180, 125], dtype = object))
# obstacles.append(np.array([170, 95, 200, 100], dtype = object))

##MAP 2
obstacles.append(np.array([30, 0, 35, 70], dtype = object))
obstacles.append(np.array([30, 70, 110, 75], dtype = object))
obstacles.append(np.array([105, 50, 110, 70], dtype = object))
obstacles.append(np.array([135, 25, 140, 110], dtype = object))
obstacles.append(np.array([75, 20, 175, 25], dtype = object))
obstacles.append(np.array([75, 20, 80, 50], dtype = object))
obstacles.append(np.array([160, 50, 200, 55], dtype = object))
obstacles.append(np.array([160, 55, 165, 80], dtype = object))

##MAP 1
# obstacles.append(np.array([40, 0, 50, 50], dtype = object))
# obstacles.append(np.array([90, 50, 100, 100], dtype = object))

graph = RRTstar(start, goal, mapdim, dgoal, dsearch, dcheaper, obstacles, obsmargin, max_iter)
#graph.addobst()
graph.makemap()
while not graph.pathfound() and graph.iterations():
    graph.expand()
endtime = time.time()
print("Time elapsed: ", endtime - starttime)
graph.makemap()
