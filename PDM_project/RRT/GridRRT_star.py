import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy import interpolate
import time
from tqdm import tqdm
from math import ceil, floor


class GridRRTstar:
    #initializing self
    def __init__(self, startpos, goalpos, mapdim, d_goal, d_search, d_cheaper, grid, max_iter):
        #Initializing map parameters
        self.start = startpos
        self.goal = goalpos
        self.mapw, self.maph = mapdim
        #Initializing obstacle parameters
        self.grid = grid
        locs = np.where(grid == 1)
        self.obstacles = np.concatenate((locs[0].reshape(len(locs[0]), 1), locs[1].reshape(len(locs[0]), 1)), axis=1)
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
    ####### Functions to create and remove edges and nodes #######

    #Function to add a node to the list
    def addnode(self, x, y):
        self.x.append(x)
        self.y.append(y)
    
    #Function to remove the nth node from the list
    def removenode(self, n):
        self.x.pop(n)
        self.y.pop(n)

    #Function to add an edge between the parent and the child node
    def addedge(self, nparent):
        self.parent.append(nparent)
    
    #Function to remove an edge
    def removeedge(self, n):
        self.parent.pop(n)
    

    ####### Functions to handle obstacles in the map #######

    #Function to check if a point is free of obstacles
    def isfree(self, x, y, margin=0):
        ps = []
        for i in range(margin+1):
            xmargin = min(max(x+i,0), self.mapw-1)
            ymargin = min(max(y+i,0), self.maph-1)
            xmargin2 = min(max(x-i,0), self.mapw-1)
            ymargin2 = min(max(y-i,0), self.maph-1)
            ps.append(self.grid[xmargin2][ymargin2])
            ps.append(self.grid[xmargin][ymargin])
            if self.grid[xmargin2][ymargin2] + self.grid[xmargin][ymargin] > 0:
                return False
        if sum(ps) == 0:
            return True
        else:
            return False

    #Function to check if the node is in the free space
    def nodefree(self, n):
        return self.isfree(self.x[n], self.y[n])

    #Function to check if the edge is in the free space
    def edgefree(self, n1, n2):
        steps = np.linspace(0, 1, round(self.distance(n1, n2))*2)
        for step in steps:
            x = round(self.x[n1] * step + self.x[n2] * (1-step))
            y = round(self.y[n1] * step + self.y[n2] * (1-step))
            if self.isfree(x, y) == False:
                return False
        return True

    
    ####### Functions to measure distances and costs ########

    #Function to calculate the distance between nodes n1 and n2
    def distance(self, n1, n2):
        dx = (self.x[n1] - self.x[n2])
        dy = (self.y[n1] - self.y[n2])
        return np.sqrt(dx**2 + dy**2)

    def distance_dxdy(self, n1, n2):
        dx = (self.x[n1] - self.x[n2])
        dy = (self.y[n1] - self.y[n2])
        return dx, dy
    #Function to calculate distances to each node
    def distances(self, n):
        distances = np.zeros(n)
        for i in range(n):
            distances[i] = self.distance(i, n)
        return distances

    #Function to calculate the distance to the endgoal
    def goaldistance(self, n):
        dx = (self.x[n] - self.goal[0])
        dy = (self.y[n] - self.goal[1])
        return np.sqrt(dx**2 + dy**2)
    
    #Function to calculate the cost to reach a node from the start
    def reachcost(self, n):
        nparent = self.parent[n]
        cost = 0
        while n != 0:
            cost += self.distance(n, nparent)
            n = nparent
            nparent = self.parent[n]
        return cost   

    #Function to take a random sample in the map
    def randomsample(self):
        x = np.random.randint(0, self.mapw)
        y = np.random.randint(0, self.maph)
        return (x, y)
    

    #######Functions to make the path optimal (RRT*)#######
    
    #Function to search for cheaper nodes to connect to
    def cheapernodes(self, n, distances):
        cheapestparent = self.parent[n]
        lowestcost = self.reachcost(n)
        for i in range(n-1):
            if distances[i] < self.d_cheaper:
                if self.edgefree(n, i):
                    cost = self.reachcost(i) + self.distance(n, i)
                    if cost < lowestcost:
                        cheapestparent = i
                        lowestcost = cost
        return cheapestparent
    
    #Function to rewire nodes
    def rewirecheck(self, n, distances):
        for i in range(n-1):
            if distances[i] < self.d_cheaper:
                if self.edgefree(i, n):
                    newcost = self.reachcost(n) + self.distance(n, i)
                    if newcost < self.reachcost(i):
                        self.parent[i] = n       


    ####### Main functions used while iterating #######

    #Function to expand the graph
    def expand(self):
        (x, y) = self.randomsample()
        self.addnode(x, y)
        n = len(self.x) - 1
        distances = self.distances(n)
        nnear = np.argmin(distances)
        if self.edgefree(nnear, n) and (self.distance(n, nnear) <= self.d_search):
            self.addedge(nnear)
            cheapernode = self.cheapernodes(n, distances)
            if cheapernode != nnear:
                self.removeedge(n)
                self.addedge(cheapernode)
            self.rewirecheck(n, distances)

            if self.goaldistance(n) <= self.d_goal:
                if self.goalfound == False:
                    print("Goal found after", self.iters, "iterations! Searching for better path...")
                self.addnode(self.goal[0], self.goal[1])
                self.addedge(n)
                distances = self.distances(n+1)
                cheapernode = self.cheapernodes(n+1, distances)
                self.goalindex = n+1
                if cheapernode != n:
                    self.removeedge(n+1)
                    self.addedge(cheapernode)
                self.goalfound = True
        else: 
            self.removenode(n)
             
    #Function to count the iterations
    def iterations(self):
        self.iters += 1
        if self.iters < self.max_iter:
            return True
        else: 
            if self.goalfound:
                print("Goal found with", len(self.x), 'nodes after ', self.iters, " iterations.")
                print("Cost to reach goal: ", self.reachcost(len(self.x)-1))
            else:
                print("No goal found after ", self.iters, " iterations :( \nMade", len(self.x), "nodes.")
            return False
    

    ####### Functions that compute the final path from start to finish #######

    #Function to collect the nodes and coordinates used for the path
    def makepath(self):
        if self.goalfound: 
            self.path = []
            self.pathy = [self.start[0]]
            self.pathx = [self.start[1]]
            self.path.insert(0, self.goalindex)
            nparent = self.parent[self.goalindex]
            while nparent != 0:
                self.path.insert(1, nparent)
                nparent = self.parent[nparent]
            for i in range(1, len(self.path)):
                if not ((self.x[self.path[i]] == self.x[self.path[i-1]]) and (self.y[self.path[i]] == self.y[self.path[i-1]])): 
                    self.pathx.append(self.x[self.path[i]])
                    self.pathy.append(self.y[self.path[i]])   

    #Function to get a smooth path to the goal
    def getsmoothpath(self):
        self.smoothpath = []
        if self.goalfound:
            u = np.linspace(0, 1, 200)
            tck, _ = interpolate.splprep([self.pathx, self.pathy])
            self.smoothpath = interpolate.splev(u, tck)


    ####### Visualization #######

    #Function to visualize the map and graph
    def makemap(self, showpath = True, showrest = False, shownodes = False):   
        self.makepath()
        fig, ax = plt.subplots(1, 1)

        #Adding start and goal points to the map
        ax.scatter(self.start[0], self.start[1], s = 100)
        ax.add_patch(plt.Circle(self.goal, self.d_goal, ec = '#FFA500', fill = False, lw = 5))
        #Adding obstacles to the map
        ax.scatter(self.obstacles[:, 0], self.obstacles[:, 1])
        # #Adding nodes to the map
        if shownodes:
            for i in range(1, len(self.x)):
                if (i in self.path) and showpath:
                    ax.scatter(self.x[i], self.y[i], color = '#FF0000')
                elif showrest: 
                    ax.scatter(self.x[i], self.y[i], color = '#000000')

        #Adding edges to the map
        if showrest or showpath:
            for i in range(1, len(self.parent)):
                x_values = [self.x[i], self.x[self.parent[i]]]
                y_values = [self.y[i], self.y[self.parent[i]]]
                if (i in self.path) and showpath:
                    ax.plot(x_values, y_values, color = '#FF0000')
                elif showrest:
                    ax.plot(x_values, y_values, color = '#000000')

        #Adding smooth path to the map
        self.getsmoothpath()
        coords = self.smoothpath
        # print("coords: ", coords)
        if len(coords) != 0:
            for i in range(len(coords[0])):
                ax.scatter(coords[0][i], coords[1][i], color = '#00FF00')

        #Setting map size
        plt.xlim([0, self.mapw])
        plt.ylim([0, self.maph])
    
        #Showing map
        plt.show()   
