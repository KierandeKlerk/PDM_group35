import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy import interpolate

###################################################################################

class GridRRTstar3D:
    def __init__(self, startpos, goalpos, d_goal, d_search, d_cheaper, grid, mgrid, max_iter):
        '''
        Function that initializes all the class parameters

        Inputs
        ------
            - startpos: list of [x y z] coordinates of the starting position (int list with length 3)
            - goalpos: list of [x y z] coordinates of the goal position (int list with length 3)
            - d_goal: the maximum distance between a node to the goal with which we can say that the goal is found (float)
            - d_search: the maximum distance between a new node and existing nodes to be able to make the node (float)
            - d_cheaper: the maximum distance in which the algorithm searches for cheaper nodes to connect to (float)
            - grid: grid of zeros (free space) and ones (obstacles) of the map, used for visualization (int array of size (maxx, maxy, maxz))
            - mgrid: margin grid of zeros (free space) and ones (obstacles + margins) of the map (int array of size (maxx, maxy, maxz))
            - max_iter: the maximum amount of iterations that the algorithm should run (int)
        
        Parameters
        -----
            - self.start: list of [x y z] coordinates of the starting position (list with length 3)
            - self.goal: list of [x y z] coordinates of the goal position (list with length 3)
            - self.mapw: width of the map (int)
            - self.mapd: depth of the map (int)
            - self.maph: height of the map (int)
            - self.grid: grid of zeros (free space) and ones (obstacles) of the map, used for visualization (int array of size (maxx, maxy, maxz))
            - self.mgrid: margin grid of zeros (free space) and ones (obstacles + margins) of the map (int array of size (maxx, maxy, maxz))
            - self.obstacle: coordinates of all ones (obstacle coordinates) in the grid (int array of size (..., 3))
            - self.x: list of x coordinates of nodes (int list of length n)
            - self.y: list of y coordinates of nodes (int list of length n)
            - self.z: list of z coordinates of nodes (int list of length n)
            - self.parent: list of parent of each node (e.g. node 1 has a parent self.parent[1]) (int list of length n)
            - self.path: list of nodes that are used to get to the goal (int list of length ...)
            - self.smoothpath: coordinates of the final splinepath from start to goal (float array of shape (3, ...))
            - self.smoothpathfirst: coordinates of the first splinepath from start to goal (float array of shape (3, ...))
            - self.d_goal: the maximum distance between a node to the goal with which we can say that the goal is found (float)
            - self.d_search: the maximum distance between a new node and existing nodes to be able to make the node (float)
            - self.d_cheaper: the maximum distance in which the algorithm searches for cheaper nodes to connect to (float)
            - self.iters: iterations passed (int)
            - self.max_iter: the maximum amount of iterations that the algorithm should run (int)
            - self.goalfound: True if the goal is found, False if not (bool)
        '''  
        #Initializing map parameters
        self.start = startpos
        self.goal = goalpos
        self.mapw, self.mapd, self.maph = grid.shape

        #Initializing obstacle parameters
        self.grid = grid
        self.mgrid = mgrid
        locs = np.where(grid == 1)
        self.obstacles = np.concatenate((locs[0].reshape(len(locs[0]), 1), locs[1].reshape(len(locs[0]), 1), locs[2].reshape(len(locs[2]), 1)), axis=1)

        #Initializing node coordinates with start position
        self.x = [startpos[0]]
        self.y = [startpos[1]]
        self.z = [startpos[2]]

        #Initializing edge parameter
        self.parent = [0]

        #Initializing path parameters
        self.path = []
        self.smoothpath = [] 
        self.smoothpathfirst = []

        #Initializing search parameters
        self.d_goal = d_goal
        self.d_search = d_search
        self.d_cheaper = d_cheaper 

        #Initializing iteration parameters
        self.iters = 0
        self.max_iter = max_iter

        #Initializing goalfound parameter
        self.goalfound = False     


    ####### Functions to create and remove edges and nodes #######  
    
    def addnode(self, x, y, z):
        '''
        Function that adds a new node to the node lists from given coordinates

        Inputs
        ------
            - x: x coordinate of new node (int)
            - y: y coordinate of new node (int)
            - z: z coordinate of new node (int)
        '''  
        
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
    
    def removenode(self, n):
        '''
        Function that removes a node from the node lists

        Inputs
        ------
            - n: number of the node that needs to be removed (int)
        '''      

        self.x.pop(n)
        self.y.pop(n)
        self.z.pop(n)

    def addedge(self, nparent):
        '''
        Function that adds a new edge by saving the nodenumber of the parent of the new node

        Inputs
        ------
            - nparent: the nodenumber of the parent of the new node (int)
        '''  

        self.parent.append(nparent)

    def removeedge(self, n):
        '''
        Function that removes an edge by removing the nodenumber of the parent from the list

        Inputs
        ------
            - n: the nodenumber of the child which edge needs to be removed (int)
        '''  

        self.parent.pop(n)


    ####### Functions to handle obstacles in the map #######

    def isfree(self, x, y, z):
        '''
        Function that checks if a point is free by checking the grid (with margins)

        Inputs
        ------
            - x: x coordinate of point to check (int)
            - y: y coordinate of point to check (int)
            - z: z coordinate of point to check (int)
        
        Outputs
        ------
            - Returns a bool which indicates if a position is free (True) or occupied (False) (bool)
        '''  

        if self.mgrid[x][y][z] == 0:
            return True
        else:
            return False

    def nodefree(self, n):
        '''
        Function that checks if a node is free by checking the grid (with margins)

        Inputs
        ------
            - n: the nodenumber of the node that has to be checked (int)
        
        Outputs
        ------
            - Returns a bool which indicates if a node is free (True) or occupied (False) (bool)
        '''  

        return self.isfree(self.x[n], self.y[n], self.z[n])

    #Function to check if the edge is in the free space
    def edgefree(self, n1, n2):
        '''
        Function that achecks if an edge between node n1 and node n1 is free on the grid (with margins) by checking if interpolated points are free

        Inputs
        ------
            - n1: the nodenumber of the first node of the edge (int)
            - n2: the nodenumber of the second node of the edge (int)
        
        Outputs
        ------
            - Returns a bool which indicates if an edge is free (True) or occupied (False) (bool)
        ''' 

        #Interpolate with the same amount of steps as the distance between the 2 nodes
        steps = np.linspace(0, 1, np.int64(self.distance(n1, n2)))
        for step in steps:
            x = round(self.x[n1] * step + self.x[n2] * (1-step))
            y = round(self.y[n1] * step + self.y[n2] * (1-step))
            z = round(self.z[n1] * step + self.z[n2] * (1-step))
            if self.isfree(x, y, z) == False:
                return False
        return True


    ####### Functions to measure distances and costs ########

    def distance(self, n1, n2):
        '''
        Function that calculates the distance between node n1 and node n2

        Inputs
        ------
            - n1: the nodenumber of the first node (int)
            - n2: the nodenumber of the second node (int)
        
        Outputs
        ------
            - Returns the distance between the 2 given nodes (float)
        ''' 

        dx = (self.x[n1] - self.x[n2])
        dy = (self.y[n1] - self.y[n2])
        dz = (self.z[n1] - self.z[n2])
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def distances(self, n):
        '''
        Function that calculates the distances from node n to every other node

        Inputs
        ------
            - n: the nodenumber of the node which the distances should be calculated for (int)
        
        Outputs
        ------
            - distances: distances from node n to each node (float array of shape (n, ))
        ''' 

        distances = np.zeros(n)
        for i in range(n):
            distances[i] = self.distance(i, n)
        return distances

    def goaldistance(self, n):
        '''
        Function that calculates the distance between node n and the goal position

        Inputs
        ------
            - n: the nodenumber of the node that the distance should be calculated (int)
        
        Outputs
        ------
            - Returns the distance between node n and the goal position (float)
        ''' 

        dx = (self.x[n] - self.goal[0])
        dy = (self.y[n] - self.goal[1])
        dz = (self.z[n] - self.goal[2])
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def reachcost(self, n):
        '''
        Function that calculates the cost to reach node n by checking the length of each edge required to reach the node from the start position

        Inputs
        ------
            - n: the nodenumber of the node for which the reach cost should be calculated (int)
        
        Outputs
        ------
            - cost: the total distance of reaching node n from the start position following the edges (float)
        ''' 

        nparent = self.parent[n]
        cost = 0
        while n != 0:
            cost += self.distance(n, nparent)
            n = nparent
            nparent = self.parent[n]
        return cost   

    
    #######Functions to make the path optimal (RRT*)#######

    def cheapernodes(self, n, distances):
        '''
        Function that searches for the cheapest node to connect to, which makes the path optimal

        Inputs
        ------
            - n: the nodenumber of the node for which the cheaper parent should be found (int)
            - distances: an array of length n which gives distances to each node from node n (float array of shape (n, ))
        
        Outputs
        ------
            - cheapestparent: the nodenumber of the cheapestparent for node n (int)
        ''' 

        #Initialize cheapest parent as current parent and lowestcost as current cost
        cheapestparent = self.parent[n]
        lowestcost = self.reachcost(n)
        #Loop through all the nodes
        for i in range(n-1):
            #Check if the nodes are close enough to node n and if the edge between them is free.
            #If that is the case: check if connecting them would improve the reachcost of n and change the parent to the node if so
            if distances[i] < self.d_cheaper:
                if self.edgefree(n, i):
                    cost = self.reachcost(i) + self.distance(n, i)
                    if cost < lowestcost:
                        cheapestparent = i
                        lowestcost = cost
        return cheapestparent

    def rewirecheck(self, n, distances):
        '''
        Function that rewires nodes (connect to node n as a parent) if that reduces the cost to reach a node

        Inputs
        ------
            - n: the nodenumber of the node which can be a cheaper parent for other nodes (int)
            - distances: an array of length n which gives distances to each node from node n (float array of shape (n, ))
        ''' 

        #Loop through all the nodes 
        for i in range(n-1):
            #Check if the nodes are close enough to node n and if the edge between them is free.
            #If that is the case: check if connecting them would improve the reachcost of the node and change the parent to n if so
            if distances[i] < self.d_cheaper:
                if self.edgefree(i, n):
                    newcost = self.reachcost(n) + self.distance(n, i)
                    if newcost < self.reachcost(i):
                        self.parent[i] = n   


    ####### Main functions used while iterating #######

    def randomsample(self):
        '''
        Function that takes a random sample in the map
                
        Outputs
        ------
            - x: returns a random x coordinate in the map (int)
            - y: returns a random y coordinate in the map (int)
            - z: returns a random z coordinate in the map (int)
        ''' 

        x = np.random.randint(0, self.mapw)
        y = np.random.randint(0, self.mapd)
        z = np.random.randint(0, self.maph)
        return (x, y, z)

    def expand(self):
        '''
        Function that expands the graph by taking a random sample and perform the main steps of RRT*
        ''' 

        #Take random sample as new node
        (x, y, z) = self.randomsample()
        #Add node to nodelist
        self.addnode(x, y, z)
        #Calculate distances to other nodes and nearest node
        n = len(self.x) - 1
        distances = self.distances(n)
        nnear = np.argmin(distances)

        #Check if node is free and close to other nodes. If this is the case, add the node. If not remove the node
        if self.edgefree(nnear, n) and (self.distance(n, nnear) <= self.d_search):
            self.addedge(nnear)
            
            #Performing RRT* steps: searching for a cheaper node to connect to (and reconnect if necessary) and do a rewire check
            cheapernode = self.cheapernodes(n, distances)
            if cheapernode != nnear:
                self.removeedge(n)
                self.addedge(cheapernode)

            self.rewirecheck(n, distances)

            #Check if the goal is found (by being close to the goal)
            if self.goaldistance(n) <= self.d_goal:

                #Check if this is the first goal encounter
                if self.goalfound == False:
                    print("Goal found after", self.iters, "iterations! Searching for better path...")

                    #Make a new node, located on the endpoint, and connect with the node n that was close to the goal
                    self.addnode(self.goal[0], self.goal[1], self.goal[2])
                    self.addedge(n)
                    self.goalindex = n+1

                    #Performing RRT* steps: searching for a cheaper node to connect the endnode to (and reconnect if necessary)
                    distances = self.distances(self.goalindex)
                    cheapernode = self.cheapernodes(self.goalindex, distances)
                    if cheapernode != n:
                        self.removeedge(n+1)
                        self.addedge(cheapernode)
                    
                    #Set goalfound parameter to True and compute the smoothpath to the goal (this is the path of the first encounter, for comparison)
                    self.goalfound = True
                    self.smoothpathfirst = self.getsmoothpath()

        else: 
            self.removenode(n)
 
    def iterations(self):
        '''
        Function that monitors the iterations and gives some information when the maximum is reached
        ''' 

        self.iters += 1
        if self.iters <= self.max_iter:
            return True
        else: 
            self.smoothpath = self.getsmoothpath()
            if self.goalfound:
                print("Goal found with", len(self.x), 'nodes after ', self.iters, " iterations.")
                print("Cost to reach goal: ", self.reachcost(self.goalindex))
            else:
                print("No goal found after ", self.iters, " iterations :( \nMade", len(self.x), "nodes.")
            return False
        

    ####### Functions that compute the final path from start to finish #######

    def makepath(self):
        '''
        Function that computes a list of nodenumbers and their coordinates that are used on the path from start to finish
        '''

        if self.goalfound: 
            #Compute a list of nodes in the path
            self.path = [self.goalindex]
            nparent = self.parent[self.goalindex]
            while nparent != 0:
                self.path.insert(0, nparent)
                nparent = self.parent[nparent]
            self.path.insert(0, 0)
            
            #Compute lists of x y and z coordinates of the path nodes
            self.pathx = [self.start[0]]
            self.pathy = [self.start[1]]
            self.pathz = [self.start[2]]
            for i in range(1, len(self.path)):
                #Check if nodes are not laying on top of each other to avoid errors in the spline
                if not ((self.x[self.path[i]] == self.x[self.path[i-1]]) and (self.y[self.path[i]] == self.y[self.path[i-1]])): 
                    self.pathx.append(self.x[self.path[i]])
                    self.pathy.append(self.y[self.path[i]])  
                    self.pathz.append(self.z[self.path[i]]) 
    

    def getsmoothpath(self):
        '''
        Function that makes a spline from the node coordinates used in the path from start to finish
                
        Outputs
        ------
            - smoothpath: coordinates of the splinepath from start to goal (float array of shape (3, ...))
        ''' 

        #Call makepath function to compute the x y and z coordinates of the waypoints (nodes)
        self.makepath()
        if self.goalfound:
            u = np.linspace(0, 1, int(self.reachcost(self.goalindex)))
            tck, _ = interpolate.splprep([self.pathx, self.pathy, self.pathz])
            smoothpath = interpolate.splev(u, tck)
            return smoothpath


    ####### Visualization #######

    def makemap(self, showpath = True, showrest = False, shownodes = False, showfirstpath = False):   
        '''
        Function that plots a map of the environment with the final smooth path
                
        Inputs
        ------
            - showpath: show the edges used to go from start to finish in the map if True(bool)
            - showrest: show the edges that are not used in the path from finish to start if True (bool)
            - shownodes: show the nodes in the map if True (bool)
            - showfirstpath: shows the first smooth path found from start to goal in the map if True (bool)
        ''' 

        #Initialize the figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        #Adding start and goal points to the map
        ax.scatter(self.start[0], self.start[1], self.start[2], s = 100)
        ax.scatter(self.goal[0], self.goal[1], self.goal[2], s=80, facecolors='none', edgecolors='r')

        #Adding obstacles to the map
        ax.scatter(self.obstacles[:, 0], self.obstacles[:, 1], self.obstacles[:, 2], c = 'b', alpha = 0.1)

        #Adding nodes to the map
        if shownodes:
            for i in range(1, len(self.x)):
                if (i in self.path) and showpath:
                    ax.scatter(self.x[i], self.y[i], self.z[i], color = '#FF0000')
                elif showrest: 
                    ax.scatter(self.x[i], self.y[i], self.z[i], color = '#000000')

        #Adding edges to the map
        if showrest or showpath:
            for i in range(1, len(self.parent)):
                x_values = [self.x[i], self.x[self.parent[i]]]
                y_values = [self.y[i], self.y[self.parent[i]]]
                z_values = [self.z[i], self.z[self.parent[i]]]
                if (i in self.path) and showpath:
                    ax.plot(x_values, y_values, z_values, color = '#FF0000')
                elif showrest:
                    ax.plot(x_values, y_values, z_values, color = '#000000')

        #Adding smooth paths to the map
        firstcoords = self.smoothpathfirst
        coords = self.smoothpath
        if showfirstpath and len(firstcoords) != 0:
            ax.scatter(firstcoords[0], firstcoords[1], firstcoords[2], color = '#00FFFF')
        if len(coords) != 0:
            ax.scatter(coords[0], coords[1], coords[2], color = '#00FF00')

        #Setting map scale
        ax.set_aspect('equal')

        #Showing map
        plt.show()

######################################################################################

class GridRRTstar2D:
    def __init__(self, startpos, goalpos, d_goal, d_search, d_cheaper, grid, mgrid, max_iter):
        '''
        Function that initializes all the class parameters

        Inputs
        ------
            - startpos: list of [x y] coordinates of the starting position (int list with length 2)
            - goalpos: list of [x y] coordinates of the goal position (int list with length 2)
            - d_goal: the maximum distance between a node to the goal with which we can say that the goal is found (float)
            - d_search: the maximum distance between a new node and existing nodes to be able to make the node (float)
            - d_cheaper: the maximum distance in which the algorithm searches for cheaper nodes to connect to (float)
            - grid: grid of zeros (free space) and ones (obstacles) of the map, used for visualization (int array of size (maxx, maxy))
            - mgrid: margin grid of zeros (free space) and ones (obstacles + margins) of the map (int array of size (maxx, maxy))
            - max_iter: the maximum amount of iterations that the algorithm should run (int)
        
        Parameters
        -----
            - self.start: list of [x y] coordinates of the starting position (list with length 2)
            - self.goal: list of [x y] coordinates of the goal position (list with length 2)
            - self.mapw: width of the map (int)
            - self.maph: height of the map (int)
            - self.grid: grid of zeros (free space) and ones (obstacles) of the map, used for visualization (int array of size (maxx, maxy))
            - self.mgrid: margin grid of zeros (free space) and ones (obstacles + margins) of the map (int array of size (maxx, maxy))
            - self.obstacle: coordinates of all ones (obstacle coordinates) in the grid (int array of size (..., 2))
            - self.x: list of x coordinates of nodes (int list of length n)
            - self.y: list of y coordinates of nodes (int list of length n)
            - self.parent: list of parent of each node (e.g. node 1 has a parent self.parent[1]) (int list of length n)
            - self.path: list of nodes that are used to get to the goal (int list of length ...)
            - self.smoothpath: coordinates of the final splinepath from start to goal (float array of shape (2, ...))
            - self.smoothpathfirst: coordinates of the first splinepath from start to goal (float array of shape (2, ...))
            - self.d_goal: the maximum distance between a node to the goal with which we can say that the goal is found (float)
            - self.d_search: the maximum distance between a new node and existing nodes to be able to make the node (float)
            - self.d_cheaper: the maximum distance in which the algorithm searches for cheaper nodes to connect to (float)
            - self.iters: iterations passed (int)
            - self.max_iter: the maximum amount of iterations that the algorithm should run (int)
            - self.goalfound: True if the goal is found, False if not (bool)
        ''' 
        #Initializing map parameters
        self.start = startpos
        self.goal = goalpos
        self.mapw, self.maph = grid.shape

        #Initializing obstacle parameters
        self.grid = grid
        self.mgrid = mgrid
        locs = np.where(grid == 1)
        self.obstacles = np.concatenate((locs[0].reshape(len(locs[0]), 1), locs[1].reshape(len(locs[0]), 1)), axis=1)

        #Initializing node coordinates with start position
        self.x = [startpos[0]]
        self.y = [startpos[1]]

        #Initializing edge parameters
        self.parent = [0]

        #Initializing path parameters
        self.path = [] 
        self.smoothpath = []
        self.smoothpathfirst = []

        #Initializing search parameters
        self.d_goal = d_goal
        self.d_search = d_search
        self.d_cheaper = d_cheaper

        #Initializing iterations
        self.iters= 0
        self.max_iter = max_iter

        #Initializing goalfound parameter
        self.goalfound = False


    ####### Functions to create and remove edges and nodes #######

    def addnode(self, x, y):        
        '''
        Function that adds a new node to the node lists from given coordinates

        Inputs
        ------
            - x: x coordinate of new node (int)
            - y: y coordinate of new node (int)
        '''  

        self.x.append(x)
        self.y.append(y)
    
    def removenode(self, n):
        '''
        Function that removes a node from the node lists

        Inputs
        ------
            - n: number of the node that needs to be removed (int)
        '''   

        self.x.pop(n)
        self.y.pop(n)

    def addedge(self, nparent):
        '''
        Function that adds a new edge by saving the nodenumber of the parent of the new node

        Inputs
        ------
            - nparent: the nodenumber of the parent of the new node (int)
        '''
        self.parent.append(nparent)
    
    def removeedge(self, n):
        '''
        Function that removes an edge by removing the nodenumber of the parent from the list

        Inputs
        ------
            - n: the nodenumber of the child which edge needs to be removed (int)
        '''  

        self.parent.pop(n)
    

    ####### Functions to handle obstacles in the map #######

    def isfree(self, x, y):
        '''
        Function that checks if a point is free by checking the grid (with margins)

        Inputs
        ------
            - x: x coordinate of point to check (int)
            - y: y coordinate of point to check (int)
        
        Outputs
        ------
            - Returns a bool which indicates if a position is free (True) or occupied (False) (bool)
        '''  

        if self.mgrid[x][y] == 0:
            return True
        else:
            return False

    #Function to check if the node is in the free space
    def nodefree(self, n):
        '''
        Function that checks if a node is free by checking the grid (with margins)

        Inputs
        ------
            - n: the nodenumber of the node that has to be checked (int)
        
        Outputs
        ------
            - Returns a bool which indicates if a node is free (True) or occupied (False) (bool)
        '''  

        return self.isfree(self.x[n], self.y[n])

    def edgefree(self, n1, n2):
        '''
        Function that achecks if an edge between node n1 and node n1 is free on the grid (with margins) by checking if interpolated points are free

        Inputs
        ------
            - n1: the nodenumber of the first node of the edge (int)
            - n2: the nodenumber of the second node of the edge (int)
        
        Outputs
        ------
            - Returns a bool which indicates if an edge is free (True) or occupied (False) (bool)
        ''' 
        #Interpolate with the same amount of steps as the distance between the 2 nodes
        steps = np.linspace(0, 1, round(self.distance(n1, n2)))
        for step in steps:
            x = round(self.x[n1] * step + self.x[n2] * (1-step))
            y = round(self.y[n1] * step + self.y[n2] * (1-step))
            if self.isfree(x, y) == False:
                return False
        return True

    
    ####### Functions to measure distances and costs ########

    def distance(self, n1, n2):
        '''
        Function that calculates the distance between node n1 and node n2

        Inputs
        ------
            - n1: the nodenumber of the first node (int)
            - n2: the nodenumber of the second node (int)
        
        Outputs
        ------
            - Returns the distance between the 2 given nodes (float)
        ''' 
        dx = (self.x[n1] - self.x[n2])
        dy = (self.y[n1] - self.y[n2])
        return np.sqrt(dx**2 + dy**2)

    def distances(self, n):
        '''
        Function that calculates the distances from node n to every other node

        Inputs
        ------
            - n: the nodenumber of the node which the distances should be calculated for (int)
        
        Outputs
        ------
            - distances: distances from node n to each node (float array of shape (n, ))
        ''' 

        distances = np.zeros(n)
        for i in range(n):
            distances[i] = self.distance(i, n)
        return distances

    def goaldistance(self, n):
        '''
        Function that calculates the distance between node n and the goal position

        Inputs
        ------
            - n: the nodenumber of the node that the distance should be calculated (int)
        
        Outputs
        ------
            - Returns the distance between node n and the goal position (float)
        ''' 

        dx = (self.x[n] - self.goal[0])
        dy = (self.y[n] - self.goal[1])
        return np.sqrt(dx**2 + dy**2)
    
    def reachcost(self, n):
        '''
        Function that calculates the cost to reach node n by checking the length of each edge required to reach the node from the start position

        Inputs
        ------
            - n: the nodenumber of the node for which the reach cost should be calculated (int)
        
        Outputs
        ------
            - cost: the total distance of reaching node n from the start position following the edges (float)
        ''' 

        nparent = self.parent[n]
        cost = 0
        while n != 0:
            cost += self.distance(n, nparent)
            n = nparent
            nparent = self.parent[n]
        return cost   

    
    #######Functions to make the path optimal (RRT*)#######
    
    def cheapernodes(self, n, distances):
        '''
        Function that searches for the cheapest node to connect to, which makes the path optimal

        Inputs
        ------
            - n: the nodenumber of the node for which the cheaper parent should be found (int)
            - distances: an array of length n which gives distances to each node from node n (float array of shape (n, ))
        
        Outputs
        ------
            - cheapestparent: the nodenumber of the cheapestparent for node n (int)
        ''' 

        #Initialize cheapest parent as current parent and lowestcost as current cost
        cheapestparent = self.parent[n]
        lowestcost = self.reachcost(n)
        #Loop through all the nodes
        for i in range(n-1):
            #Check if the nodes are close enough to node n and if the edge between them is free.
            #If that is the case: check if connecting them would improve the reachcost of n and change the parent to the node if so
            if distances[i] < self.d_cheaper:
                if self.edgefree(n, i):
                    cost = self.reachcost(i) + self.distance(n, i)
                    if cost < lowestcost:
                        cheapestparent = i
                        lowestcost = cost
        return cheapestparent
    
    def rewirecheck(self, n, distances):
        '''
        Function that rewires nodes (connect to node n as a parent) if that reduces the cost to reach a node

        Inputs
        ------
            - n: the nodenumber of the node which can be a cheaper parent for other nodes (int)
            - distances: an array of length n which gives distances to each node from node n (float array of shape (n, ))
        ''' 

        #Loop through all the nodes 
        for i in range(n-1):
            #Check if the nodes are close enough to node n and if the edge between them is free.
            #If that is the case: check if connecting them would improve the reachcost of the node and change the parent to n if so
            if distances[i] < self.d_cheaper:
                if self.edgefree(i, n):
                    newcost = self.reachcost(n) + self.distance(n, i)
                    if newcost < self.reachcost(i):
                        self.parent[i] = n       


    ####### Main functions used while iterating #######

    def randomsample(self):
        '''
        Function that takes a random sample in the map
                
        Outputs
        ------
            - x: returns a random x coordinate in the map (int)
            - y: returns a random y coordinate in the map (int)
        ''' 

        x = np.random.randint(0, self.mapw)
        y = np.random.randint(0, self.maph)
        return (x, y)

    #Function to expand the graph
    def expand(self):
        '''
        Function that expands the graph by taking a random sample and perform the main steps of RRT*
        ''' 

        #Take random sample as new node
        (x, y) = self.randomsample()
        #Add node to nodelist
        self.addnode(x, y)
        #Calculate distances to other nodes and nearest node
        n = len(self.x) - 1
        distances = self.distances(n)
        nnear = np.argmin(distances)

        #Check if node is free and close to other nodes. If this is the case, add the node. If not remove the node       
        if self.edgefree(nnear, n) and (self.distance(n, nnear) <= self.d_search):
            self.addedge(nnear)

            #Performing RRT* steps: searching for a cheaper node to connect to (and reconnect if necessary) and do a rewire check
            cheapernode = self.cheapernodes(n, distances)
            if cheapernode != nnear:
                self.removeedge(n)
                self.addedge(cheapernode)

            self.rewirecheck(n, distances)

            #Check if the goal is found (by being close to the goal)
            if self.goaldistance(n) <= self.d_goal:

                #Check if this is the first goal encounter
                if self.goalfound == False:
                    print("Goal found after", self.iters, "iterations! Searching for better path...")

                    #Make a new node, located on the endpoint, and connect with the node n that was close to the goal
                    self.addnode(self.goal[0], self.goal[1])
                    self.addedge(n)
                    self.goalindex = n+1

                    #Performing RRT* steps: searching for a cheaper node to connect the endnode to (and reconnect if necessary)
                    distances = self.distances(n+1)
                    cheapernode = self.cheapernodes(n+1, distances)
                    if cheapernode != n:
                        self.removeedge(n+1)
                        self.addedge(cheapernode)

                    #Set goalfound parameter to True and compute the smoothpath to the goal (this is the path of the first encounter, for comparison)                       
                    self.goalfound = True
                    self.smoothpathfirst = self.getsmoothpath()
                
        else: 
            self.removenode(n)

    def iterations(self):
        '''
        Function that monitors the iterations and gives some information when the maximum is reached
        ''' 

        self.iters += 1
        if self.iters < self.max_iter:
            return True
        else:
            self.smoothpath = self.getsmoothpath()
            if self.goalfound:
                print("Goal found with", len(self.x), 'nodes after ', self.iters, " iterations.")
                print("Cost to reach goal: ", self.reachcost(len(self.x)-1))
            else:
                print("No goal found after ", self.iters, " iterations :( \nMade", len(self.x), "nodes.")
            return False
    

    ####### Functions that compute the final path from start to finish #######

    def makepath(self):
        '''
        Function that computes a list of nodenumbers and their coordinates that are used on the path from start to finish
        '''

        if self.goalfound: 
            #Compute a list of nodes in the path
            self.path = [self.goalindex]
            nparent = self.parent[self.goalindex]
            while nparent != 0:
                self.path.insert(0, nparent)
                nparent = self.parent[nparent]
            self.path.insert(0, 0)
        
            #Compute lists of x y and z coordinates of the path nodes
            self.pathx = [self.start[0]]
            self.pathy = [self.start[1]]
            for i in range(1, len(self.path)):
                #Check if nodes are not laying on top of each other to avoid errors in the spline
                if not ((self.x[self.path[i]] == self.x[self.path[i-1]]) and (self.y[self.path[i]] == self.y[self.path[i-1]])): 
                    self.pathx.append(self.x[self.path[i]])
                    self.pathy.append(self.y[self.path[i]])               

    def getsmoothpath(self):
        '''
        Function that makes a spline from the node coordinates used in the path from start to finish
                
        Outputs
        ------
            - smoothpath: coordinates of the splinepath from start to goal (float array of shape (3, ...))
        ''' 

        self.makepath()
        if len(self.smoothpath) == 0:
            if self.goalfound:
                u = np.linspace(0, 1, int(self.reachcost(self.goalindex)))
                tck, _ = interpolate.splprep([self.pathx, self.pathy])
                smoothpath = interpolate.splev(u, tck)
                return smoothpath


    ####### Visualization #######

    def makemap(self, showpath = True, showrest = False, shownodes = False, showfirstpath = False):   
        '''
        Function that plots a map of the environment with the final smooth path
                
        Inputs
        ------
            - showpath: show the edges used to go from start to finish in the map if True(bool)
            - showrest: show the edges that are not used in the path from finish to start if True (bool)
            - shownodes: show the nodes in the map if True (bool)
            - showfirstpath: shows the first smooth path found from start to goal in the map if True (bool)
        ''' 

        #Initialize the figure
        fig, ax = plt.subplots(1, 1)

        #Adding start and goal points to the map
        ax.scatter(self.start[0], self.start[1], s = 100)
        ax.add_patch(plt.Circle(self.goal, self.d_goal, ec = '#FFA500', fill = False, lw = 5))
        
        #Adding obstacles to the map
        ax.scatter(self.obstacles[:, 0], self.obstacles[:, 1], c = 'b')
        
        #Adding nodes to the map
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
        firstcoords = self.smoothpathfirst
        coords = self.smoothpath
        if showfirstpath and len(firstcoords) != 0:
            ax.scatter(firstcoords[0], firstcoords[1], color = '#00FFFF')
        if len(coords) != 0:
            ax.scatter(coords[0], coords[1], color = '#00FF00')

        #Setting map scale
        ax.set_aspect('equal')
    
        #Showing map
        plt.show() 