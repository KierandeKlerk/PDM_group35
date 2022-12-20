import pybullet as p
import time
import pybullet_data
import pkg_resources
import numpy as np
grav = 10
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-grav)
planeId = p.loadURDF("plane.urdf")
obstacleId = p.loadURDF(pkg_resources.resource_filename('quadrotor_project', 'assets/drone_parcours.urdf'))
startPos = [0,0,0.8]
startOrientation = p.getQuaternionFromEuler([0,0,0])
droneid = p.loadURDF(pkg_resources.resource_filename('quadrotor_project', 'assets/cf2p_pointmass.urdf'), startPos, startOrientation)
m = 0.027

for i in range (10000):
    # p.applyExternalForce(droneid, -1, forceObj = [np.cos(3*i/700)/250, np.sin(3*i/700)/250, grav*m], posObj = [0,0,0], flags = p.LINK_FRAME)
    p.applyExternalForce(droneid, -1, forceObj = [0, 0, grav*m], posObj = [0,0,0], flags = p.LINK_FRAME)
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(droneid)
print(cubePos,cubeOrn)
p.disconnect()