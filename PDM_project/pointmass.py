import pybullet as p
import time
import pybullet_data
import pkg_resources
#print(pkg_resources.resource_filename('quadrotor_project', 'assets/cf2p_pointmass.urdf'))
# print(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/cf2p.urdf'))
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
droneid = p.loadURDF(pkg_resources.resource_filename('quadrotor_project', 'assets/cf2p_pointmass.urdf'), startPos, startOrientation)
# droneid = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/cf2p.urdf'), startPos, startOrientation)
# droneid = p.loadURDF(pkg_resources.resource_filename('quadrotor_project', 'assets/cf2p.urdf'), startPos, startOrientation)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(droneid)
print(cubePos,cubeOrn)
p.disconnect()