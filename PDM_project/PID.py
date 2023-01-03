import pybullet as p
import time
import pybullet_data
import pkg_resources
import numpy as np

import sys
sys.path.append('../gym-pybullet-drones')
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
grav = 9.81
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-grav)
planeId = p.loadURDF("plane.urdf")
obstacleId = p.loadURDF(pkg_resources.resource_filename('quadrotor_project', 'assets/drone_parcours.urdf'))
startPos = [0,0,0.8]
startOrientation = p.getQuaternionFromEuler([0,0,0])
droneid = p.loadURDF(pkg_resources.resource_filename('quadrotor_project', 'assets/cf2x.urdf'), startPos, startOrientation)
m = 0.027

for i in range (1,14000): # 10000
    control_freq = 48 # from gym_pybullet_drones, but probably where the issue is!!!
    cur_pos, cur_quat = p.getBasePositionAndOrientation(droneid)
    cur_vel, cur_ang_vel = p.getBaseVelocity(droneid)
    control_timestep = i/control_freq
    PID_control = DSLPIDControl(DroneModel("cf2x"))
    # x_array = np.linspace(0, 1.6, 1500)
    # print(f'X ARRAY!!!!!!!!!!!!!!!!!!!!!!!! {x_array}')
    # y_array = np.linspace(0,0,1500)
    # z_array = np.linspace(0.8,0.8,1500)
    point_A = np.array([0,0,0.8])
    point_B = np.array([1.6, 0, 0.8])
    point_C = np.array([1.6, 3.2, 0.8])
    point_D = np.array([0.8, 3.2, 0.8])
    point_E = np.array([0.8, 1, 0.8])
    point_F = np.array([-0.1, 1, 0.8])
    point_G = np.array([-0.1, 4.6, 0.8])
    point_H = np.array([1.6, 4.6, 0.8])
    thrust_motors = []

    def interpolate(start_pos, final_target_pos, step, total_steps=1500):
        x_array = np.linspace(start_pos[0], final_target_pos[0], total_steps)
        print(x_array)
        y_array = np.linspace(start_pos[1], final_target_pos[1], total_steps)
        z_array = np.linspace(start_pos[2], final_target_pos[2], total_steps)
        current_target = np.array([x_array[step], y_array[step], z_array[step]])
        return  current_target
    
    if i < 1500:
        # target_pos = np.array([x_array[i], y_array[i], z_array[i]])
        target_pos = interpolate(point_A, point_B, i, 1500)
        print('positon: ' , cur_pos)
        # print(f'TARGET:::::: {target_pos}, {target_pos2}') # point_B
    if i > 1500:
        target_pos = interpolate(point_B, point_C, i-1499, 1500)
    if i > 3000:
        target_pos = target_pos = interpolate(point_C, point_D, i-2999, 1500)
    if i > 4500:
        target_pos = target_pos = interpolate(point_D, point_E, i-4499, 1500)
    if i > 6000:
        target_pos = target_pos = interpolate(point_E, point_F, i-5999, 1500)
    if i > 7500:
        target_pos = target_pos = interpolate(point_F, point_G, i-7499, 1500)
    if i > 9000:
        target_pos = target_pos = interpolate(point_G, point_H, i-8999, 1500)

    target_rpy = np.zeros(3)
    target_vel = np.zeros(3)
    target_rpy_rates = np.zeros(3)
    
    new_RPMS, pos_e, yaw_e = PID_control.computeControl(control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       )
    

    def RPM_to_thrust(RPM):
        # RPM = RPM*1.14 # compensate for steady state error when hovering
        thrust_motor = (9.919*10**(-8) * (RPM)**2)/4 # 0.109 * 10**(-6) * RPM**2 -210.6 * 10**(-6) *RPM + 0.154# https://wiki.bitcraze.io/misc:investigations:thrust
        return thrust_motor * (9.81/1000)

    for RPM in new_RPMS:
        print(' max_RPM = ', max(new_RPMS) )
        thrust_motors.append(RPM_to_thrust(RPM))


  
    # grav_comp = grav*m/4
    # p.applyExternalForce(droneid, -1, forceObj = [0,0,PWM_to_force(thrust)], posObj = [0,0,0], flags = p.LINK_FRAME) # right,left ; forward backward; up down
    p.applyExternalForce(droneid, -1, forceObj = [0,0,thrust_motors[0]], posObj = [0.028,0.028,0], flags = p.LINK_FRAME)
    print(f'motor thrust: {type(thrust_motors[0])}, {thrust_motors}')
    p.applyExternalForce(droneid, -1, forceObj = [0,0,thrust_motors[1]], posObj = [-0.028,0.028,0], flags = p.LINK_FRAME)
    p.applyExternalForce(droneid, -1, forceObj = [0,0,thrust_motors[2]], posObj = [-0.028,-0.028,0], flags = p.LINK_FRAME)
    p.applyExternalForce(droneid, -1, forceObj = [0,0,thrust_motors[3]], posObj = [0.028,-0.028,0], flags = p.LINK_FRAME)
    p.stepSimulation()
    print('current position: ', cur_pos)
    time.sleep(1./240.)

    # def RPM_to_PWM(RPM):
    #     PWM_new = []
    #     for i in RPM:
    #         PWM_i = (i -  380.8359)/0.04076521
    #         PWM_new.append(PWM_i)
    #     return np.array(PWM_new)

    # def PWM_to_force(PWM): # seems to be correct
    #     PWM = PWM # + thrust/4
    #     if PWM > 65535:
    #         PWM = 65535
    #     thrust_motor = 2.130295 * (10**(-11)) * PWM**2 + 1.032633 * (10**(-6)) * PWM + 5.484560 * 10**(-4) # research-collection.ethz.ch/handle/20.500.11850/214143
    #     print(PWM, thrust_motor)
    #     return thrust_motor

    # thrust, target_euler, pos_e = PID_control._dslPIDPositionControl(
    #                            control_timestep,
    #                            cur_pos,
    #                            cur_quat,
    #                            cur_vel,
    #                            target_pos,
    #                            target_rpy,
    #                            target_vel
    #                            )

    # # 65535
    # print(thrust)

    # PWMS = PID_control._dslPIDAttitudeControl(control_timestep,
    #                            thrust,
    #                            cur_quat,
    #                            target_euler,
    #                            target_rpy_rates
    #                            )

    # print(PWMS)
    # thrust_motors = []
    # # avg_PWM = sum(PWMS)/4

      # new_PWMS = RPM_to_PWM(new_RPMS)

    # for PWM in new_PWMS:
    #     # PWM = thrust + (PWM-avg_PWM)
    #     print('PWM: ', PWM)
    #     thrust_motors.append(PWM_to_force(PWM))

    # thrust_motors = np.array(thrust_motors)
    # print('test' , PWMS, (9.81*m), sum(thrust_motors), thrust_motors)
    # print(PID_control.MIXER_MATRIX)
    
    # print(i, cur_pos)
    # for tr in thrust_motors:
    #     print('===============================================')
    #     print(tr)
    #     tr += PWM_to_force(thrust)
    #     print(tr)
    # print(thrust_motors)
    # thrust_1 = thrust_motors[0]
    # x_error = target_pos[0] - cur_pos[0] 
    # y_error = target_pos[1] - cur_pos[1]
    # F_x = 0.01 * x_error - cur_vel[0]*0.02
    # F_y = 0.01 * y_error - cur_vel[1]*0.02
    # applied_force = [F_x, F_y, grav*m]

cubePos, cubeOrn = p.getBasePositionAndOrientation(droneid)
print(cubePos,cubeOrn)
p.disconnect()