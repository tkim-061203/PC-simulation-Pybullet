import pybullet as p
import numpy as np
import time
import pybullet_data
from pybullet_debuger import pybulletDebug  
from kinematic_model import robotKinematics
from gaitPlanner import trotGait
import pandas as pd

try:
    # Initialize connection and environment
    p.connect(p.GUI)  # or p.DIRECT
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.8)
    
    cubeStartPos = [0,0,0.2]
    FixedBase = False #if fixed no plane is imported
    
    if (FixedBase == False):
        p.loadURDF("plane.urdf")
       
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId = p.loadURDF("K://Desktop//dog//PC-simulation-Pybullet//src//4leggedRobot_with_sensors.urdf",
                       cubeStartPos, useFixedBase=FixedBase)
    
    jointIds = []
    paramIds = []
    time.sleep(0.5)

    """initial foot position"""
    #foot separation (Ydist = 0.16 -> tetta=0) and distance to floor
    Xdist = 0.20
    Ydist = 0.15
    height = 0.15
    #body frame to foot frame vector
    bodytoFeet0 = np.matrix([[ Xdist/2 , -Ydist/2 , -height],
                            [ Xdist/2 ,  Ydist/2 , -height],
                            [-Xdist/2 , -Ydist/2 , -height],
                            [-Xdist/2 ,  Ydist/2 , -height]])

    T = 0.5 #period of time (in seconds) of every step
    offset = np.array([0.5 , 0. , 0. , 0.5]) #defines the offset between each foot step in this order (FR,FL,BR,BL)
    
    for j in range(p.getNumJoints(boxId)):
        # p.changeDynamics(boxId, j, linearDamping=0, angularDamping=0)
        info = p.getJointInfo(boxId, j)
        jointName = info[1]
        jointType = info[2]
        jointIds.append(j)
       
    footFR_index = 3
    footFL_index = 7
    footBR_index = 11
    footBL_index = 15  
    
    pybulletDebug = pybulletDebug()
    robotKinematics = robotKinematics()
    trot = trotGait()
    
    # Your simulation loop
    while True:
        pos, orn, L, angle, Lrot, T = pybulletDebug.cam_and_robotstates(boxId)
        
        # Add other operations here
        # For example, calculating and setting leg positions:
        # FR_angles = robotKinematics.inverseKinematics(FR_pos)
        # p.setJointMotorControl2(boxId, footFR_index, p.POSITION_CONTROL, FR_angles[0])
        
        # Step simulation

        bodytoFeet = trot.loop(L , angle , Lrot , T , offset , bodytoFeet0)

    #####################################################################################
    #####   kinematics Model: Input body orientation, deviation and foot position    ####
    #####   and get the angles, neccesary to reach that position, for every joint    ####
        FR_angles, FL_angles, BR_angles, BL_angles , transformedBodytoFeet = robotKinematics.solve(orn , pos , bodytoFeet)
        print("FR:", FR_angles, "FL:", FL_angles, "BR:", BR_angles, "BL:", BL_angles)
        #move movable joints
        for i in range(0, footFR_index):
            p.setJointMotorControl2(boxId, i, p.POSITION_CONTROL, FR_angles[i - footFR_index])
        for i in range(footFR_index + 1, footFL_index):
            p.setJointMotorControl2(boxId, i, p.POSITION_CONTROL, FL_angles[i - footFL_index])
        for i in range(footFL_index + 1, footBR_index):
            p.setJointMotorControl2(boxId, i, p.POSITION_CONTROL, BR_angles[i - footBR_index])
        for i in range(footBR_index + 1, footBL_index):
            p.setJointMotorControl2(boxId, i, p.POSITION_CONTROL, BL_angles[i - footBL_index])

        p.stepSimulation()
        
        # Add a small delay to control simulation speed
        time.sleep(1/240)  # 240 Hz
        
        # Optional: Break condition
        # if some_condition:
        #     break
    
finally:
    # Disconnect only when the program exits
    if p.isConnected():
        p.disconnect()
        print("Disconnected from physics server")