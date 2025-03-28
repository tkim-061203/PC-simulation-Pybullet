import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from kinematic_model import robotKinematics
from gaitPlanner import trotGait
import time

class QuadrupedBulletEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        
        # Initialize PyBullet
        self.render = render
        if self.render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Robot dimensions and limits
        self.robotKinematics = robotKinematics()
        self.trot = trotGait()
        
        # Conservative limits
        self.Xdist = 0.18
        self.Ydist = 0.13
        self.height = 0.18
        self.max_reach = 0.24  # Maximum reach of the leg
        
        # Position limits
        self.pos_limit = 0.1  # Maximum deviation from initial position
        
        # Initial foot positions
        self.bodytoFeet0 = np.matrix([
            [ self.Xdist/2, -self.Ydist/2, -self.height],
            [ self.Xdist/2,  self.Ydist/2, -self.height],
            [-self.Xdist/2, -self.Ydist/2, -self.height],
            [-self.Xdist/2,  self.Ydist/2, -self.height]
        ])
        
        # Movement parameters
        self.max_linear_velocity = 0.1  # Reduced from 0.2
        self.max_angular_velocity = 0.1  # Reduced from 0.2
        self.step_period = 0.6
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 29),
            high=np.array([np.inf] * 29),
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)
        
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("K://Desktop//dog//PC-simulation-Pybullet//src//4leggedRobot_with_sensors.urdf",
                               [0, 0, 0.2],
                               useFixedBase=False)
        
        self._reset_robot_position()
        
        observation = self._get_observation()
        info = {}
        robot_position, robot_orientation = p.getBasePositionAndOrientation(self.robot)
        p.resetDebugVisualizerCamera(
            cameraDistance=1,  # Distance from the robot
            cameraYaw=50,      # Yaw angle
            cameraPitch=-30,   # Pitch angle
            cameraTargetPosition=robot_position  # Track the robot position
        ) 
        return observation, info
    
    def _reset_robot_position(self):
        initial_angles = {
            'FR': [0, np.pi/4, -np.pi/2],
            'FL': [0, np.pi/4, -np.pi/2],
            'BR': [0, np.pi/4, -np.pi/2],
            'BL': [0, np.pi/4, -np.pi/2]
        }
        
        for i in range(12):
            leg_type = ['FR', 'FL', 'BR', 'BL'][i // 3]
            joint_index = i % 3
            angle = initial_angles[leg_type][joint_index]
            
            p.resetJointState(self.robot, i, angle)
            p.setJointMotorControl2(
                self.robot,
                i,
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=10.0,
                maxVelocity=8.0
            )
        
        for _ in range(100):
            p.stepSimulation()
    
    def _validate_foot_positions(self, positions):
        """Validate and clip foot positions to prevent excessive reaches"""
        positions = np.array(positions)
        for i in range(positions.shape[0]):
            pos = positions[i]
            dist = np.linalg.norm(pos)
            if dist > self.max_reach:
                positions[i] = pos * (self.max_reach / dist * 0.9)  # 90% of max reach for safety
        return positions
    
    def _get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot)
        orientation = np.array(p.getEulerFromQuaternion(orn))
        
        foot_contacts = []
        for i in [3, 7, 11, 15]:
            contact = len(p.getContactPoints(bodyA=self.robot, linkIndexA=i)) > 0
            foot_contacts.append(float(contact))
        
        joint_positions = []
        for i in range(12):
            state = p.getJointState(self.robot, i)
            joint_positions.append(state[0])
        
        observation = np.concatenate([
            np.array(pos),
            orientation,
            np.array(linear_vel),
            np.array(angular_vel),
            np.array(foot_contacts),
            np.array(joint_positions),
            np.array([pos[2]])
        ]).astype(np.float32)
        
        return observation
    
    def step(self, action):
        forward_vel = action[0] * self.max_linear_velocity
        lateral_vel = action[1] * self.max_linear_velocity
        rot_vel = action[2] * self.max_angular_velocity
        
        L = np.sqrt(forward_vel**2 + lateral_vel**2)
        L = np.clip(L, 0, self.max_linear_velocity)
        angle = np.rad2deg(np.arctan2(lateral_vel, forward_vel)) if L > 0.01 else 0
        
        try:
            # Get new foot positions with safety checks
            new_bodytoFeet = self.trot.loop(
                L, angle, rot_vel,
                self.step_period,
                np.array([0.5, 0., 0., 0.5]),
                self.bodytoFeet0
            )
            
            # Validate foot positions
            new_bodytoFeet = self._validate_foot_positions(new_bodytoFeet)
            
            # Calculate joint angles
            FR_angles, FL_angles, BR_angles, BL_angles, _ = self.robotKinematics.solve(
                np.zeros(3),
                np.zeros(3),
                new_bodytoFeet
            )
            
            # Apply joint angles
            all_angles = np.concatenate([FR_angles, FL_angles, BR_angles, BL_angles])
            for i in range(12):
                p.setJointMotorControl2(
                    self.robot,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=all_angles[i],
                    force=5.0,
                    maxVelocity=8.0
                )
                
        except Exception as e:
            print(f"Control error: {e}")
            self._reset_robot_position()
        
        for _ in range(4):
            p.stepSimulation()
        
        if self.render:
            time.sleep(0.01)
        
        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = self._is_terminated(observation)
        info = {}
        
        return observation, reward, terminated, False, info
    
    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot)
        
        # Height stability reward
        height_error = abs(pos[2] - 0.3)
        height_reward = -height_error * 2
        
        # Orientation stability reward
        orientation = np.array(p.getEulerFromQuaternion(orn))
        orientation_error = np.sum(np.abs(orientation))
        orientation_reward = -orientation_error
        
        # Forward velocity reward
        velocity_reward = linear_vel[0]
        
        # Contact reward
        contact_points = sum([len(p.getContactPoints(bodyA=self.robot, linkIndexA=i)) > 0 
                            for i in [3, 7, 11, 15]])
        contact_reward = contact_points * 0.2
        
        total_reward = height_reward + orientation_reward + velocity_reward + contact_reward
        return float(total_reward)
    
    def _is_terminated(self, observation):
        pos = observation[:3]
        orientation = observation[3:6]
        
        # Check height
        if pos[2] < 0.1:
            return True
        
        # Check orientation
        if np.abs(orientation).max() > 0.5:
            return True
            
        # Check position bounds
        if np.abs(pos[:2]).max() > self.pos_limit:
            return True
        
        return False
    
    def close(self):
        p.disconnect(self.physicsClient)