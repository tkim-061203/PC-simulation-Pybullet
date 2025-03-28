import numpy as np

def checkdomain(D):
    """Check if the domain is valid and handle edge cases more gracefully"""
    if D > 1:
        return 0.99
    elif D < -1:
        return -0.99
    return D

#this is based on this paper: 
#"https://www.researchgate.net/publication/320307716_Inverse_Kinematic_Analysis_Of_A_Quadruped_Robot"
"""
"using pybullet frame"
"  z                     "
"    |                   "
"    |                   "
"    |    /  y           "
"    |   /               "
"    |  /                "
"    | /                 "
"    |/____________  x       "
"""
#IK equations now written in pybullet frame.
def solve_R(coord, coxa, femur, tibia):
    try:
        # Calculate squared distance
        dist_squared = coord[1]**2 + (-coord[2])**2 + (-coord[0])**2
        
        # Check if target is reachable
        max_reach = coxa + femur + tibia
        if np.sqrt(dist_squared) > max_reach:
            print(f"Warning: Target position {coord} exceeds maximum reach {max_reach}")
            # Scale down the position to maximum reach
            scale = max_reach / np.sqrt(dist_squared) * 0.99
            coord = coord * scale
        
        D = (coord[1]**2 + (-coord[2])**2 - coxa**2 + (-coord[0])**2 - femur**2 - tibia**2)/(2*tibia*femur)
        D = checkdomain(D)
        
        gamma = np.arctan2(-np.sqrt(1-D**2), D)
        tetta = -np.arctan2(coord[2], coord[1]) - np.arctan2(np.sqrt(coord[1]**2 + (-coord[2])**2 - coxa**2), -coxa)
        alpha = np.arctan2(-coord[0], np.sqrt(coord[1]**2 + (-coord[2])**2 - coxa**2)) - np.arctan2(tibia*np.sin(gamma), femur + tibia*np.cos(gamma))
        
        # Check for NaN values
        if np.isnan([tetta, alpha, gamma]).any():
            print("Warning: IK produced NaN values, using fallback position")
            return np.array([0, np.pi/4, -np.pi/2])  # Safe fallback
            
        return np.array([-tetta, alpha, gamma])
        
    except Exception as e:
        print(f"IK solver error: {e}")
        return np.array([0, np.pi/4, -np.pi/2])  # Safe fallback

def solve_L(coord, coxa, femur, tibia):
    try:
        # Calculate squared distance
        dist_squared = coord[1]**2 + (-coord[2])**2 + (-coord[0])**2
        
        # Check if target is reachable
        max_reach = coxa + femur + tibia
        if np.sqrt(dist_squared) > max_reach:
            print(f"Warning: Target position {coord} exceeds maximum reach {max_reach}")
            # Scale down the position to maximum reach
            scale = max_reach / np.sqrt(dist_squared) * 0.99
            coord = coord * scale
            
        D = (coord[1]**2 + (-coord[2])**2 - coxa**2 + (-coord[0])**2 - femur**2 - tibia**2)/(2*tibia*femur)
        D = checkdomain(D)
        
        gamma = np.arctan2(-np.sqrt(1-D**2), D)
        tetta = -np.arctan2(coord[2], coord[1]) - np.arctan2(np.sqrt(coord[1]**2 + (-coord[2])**2 - coxa**2), coxa)
        alpha = np.arctan2(-coord[0], np.sqrt(coord[1]**2 + (-coord[2])**2 - coxa**2)) - np.arctan2(tibia*np.sin(gamma), femur + tibia*np.cos(gamma))
        
        # Check for NaN values
        if np.isnan([tetta, alpha, gamma]).any():
            print("Warning: IK produced NaN values, using fallback position")
            return np.array([0, np.pi/4, -np.pi/2])  # Safe fallback
            
        return np.array([-tetta, alpha, gamma])
        
    except Exception as e:
        print(f"IK solver error: {e}")
        return np.array([0, np.pi/4, -np.pi/2])  # Safe fallback