from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from quadruped_env import QuadrupedBulletEnv
import time

def visualize():
    # Create environment with GUI
    env = QuadrupedBulletEnv(render=True)
    env = DummyVecEnv([lambda: env])
    
    # Load the trained model
    env = VecNormalize.load("vec_normalize.pkl", env)
    env.training = False  # Don't update running statistics
    env.norm_reward = False
    
    model = PPO.load("quadruped_final_model")
    
    obs = env.reset()
    
    # Run the trained policy
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        
        if dones:
            obs = env.reset()
        
        time.sleep(0.01)  # Add delay for better visualization

if __name__ == "__main__":
    visualize()
