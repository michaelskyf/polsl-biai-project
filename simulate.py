import gymnasium as gym
import pybullet_envs_gymnasium
from stable_baselines3 import PPO
import time

def simulate_model():
    # Create the environment with visualization enabled.
    env = gym.make("HumanoidBulletEnv-v0", render_mode="human")
    
    # Load the previously trained model.
    model = PPO.load("ppo_humanoid")
    
    # Reset the environment
    observation, info = env.reset()
    
    while True:
        # Predict the next action using the trained model.
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
        
        # If the episode is done or truncated, reset the environment.
        if done or truncated:
            observation, info = env.reset()
        
        # Sleep to slow down the visualization (adjust as needed).
        time.sleep(1/240)

if __name__ == "__main__":
    simulate_model()
