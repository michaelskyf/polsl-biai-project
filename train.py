import gymnasium as gym
import pybullet_envs  # This registers the PyBullet environments
from stable_baselines3 import PPO

def train_model():
    # Create the environment with a suitable render mode (optional for faster training)
    # For training, it is common to use a non-rendering mode to speed up learning.
    env = gym.make("HumanoidBulletEnv-v0")
    
    # Define the PPO agent using a multi-layer perceptron (MLP) policy
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Start training the model for a set number of timesteps
    model.learn(total_timesteps=200_000)
    
    # Save the trained model for later use
    model.save("ppo_humanoid")
    print("Training complete and model saved as ppo_humanoid.zip")
    
    # Close the environment to free resources
    env.close()

if __name__ == "__main__":
    train_model()
