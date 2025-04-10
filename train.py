import gymnasium as gym
import torch
import pybullet_envs_gymnasium
from stable_baselines3 import PPO

def train_model():
    # Create the environment with a suitable render mode (optional for faster training)
    # For training, it is common to use a non-rendering mode to speed up learning.
    env = gym.make("HumanoidBulletEnv-v0")
    print(torch.version.cuda)
    # Check if CUDA (GPU) is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the model with GPU (if available)
    model = PPO("MlpPolicy", env, verbose=1, device=device)
    
    # Start training the model for a set number of timesteps
    model.learn(total_timesteps=1_000_000)
    
    # Save the trained model for later use
    model.save("ppo_humanoid")
    print("Training complete and model saved as ppo_humanoid.zip")
    
    # Close the environment to free resources
    env.close()

if __name__ == "__main__":
    train_model()
