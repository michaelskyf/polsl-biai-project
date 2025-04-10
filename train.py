import gymnasium as gym
import pybullet_envs_gymnasium  # This registers the PyBullet environments
from stable_baselines3 import PPO

def train_model():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create the environment with a suitable render mode (optional for faster training)
    # For training, it is common to use a non-rendering mode to speed up learning.
    env = gym.make("HumanoidBulletEnv-v0")
    
    # Define the PPO agent using a multi-layer perceptron (MLP) policy
    model = PPO("MlpPolicy", env, verbose=1)
    
    #model = PPO.load("ppo_humanoid", env=env)
    # Start training the model for a set number of timesteps
    model.learn(total_timesteps=1_000_000)
    
    # Save the trained model for later use
    model.save("ppo_humanoid")
    print("Training complete and model saved as ppo_humanoid.zip")
    
    # Close the environment to free resources
    env.close()

if __name__ == "__main__":
    train_model()
