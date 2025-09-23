import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
torch.cuda.set_device(1)
print(f"Currently using device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
import numpy as np
import gym
import time
import random
from pathlib import Path
from src.cfg import parse_cfg
from src.env import make_env
from src.algorithm.tdmpc import TDMPC
from gym.wrappers import RecordVideo
from gym.wrappers.monitoring.video_recorder import VideoRecorder


torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__, __TEST__ = 'cfgs', 'logs', 'tests'


def create_test_directory(cfg):
    # Define the base test directory
    base_test_dir = Path().cwd() / __TEST__ / cfg.task / cfg.exp_name
    
    test_number = get_next_test_number(base_test_dir)
    cfg.seed = test_number
    
    test_dir = base_test_dir / str(test_number)
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test directory: {test_dir}")
    
    return test_dir

def get_next_test_number(base_dir):
    """Get the next available test number for unique directory naming."""
    if not base_dir.exists():
        return 1
    
    # List all directories that have numeric names
    existing_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not existing_dirs:
        return 1
    
    # Return the maximum existing directory number + 1
    return len(existing_dirs) + 1


def test_agent(env, agent, num_episodes, step, test_dir):
    """Test a trained agent and save planned actions for each step."""
    episode_rewards = []
    video_dir = test_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    # env.metadata
    if not hasattr(env, 'metadata') or env.metadata is None:
        env.metadata = {'render.modes': ['human', 'rgb_array']}
    
    for episode_idx in range(num_episodes):
        print(f"Testing episode {episode_idx + 1}/{num_episodes}")
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        
        # Record video
        video_path = video_dir / f"episode_{episode_idx + 1}.mp4"
        video_recorder = VideoRecorder(
            env, 
            path=str(video_path),
            enabled=True
        )

        # Create episode-specific directory
        episode_dir = test_dir / f"episode_{episode_idx + 1}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        while not done:
            video_recorder.capture_frame()

            # Plan action with test mode enabled
            action = agent.plan(
                obs, 
                eval_mode=True,      # Use deterministic policy
                step=step, 
                t0=(t == 0),
                store_traj=True,      # Enable CSV output
                time=t,                 # Current time step
                test_dir=episode_dir,  # Directory to save CSV files
                reuse=True
            )
            
            # Execute action in environment
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            t += 1
            
            if t % 10 == 0:  # Print progress every 10 steps
                print(f"  Step {t}, reward so far: {ep_reward:.2f}")
        
        # close the video
        video_recorder.close()
        episode_rewards.append(ep_reward)
        print(f"Episode {episode_idx + 1} completed with reward: {ep_reward:.2f}")
        print(f"Video saved to: {video_path}")
    
    avg_reward = np.nanmean(episode_rewards)
    print(f"\nTest completed!")
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"Test data saved to: {test_dir}")
    
    return avg_reward, episode_rewards


def load_trained_agent(cfg, model_path):
    """Load a trained agent from checkpoint."""
    agent = TDMPC(cfg)
    
    if os.path.exists(model_path):
        print(f"Loading trained model from: {model_path}")
        agent.load(model_path)
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Using randomly initialized model for testing.")
    
    return agent


def main():
    """Main testing function."""
    # Parse configuration
    cfg = parse_cfg(Path().cwd() / __CONFIG__)
    
    # Create test directory structure
    test_dir = create_test_directory(cfg)

    # Create environment
    env = make_env(cfg)
    
    # Load trained agent
    # Adjust this path to where your trained model is saved
    model_path = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / '1' / 'models' / 'model.pt'
    agent = load_trained_agent(cfg, model_path)
    
    # Test configuration
    num_test_episodes = getattr(cfg, 'test_episodes', 5)  # Default to 5 episodes if not specified
    test_step = cfg.train_steps  # Use final training step for consistency
    
    print(f"Starting test with {num_test_episodes} episodes...")
    print(f"Task: {cfg.task}")
    print(f"Experiment: {cfg.exp_name}")
    print(f"Seed: {cfg.seed}")
    
    # Run test
    start_time = time.time()
    avg_reward, episode_rewards = test_agent(
        env=env,
        agent=agent,
        num_episodes=num_test_episodes,
        step=test_step,
        test_dir=test_dir
    )
    test_time = time.time() - start_time
    
    # Save test summary
    summary_file = test_dir / 'test_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Test Summary\n")
        f.write(f"============\n")
        f.write(f"Task: {cfg.task}\n")
        f.write(f"Experiment: {cfg.exp_name}\n")
        f.write(f"Seed: {cfg.seed}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Number of episodes: {num_test_episodes}\n")
        f.write(f"Test time: {test_time:.2f} seconds\n")
        f.write(f"Average reward: {avg_reward:.2f}\n")
        f.write(f"Episode rewards: {episode_rewards}\n")
        f.write(f"Standard deviation: {np.std(episode_rewards):.2f}\n")
        f.write(f"Min reward: {np.min(episode_rewards):.2f}\n")
        f.write(f"Max reward: {np.max(episode_rewards):.2f}\n")
    
    print(f"Test summary saved to: {summary_file}")
    print("Testing completed successfully!")


if __name__ == '__main__':
    main()