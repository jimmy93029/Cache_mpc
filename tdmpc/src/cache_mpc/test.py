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
from gym.wrappers import RecordVideo
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
import gym
import time
import random
from pathlib import Path
from src.cfg import parse_cfg
from src.env import make_env
from src.algorithm.tdmpc import TDMPC
from src.algorithm.tdmpc_mppi import TDMPC_MPPI


torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__, __TEST__ = 'cfgs', 'logs', 'tests'


def create_test_directory(cfg):
    """a                                 
    Creates the test directory with the structure:
    tests/{task}/{exp_name}/{test_seed}/
    """
    # --- MODIFIED LINE ---
    # The directory structure is now grouped by experiment name first.
    test_dir = Path().cwd() / __TEST__ / cfg.task / cfg.exp_name / str(cfg.test_seed)
    
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test directory created: {test_dir}")
    
    return test_dir


def set_seed(seed):
    """Sets the seed for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False    # Turns off cuDNN auto-tuner to make results reproducible


def test_agent(env, agent, num_episodes, step, test_dir):
    """Test a trained agent and save planned actions for each step."""
    episode_rewards = []
    video_dir = test_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    if not hasattr(env, 'metadata') or env.metadata is None:
        env.metadata = {'render.modes': ['human', 'rgb_array']}
    
    for episode_idx in range(num_episodes):
        print(f"Testing episode {episode_idx + 1}/{num_episodes}")
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        
        video_path = video_dir / f"episode_{episode_idx + 1}.mp4"
        video_recorder = VideoRecorder(env, path=str(video_path), enabled=True)

        episode_dir = test_dir / f"episode_{episode_idx + 1}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        while not done:
            video_recorder.capture_frame()
            action = agent.plan(
                obs, 
                eval_mode=True,
                step=step, 
                t0=(t == 0),
                time=t,
                test_dir=episode_dir,
            )
            
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            t += 1
            
            if t % 10 == 0:
                print(f"  Step {t}, reward so far: {ep_reward:.2f}")
        
        video_recorder.close()
        episode_rewards.append(ep_reward)
        print(f"Episode {episode_idx + 1} completed with reward: {ep_reward:.2f}")
        print(f"Video saved to: {video_path}")
    
    avg_reward = np.nanmean(episode_rewards)
    print(f"\nTest for seed completed!")
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    
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


def run_test_for_seed(cfg, seed):
    """
    Runs a complete test for a single given seed.
    This function encapsulates the logic from the original main().
    """
    print(f"\n{'='*60}\nRunning test for seed: {seed}\n{'='*60}")

    # 1. Override the config's test_seed with the one for this run
    cfg.test_seed = seed

    # 2. Create seed-specific directory and set the seed
    test_dir = create_test_directory(cfg)
    set_seed(cfg.test_seed)

    # 3. Create environment
    env = make_env(cfg)
    
    # 4. Load the *same* trained agent for each test run
    model_path = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / str(cfg.seed) / 'models' / 'model.pt'
    agent = load_trained_agent(cfg, model_path)
    
    # Test configuration
    num_test_episodes = getattr(cfg, 'test_episodes', 5)
    test_step = cfg.train_steps
    
    print(f"Starting test with {num_test_episodes} episodes...")
    print(f"Task: {cfg.task}")
    print(f"Experiment: {cfg.exp_name}")
    print(f"Test Seed: {cfg.test_seed}")
    
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

    # Save test summary for this specific seed
    summary_file = test_dir / 'test_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Test Summary for Seed: {seed}\n")
        f.write(f"============\n")
        f.write(f"Task: {cfg.task}\n")
        f.write(f"Experiment: {cfg.exp_name}\n")
        f.write(f"Training Seed (of model): {cfg.seed}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Number of episodes: {num_test_episodes}\n")
        f.write(f"store_traj: {cfg.store_traj}\n")
        f.write(f"reuse: {cfg.reuse}\n")
        f.write(f"reuse_interval: {cfg.reuse_interval}\n")
        f.write(f"matching_fn: {cfg.matching_fn}\n")
        f.write(f"Test time: {test_time:.2f} seconds\n")
        f.write(f"Average reward: {avg_reward:.2f}\n")
        f.write(f"Episode rewards: {episode_rewards}\n")
        f.write(f"Standard deviation: {np.std(episode_rewards):.2f}\n")
        f.write(f"Min reward: {np.min(episode_rewards):.2f}\n")
        f.write(f"Max reward: {np.max(episode_rewards):.2f}\n")
    
    print(f"Test summary for seed {seed} saved to: {summary_file}")
    
    # Return the average reward for final summary
    return avg_reward, test_time


def main():
    """Main testing function to loop over multiple seeds."""
    # Parse the base configuration once
    cfg = parse_cfg(Path().cwd() / __CONFIG__)
    
    # Define the list of seeds you want to test over
    test_seeds = [1, 2, 3, 4, 5]
    all_rewards = {}
    all_times = {}

    for seed in test_seeds:
        avg_reward, test_time = run_test_for_seed(cfg, seed)
        all_rewards[seed] = avg_reward
        all_times[seed] = test_time
    
    # Print a final summary of all runs
    print(f"\n\n{'='*60}\nOverall Test Summary\n{'='*60}")
    print(f"Task: {cfg.task}")
    print(f"Experiment: {cfg.exp_name}")
    print(f"Tested seeds: {test_seeds}")
    print("-" * 60)
    for seed, reward in all_rewards.items():
        print(f"  - Seed {seed}: Average Reward = {reward:.2f}: test time = {all_times[seed]:.2f}")
    
    rewards = list(all_rewards.values())
    times = list(all_times.values())
    print("-" * 60)
    print(f"Mean reward over all seeds: {np.mean(rewards):.2f}")
    print(f"Std dev over all seeds: {np.std(rewards):.2f}")
    print(f"Elapsed time over all seeds: {np.mean(times)}")
    print("\nTesting completed successfully for all seeds!")


if __name__ == '__main__':
    main()