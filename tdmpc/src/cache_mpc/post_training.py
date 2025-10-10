import faiss
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import pickle
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
from pathlib import Path
from src.cfg import parse_cfg
from src.env import make_env
from src.algorithm.tdmpc import TDMPC
from .guideCache import GuideCache


torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__, __TEST__ = 'cfgs', 'logs', 'tests'


# ============================================
# Usage Example
# ============================================

def load_trained_agent(cfg, model_path):
    """Load a trained agent from checkpoint."""
    agent = TDMPC(cfg)
    
    if os.path.exists(model_path):
        print(f"Loading trained model from: {model_path}")
        agent.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return agent


def example_usage():
    """Example of how to use GuideCache with TD-MPC."""
    
    # Parse config and setup
    cfg = parse_cfg(Path().cwd() / __CONFIG__)
    env = make_env(cfg)

    model_path = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / str(cfg.seed) / 'models' / 'model.pt'
    agent = load_trained_agent(cfg, model_path)
    
    # Initialize cache
    cache = GuideCache(
        state_dim=cfg.latent_dim,  # TD-MPC's latent dimension
        num_bits=16,
        gamma=0.99
    )
    
    start_time = time.time()
    # PHASE 1: Collect trajectories
    print("=" * 60)
    print("PHASE 1: Collecting trajectories...")
    print("=" * 60)
    
    episodes = cache.collect_trajectories(
        env=env,
        agent=agent,
        n_episodes=500,
        max_steps=1000
    )
    
    # PHASE 2: Compute MC returns
    print("\n" + "=" * 60)
    print("PHASE 2: Computing Monte Carlo returns...")
    print("=" * 60)
    
    data = cache.compute_mc_returns(episodes)
    print(f"Computed returns for {len(data)} state-return pairs")
    
    # PHASE 3: Build cache (LSH + aggregation + normalization)
    print("\n" + "=" * 60)
    print("PHASE 3: Building cache...")
    print("=" * 60)
    
    cache.build_cache(data, verbose=True)
    
    # Save cache
    save_path = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / str(cfg.seed) / "guide" / f"guide_cache_{cfg.task}"
    cache.save(str(save_path))

    test_time = time.time() - start_time
    print(f"test time = {test_time}\n")
    
    # Test query
    print("\n" + "=" * 60)
    print("Testing query...")
    print("=" * 60)
    
    obs = env.reset()
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        embed = agent.model.h(obs_tensor)
    
    should_reuse = cache.query(embed)
    print(f"Query result for random state: {should_reuse}")
    
    # Get statistics
    stats = cache.get_cache_stats()
    print(f"\nCache statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    print("GuideCache for TD-MPC")
    print("=" * 60)
    print("\nThis module implements value-guide state filtering")
    print("using FAISS LSH and Monte Carlo returns.")
    print("\nKey features:")
    print("  - Works with TD-MPC latent embeddings")
    print("  - MC returns (no Q-function needed)")
    print("  - FAISS LSH for O(d) retrieval")
    print("  - Normalized returns to [0, 1]")
    print("  - Returns True/False for reuse decision")
    print("\nSee example_usage() for how to use.")
    print("=" * 60)
    example_usage()
