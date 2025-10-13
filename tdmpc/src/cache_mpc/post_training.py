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
# 您可以更改使用的GPU
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
from .guideCache import GuideCache # 假設您的 GuideCache class 在同一個資料夾


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
    """Example of how to use the updated GuideCache with TD-MPC."""
    
    # Parse config and setup
    cfg = parse_cfg(Path().cwd() / __CONFIG__)
    env = make_env(cfg)

    model_path = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / str(cfg.seed) / 'models' / 'model.pt'
    agent = load_trained_agent(cfg, model_path)
    
    # ================================================================= #
    # MODIFIED: Initialize cache using the new, smarter constructor     #
    # ================================================================= #
    cache = GuideCache(
        task_name=cfg.task,         # 傳入任務名稱以自動選擇 nlist
        state_dim=cfg.latent_dim,   # TD-MPC 的潛在維度
        index_type='LSH',           # 明確使用 IVF 索引以獲得更好性能
        gamma=cfg.discount          # 從 config 讀取 gamma 以保持一致
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
    
    # PHASE 3: Build cache (IVF clustering + aggregation + normalization)
    print("\n" + "=" * 60)
    print("PHASE 3: Building cache...")
    print("=" * 60)
    
    cache.build_cache(data, verbose=True)
    
    # Save cache, index, and the new JSON report
    save_path = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / str(cfg.seed) / "guide" / f"guide_cache_{cfg.task}"
    cache.save(str(save_path))

    test_time = time.time() - start_time
    print(f"Total cache build time = {test_time:.2f} seconds\n")
    
    # Test query
    print("\n" + "=" * 60)
    print("Testing query...")
    print("=" * 60)
    
    obs = env.reset()
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        embed = agent.model.h(obs_tensor)
    
    should_reuse = cache.query(embed)
    print(f"Query result for a random initial state: {should_reuse}")
    
    # ================================================================= #
    # MODIFIED: Get and print statistics from the new method            #
    # ================================================================= #
    stats = cache.get_cache_stats()
    print(f"\nCache Build Statistics (also in the .json report):")
    if stats:
        for k, v in stats.items():
            # 格式化輸出，使其更易讀
            if isinstance(v, float):
                print(f"  - {k}: {v:.4f}")
            else:
                print(f"  - {k}: {v}")
    else:
        print("  - No statistics available.")


if __name__ == "__main__":
    print("=" * 60)
    print("GuideCache Post-Training Builder for TD-MPC")
    print("=" * 60)
    # ================================================================= #
    # MODIFIED: Updated description to reflect new features             #
    # ================================================================= #
    print("\nThis module builds a value-guided state cache from a trained agent.")
    print("\nKey features:")
    print("  - Works with TD-MPC latent embeddings")
    print("  - Uses Monte Carlo returns for value estimation")
    print("  - Employs data-driven FAISS IVF for robust state partitioning")
    print("  - Automatically selects nlist (cluster count) based on task name")
    print("  - Generates a detailed, human-readable JSON report for analysis")
    print("=" * 60)
    example_usage()