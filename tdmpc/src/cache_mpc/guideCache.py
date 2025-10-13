import faiss
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import time
import pickle
import torch
import numpy as np
from pathlib import Path
import json
import re

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__, __TEST__ = 'cfgs', 'logs', 'tests'


class GuideCache:
    """
    Value-Guided State Cache using FAISS for efficient state-space partitioning.
    
    Automatically determines the number of clusters (nlist) based on task name complexity.
    Supports both IVF and LSH indexes and generates a detailed JSON report upon saving.
    """
    
    def __init__(
        self,
        task_name: str,
        state_dim: int,
        index_type: str = 'LSH',
        gamma: float = 0.99,
        nlist_override: Optional[int] = None
    ):
        """
        Args:
            task_name: Name of the environment task (e.g., 'humanoid-run'). Used to determine nlist.
            state_dim: Dimensionality of the state space (e.g., latent_dim for TD-MPC).
            index_type: The type of FAISS index to use ('IVF' or 'LSH').
            gamma: Discount factor for return computation.
            nlist_override: Manually specify nlist to override automatic selection.
        """
        if index_type not in ['IVF', 'LSH']:
            raise ValueError("index_type must be either 'IVF' or 'LSH'")
        
        self.task_name = task_name
        self.state_dim = state_dim
        self.index_type = index_type
        self.gamma = gamma
        self.num_bits = 16 # Default for LSH

        if nlist_override:
            self.nlist = nlist_override
        else:
            self.nlist = self._determine_nlist(task_name)
        
        print(f"Task: {self.task_name} -> Complexity-based nlist set to: {self.nlist}")

        # FAISS index
        if self.index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(state_dim)
            self.index = faiss.IndexIVFFlat(quantizer, state_dim, self.nlist)
        else:
            self.index = faiss.IndexLSH(state_dim, self.num_bits)
        
        # Cache storage & stats
        self.cache = {}
        self.build_stats = {} # To store statistics for the JSON report

    def _determine_nlist(self, task_name: str) -> int:
        """Determines nlist based on perceived task complexity."""
        # High complexity: Humanoids, dogs, quadrupeds
        if re.search('humanoid|dog|quadruped', task_name):
            return 4096
        # Medium complexity: Locomotion (cheetah, walker, etc.), complex manipulation
        elif re.search('cheetah|walker|hopper|fish|cup', task_name):
            return 2048
        # Low complexity: Simpler classic control and manipulation
        else:
            return 1024

    def collect_trajectories(self, env, agent, n_episodes: int = 500, max_steps: int = 1000, verbose: bool = True) -> List[List[Tuple]]:
        # ... (This method remains unchanged)
        """
        PHASE 1: Collect trajectories from the environment using the agent.
        """
        episodes = []
        for ep in range(n_episodes):
            trajectory = []
            obs = env.reset()
            done = False
            t = 0
            while not done and t < max_steps:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    embed = agent.model.h(obs_tensor).squeeze(0)
                    embed_np = embed.cpu().numpy().astype(np.float32)
                
                action = agent.plan(obs, eval_mode=True, step=ep, t0=(t == 0))
                next_obs, reward, done, _ = env.step(action.cpu().numpy())
                trajectory.append((embed_np, action.cpu().numpy(), float(reward)))
                obs = next_obs
                t += 1
            
            episodes.append(trajectory)
            if verbose and (ep + 1) % 10 == 0:
                avg_length = np.mean([len(ep) for ep in episodes[-10:]])
                print(f"Collected {ep + 1}/{n_episodes} episodes, avg length (last 10): {avg_length:.1f}")
        return episodes

    def compute_mc_returns(self, episodes: List[List[Tuple]]) -> List[Tuple[np.ndarray, float]]:
        # ... (This method remains unchanged)
        """
        Compute Monte Carlo returns for each state in the collected trajectories.
        """
        data = []
        for episode in episodes:
            G = 0.0
            for t in reversed(range(len(episode))):
                embed, _, reward = episode[t]
                G = reward + self.gamma * G
                data.append((np.array(embed, dtype=np.float32), float(G)))
        return data

    def build_cache(self, data: List[Tuple[np.ndarray, float]], verbose: bool = True):
        if not data:
            raise ValueError("No data provided for cache building")

        states = np.array([s for s, v in data], dtype=np.float32)
        
        self.build_stats['total_states_processed'] = len(states)

        if verbose:
            print(f"\nBuilding FAISS {self.index_type} index with {len(states)} states...")
        
        if self.index_type == 'IVF':
            if not self.index.is_trained: self.index.train(states)
        
        self.index.add(states)

        # ================================================================= #
        # CORRECTED LOGIC: 使用 index.quantizer.assign() 來取得 cluster ID  #
        # ================================================================= #
        if self.index_type == 'IVF':
            # 直接在僅包含 nlist 個簇中心的 quantizer 上搜尋最近的 1 個鄰居。
            # 回傳的索引 I 即為 cluster ID (0 到 nlist-1)。
            # D 是距離，I 是索引。我們只需要索引 I。
            _, bucket_indices = self.index.quantizer.search(states, 1)
            bucket_ids_flat = bucket_indices.flatten()
            get_bucket_id = lambda i: int(bucket_ids_flat[i])
        else: # LSH
            hash_codes = self.index.sa_encode(states)
            get_bucket_id = lambda i: int.from_bytes(hash_codes[i].tobytes(), byteorder='big')

        buckets = defaultdict(list)
        for i, (_, value) in enumerate(data):
            # 這裡的邏輯也已在上一版修正，確保呼叫 get_bucket_id(i)
            bucket_id = get_bucket_id(i)
            buckets[bucket_id].append(value)
        
        self.build_stats['num_buckets_created'] = len(buckets)
        bucket_sizes = [len(items) for items in buckets.values()]
        
        if not bucket_sizes:
            print("Warning: No buckets were created.")
            self.build_stats['bucket_size_stats'] = {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
            return

        self.build_stats['bucket_size_stats'] = {
            'mean': np.mean(bucket_sizes),
            'median': np.median(bucket_sizes),
            'std': np.std(bucket_sizes),
            'min': np.min(bucket_sizes),
            'max': np.max(bucket_sizes),
        }

        if verbose:
            print(f"Created {len(buckets)} buckets")
            # 這裡的輸出現在應該會接近 nlist 的值，例如 1024
            print(f"Bucket size - Mean: {self.build_stats['bucket_size_stats']['mean']:.1f}, "
                  f"Median: {self.build_stats['bucket_size_stats']['median']:.1f}, "
                  f"Std: {self.build_stats['bucket_size_stats']['std']:.1f}, "
                  f"Max: {self.build_stats['bucket_size_stats']['max']}")

        # ... (後續的統計與正規化部分無需修改) ...
        if verbose: print("\nAggregating buckets and computing normalization stats...")
        
        bucket_stats = {
            bucket_id: {
                'value_mean': np.mean(returns_in_bucket),
                'value_var': np.var(returns_in_bucket),
                'count': len(returns_in_bucket)
            } for bucket_id, returns_in_bucket in buckets.items()
        }
        
        all_means = [s['value_mean'] for s in bucket_stats.values()]
        all_vars = [s['value_var'] for s in bucket_stats.values()]
        
        self.build_stats['normalization_bounds'] = {
            'value_mean_min': float(np.min(all_means)),
            'value_mean_max': float(np.max(all_means)),
            'value_var_min': float(np.min(all_vars)),
            'value_var_max': float(np.max(all_vars)),
        }
        b_min, b_max = self.build_stats['normalization_bounds']['value_mean_min'], self.build_stats['normalization_bounds']['value_mean_max']
        v_min, v_max = self.build_stats['normalization_bounds']['value_var_min'], self.build_stats['normalization_bounds']['value_var_max']

        for bucket_id, stats in bucket_stats.items():
            norm_mean = (stats['value_mean'] - b_min) / (b_max - b_min) if (b_max > b_min) else 0.5
            norm_var = (stats['value_var'] - v_min) / (v_max - v_min) if (v_max > v_min) else 0.0
            self.cache[bucket_id] = {
                'value_mean_normalized': norm_mean,
                'value_var_normalized': norm_var,
                'value_mean': stats['value_mean'],
                'value_var': stats['value_var'],
                'count': stats['count']
            }
        
        if verbose: print(f"Cache statistics logged. Ready to save.")


    def query(self, state, v_thresh=0.7, var_thresh=0.3) -> bool:
        if not self.cache or not self.index.ntotal > 0:
            return False
        
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        query_state = np.array(state, dtype=np.float32).reshape(1, -1)
        
        # ================================================================= #
        # CORRECTED LOGIC: 同樣使用 index.quantizer.assign() 進行查詢       #
        # ================================================================= #
        if self.index_type == 'IVF':
            # 對查詢狀態找到它所屬的 cluster ID
            _, bucket_indices = self.index.quantizer.search(query_state, 1)
            bucket_id = int(bucket_indices[0][0])
        else: # LSH
            hash_code = self.index.sa_encode(query_state)[0]
            bucket_id = int.from_bytes(hash_code.tobytes(), byteorder='big')
            
        if bucket_id not in self.cache:
            return False
        
        entry = self.cache[bucket_id]
        return (entry['value_mean_normalized'] > v_thresh and 
                entry['value_var_normalized'] < var_thresh)


    def _get_save_path(self, path: str) -> str:
        return f"{path}_{self.index_type}"

    def _convert_for_json(self, data):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {key: self._convert_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_for_json(item) for item in data]
        else:
            return data

    def save(self, path: str):
        """Save cache, index, and a comprehensive JSON report to disk."""
        base_path = self._get_save_path(path)
        path_obj = Path(base_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Save FAISS index
        faiss.write_index(self.index, f"{base_path}_index.faiss")
        
        # 2. Save core data for loading
        metadata_to_load = {
            'cache': self.cache,
            'task_name': self.task_name,
            'state_dim': self.state_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'num_bits': self.num_bits,
            'gamma': self.gamma,
            'build_stats': self.build_stats
        }
        with open(f"{base_path}_cache.pkl", 'wb') as f:
            pickle.dump(metadata_to_load, f)
        
        # 3. Create and save human-readable JSON report
        report_data = {
            "configuration": {
                'task_name': self.task_name,
                'state_dim': self.state_dim,
                'index_type': self.index_type,
                'nlist': self.nlist,
                'num_bits': self.num_bits,
                'gamma': self.gamma,
            },
            "build_statistics": self.build_stats,
            "cache_data": self.cache
        }
        
        # Convert all numpy types for JSON compatibility
        report_data_json_safe = self._convert_for_json(report_data)

        json_path = f"{base_path}_report.json"
        with open(json_path, 'w') as f:
            json.dump(report_data_json_safe, f, indent=4)
        
        print(f"Cache saved to {base_path}_*")
        print(f"Human-readable report saved to {json_path}")


    def load(self, base_path: str):
        # ... (This method is slightly modified to load new structure)
        self.index = faiss.read_index(f"{base_path}_index.faiss")
        
        with open(f"{base_path}_cache.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.cache = metadata['cache']
        self.task_name = metadata.get('task_name', 'unknown')
        self.state_dim = metadata['state_dim']
        self.index_type = metadata.get('index_type', 'LSH')
        self.nlist = metadata.get('nlist', 256)
        self.num_bits = metadata.get('num_bits', 16)
        self.gamma = metadata['gamma']
        self.build_stats = metadata.get('build_stats', {})

        if self.index_type == 'IVF':
            self.index.nprobe = 10 
        
        print(f"Cache loaded from {base_path}_*")

# Add this method inside your GuideCache class
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache build process."""
        if not self.build_stats:
            return {
                "status": "Cache has not been built yet.",
                "num_buckets": len(self.cache)
            }
        
        # Flatten the nested dictionary for easier printing
        flat_stats = {
            'num_buckets_created': self.build_stats.get('num_buckets_created'),
            'total_states_processed': self.build_stats.get('total_states_processed'),
        }
        if 'bucket_size_stats' in self.build_stats:
            for key, val in self.build_stats['bucket_size_stats'].items():
                flat_stats[f'bucket_size_{key}'] = val

        if 'normalization_bounds' in self.build_stats:
             for key, val in self.build_stats['normalization_bounds'].items():
                flat_stats[key] = val
        
        return flat_stats