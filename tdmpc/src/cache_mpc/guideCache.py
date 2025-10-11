import faiss
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import pickle
import torch
import numpy as np
from pathlib import Path
import json


torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__, __TEST__ = 'cfgs', 'logs', 'tests'


class GuideCache:
    """
    Value-Guided State Cache using FAISS LSH.
    
    Uses Monte Carlo returns for value estimation and LSH for efficient retrieval.
    Returns True/False based on whether state meets value and stability thresholds.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_bits: int = 16,
        gamma: float = 0.99
    ):
        """
        Args:
            state_dim: Dimensionality of state space (latent_dim for TD-MPC)
            num_bits: Number of bits for LSH hashing (controls bucket size)
            gamma: Discount factor for return computation
        """
        self.state_dim = state_dim
        self.num_bits = num_bits
        self.gamma = gamma
        
        # FAISS LSH index
        self.index = faiss.IndexLSH(state_dim, num_bits)
        
        # Cache storage: bucket_id -> aggregated data
        self.cache = {}
        
        # For normalization
        self.return_min = None
        self.return_max = None
        
    def collect_trajectories(
        self,
        env,
        agent,
        n_episodes: int = 500,
        max_steps: int = 1000,
        verbose: bool = True
    ) -> List[List[Tuple]]:
        """
        PHASE 1: Collect trajectories from environment.
        
        Args:
            env: Environment
            agent: TD-MPC agent
            n_episodes: Number of episodes to collect (default 500)
            max_steps: Maximum steps per episode
            verbose: Print progress
            
        Returns:
            List of episodes, where each episode is a list of (embed, action, reward) tuples
        """
        episodes = []
        
        for ep in range(n_episodes):
            trajectory = []
            obs = env.reset()
            done = False
            t = 0
            
            while not done and t < max_steps:
                # Get current state embedding
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    embed = agent.model.h(obs_tensor)
                    
                    # Flatten if needed
                    if len(embed.shape) > 1:
                        embed = embed.squeeze(0)
                    
                    # Convert to numpy
                    embed_np = embed.cpu().numpy().astype(np.float32)
                
                # Get action from agent
                action = agent.plan(
                    obs, 
                    eval_mode=True,
                    step=ep, 
                    t0=(t == 0)
                )
                
                # Step environment
                next_obs, reward, done, _ = env.step(action.cpu().numpy())
                
                # Store transition: (current_state_embedding, action, reward)
                trajectory.append((embed_np, action.cpu().numpy(), float(reward)))
                
                # Update for next iteration
                obs = next_obs
                t += 1
            
            episodes.append(trajectory)
            
            if verbose and (ep + 1) % 10 == 0:
                avg_length = np.mean([len(ep) for ep in episodes[-10:]])
                print(f"Collected {ep + 1}/{n_episodes} episodes, "
                      f"avg length (last 10): {avg_length:.1f}")
        
        return episodes
    
    def compute_mc_returns(
        self,
        episodes: List[List[Tuple]]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Compute Monte Carlo returns for each state in trajectories (every-visit MC).
        
        Args:
            episodes: List of episodes from collect_trajectories
            
        Returns:
            List of (state_embedding, return) tuples
        """
        data = []
        
        for episode in episodes:
            # Backward pass to compute returns
            G = 0.0
            
            for t in reversed(range(len(episode))):
                embed, action, reward = episode[t]
                
                # Accumulate discounted return
                G = reward + self.gamma * G
                
                # Store: (state_embedding, return)
                # Ensure embed is numpy array
                if torch.is_tensor(embed):
                    embed = embed.cpu().numpy().astype(np.float32)
                else:
                    embed = np.array(embed, dtype=np.float32)
                
                data.append((embed, float(G)))
        
        return data
        
    def build_cache(
        self,
        data: List[Tuple[np.ndarray, float]],
        verbose: bool = True
    ):
        """
        PHASE 2 & 3: Build LSH index and aggregate per bucket.
        
        Args:
            data: List of (state_embedding, return) from compute_mc_returns
            verbose: Print statistics
        """
        if len(data) == 0:
            raise ValueError("No data provided for cache building")
        
        # Extract states and returns
        states = np.array([s for s, v in data], dtype=np.float32)
        returns = np.array([v for s, v in data], dtype=np.float32)
        
        # Store min/max for normalization (global values, used later for comparison)
        self.return_min = float(returns.min())
        self.return_max = float(returns.max())
        
        if verbose:
            print(f"\nReturn statistics:")
            print(f"  Min: {self.return_min:.2f}")
            print(f"  Max: {self.return_max:.2f}")
            print(f"  Mean: {returns.mean():.2f}")
            print(f"  Std: {returns.std():.2f}")
        
        # PHASE 2: Build FAISS LSH index
        if verbose:
            print(f"\nBuilding FAISS LSH index with {len(states)} states...")
        
        self.index.add(states)
        
        # Hash all states to get bucket assignments
        hash_codes = self.index.sa_encode(states)
        
        # Group by bucket (hash code)
        buckets = defaultdict(list)
        
        for i, (state, value) in enumerate(data):
            # Convert hash code to integer bucket ID
            bucket_id = int.from_bytes(hash_codes[i].tobytes(), byteorder='big')
            buckets[bucket_id].append((state, value))
        
        if verbose:
            print(f"Created {len(buckets)} buckets")
            bucket_sizes = [len(items) for items in buckets.values()]
            print(f"Bucket size - Mean: {np.mean(bucket_sizes):.1f}, "
                f"Median: {np.median(bucket_sizes):.1f}, "
                f"Max: {np.max(bucket_sizes)}")
        
        # PHASE 3: Aggregate per bucket with normalization across all buckets
        if verbose:
            print("\nAggregating buckets...")
        
        # First, compute the total value for each bucket
        total_values = []  # To store the sum of values in each bucket
        for bucket_id, items in buckets.items():
            values_in_bucket = np.array([v for s, v in items], dtype=np.float32)
            total_value = values_in_bucket.sum()  # Sum of rewards for the current bucket
            total_values.append(total_value)

        # Now normalize the aggregated values across all buckets
        total_values = np.array(total_values, dtype=np.float32)
        
        # Normalize aggregated values to [0, 1] across all buckets
        total_value_min = total_values.min()
        total_value_max = total_values.max()
        
        if total_value_max > total_value_min:
            normalized_total_values = (total_values - total_value_min) / \
                                    (total_value_max - total_value_min)
        else:
            normalized_total_values = np.ones_like(total_values)
        
        # Now store the normalized values in the cache for each bucket
        for idx, (bucket_id, items) in enumerate(buckets.items()):
            states_in_bucket = np.array([s for s, v in items], dtype=np.float32)
            values_in_bucket = np.array([v for s, v in items], dtype=np.float32)
            
            # Aggregate statistics for each bucket
            value_mean = float(values_in_bucket.mean())  # Mean reward in the bucket
            value_var = float(values_in_bucket.var())    # Variance of reward in the bucket
            
            # Representative state (centroid)
            state_rep = states_in_bucket.mean(axis=0).astype(np.float32)
            
            # Store in cache with normalized aggregated value
            self.cache[bucket_id] = {
                'state': state_rep,
                'value_mean': value_mean,
                'value_var': value_var,
                'total_value_normalized': float(normalized_total_values[idx]),
                'count': len(items)
            }
        
        if verbose:
            # Statistics on cache quality
            values_mean = [entry['value_mean'] for entry in self.cache.values()]
            values_var = [entry['value_var'] for entry in self.cache.values()]
            total_values_normalized = [entry['total_value_normalized'] for entry in self.cache.values()]
            
            print(f"\nCache statistics:")
            print(f"  Total buckets: {len(self.cache)}")
            print(f"  Normalized total value - Mean: {np.mean(total_values_normalized):.3f}, "
                f"Std: {np.std(total_values_normalized):.3f}")
            print(f"  Normalized value - Mean: {np.mean(values_mean):.3f}, "
                f"Std: {np.std(values_mean):.3f}")
            print(f"  Variance - Mean: {np.mean(values_var):.3f}, "
                f"Max: {np.max(values_var):.3f}")
            
            # How many buckets pass the filter?
            passed = sum(1 for e in self.cache.values() 
                        if e['value_mean'] > 0.7 and 
                        e['value_var'] < 0.3)
            print(f"  Buckets passing filter (v>{0.7}, var<{0.3}): "
                f"{passed}/{len(self.cache)} ({100*passed/len(self.cache):.1f}%)")

    
    def query(self, state, v_thresh=0.7, var_thresh=0.3, k: int = 1) -> bool:
        """
        Query cache to check if state is high-value and stable.
        
        Args:
            state: Current state embedding (torch.Tensor or np.ndarray)
            k: Number of nearest neighbors (unused, kept for compatibility)
            
        Returns:
            True if state meets value and variance thresholds, False otherwise
        """
        if len(self.cache) == 0:
            return False
        
        # Handle both tensor and numpy input
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        
        # Ensure numpy array
        state = np.array(state, dtype=np.float32)
        
        # Flatten if needed
        if len(state.shape) > 1:
            state = state.flatten()
        
        # Ensure 2D for FAISS [1, state_dim]
        query_state = state.reshape(1, -1)
        
        # Get hash code for query state
        hash_code = self.index.sa_encode(query_state)[0]
        bucket_id = int.from_bytes(hash_code.tobytes(), byteorder='big')
        
        # Check if bucket exists
        if bucket_id not in self.cache:
            return False
        
        entry = self.cache[bucket_id]
        
        # Return True if passes filter, False otherwise
        return (entry['value_mean'] > v_thresh and 
                entry['value_var'] < var_thresh)
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        if len(self.cache) == 0:
            return {}
        
        values_mean = [e['value_mean'] for e in self.cache.values()]
        values_var = [e['value_var'] for e in self.cache.values()]
        counts = [e['count'] for e in self.cache.values()]
        
        passed = sum(1 for e in self.cache.values() 
                    if e['value_mean'] > 0.7 and 
                       e['value_var'] < 0.3)
        
        return {
            'num_buckets': len(self.cache),
            'value_mean': np.mean(values_mean),
            'value_std': np.std(values_mean),
            'variance_mean': np.mean(values_var),
            'variance_max': np.max(values_var),
            'bucket_size_mean': np.mean(counts),
            'bucket_size_max': np.max(counts),
            'buckets_passing_filter': passed,
            'pass_rate': passed / len(self.cache) if len(self.cache) > 0 else 0,
            'return_min': self.return_min,
            'return_max': self.return_max,
        }
    
    def save(self, path: str):
        """Save cache to disk and also save metadata as a JSON file."""
        # Create directory if needed
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}_index.faiss")
        
        # Prepare metadata
        metadata = {
            'cache': self.cache,
            'state_dim': self.state_dim,
            'num_bits': self.num_bits,
            'gamma': self.gamma,
            'return_min': self.return_min,
            'return_max': self.return_max,
        }
        
        # Save cache as Pickle file
        with open(f"{path}_cache.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        # Convert the metadata (with NumPy arrays converted to lists) to JSON
        metadata_for_json = self.convert_ndarray_to_list(metadata)

        # Save metadata as JSON file
        json_file_path = f"{path}_cache.json"
        with open(json_file_path, 'w') as json_file:
            json.dump(metadata_for_json, json_file, indent=4)
        
        print(f"Cache saved to {path}_*")
        print(f"Metadata saved to {json_file_path}")
        
    def convert_ndarray_to_list(self, data):
        """Convert NumPy arrays inside the metadata to lists for JSON serialization."""
        if isinstance(data, np.ndarray):
            return data.tolist()  # Convert ndarray to list
        elif isinstance(data, dict):
            return {key: self.convert_ndarray_to_list(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_ndarray_to_list(item) for item in data]
        else:
            return data  # Return as is if it's not a NumPy array

    def load(self, path: str):
        """Load cache from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}_index.faiss")
        
        # Load cache and metadata
        with open(f"{path}_cache.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.cache = metadata['cache']
        self.state_dim = metadata['state_dim']
        self.num_bits = metadata['num_bits']
        self.gamma = metadata['gamma']
        self.return_min = metadata['return_min']
        self.return_max = metadata['return_max']
        
        print(f"Cache loaded from {path}_*")

