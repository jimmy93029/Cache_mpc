import pandas as pd
import os
import numpy as np
import torch


def save_planned_actions(cfg, elite_actions, elite_value, score, time, horizon, test_dir):
    """Save planned actions to CSV file."""
    # Ensure test directory exists
    os.makedirs(test_dir, exist_ok=True)
    
    # Prepare data for CSV
    data = []
    elite_actions_cpu = elite_actions.cpu().numpy()  # shape: [horizon, num_elites, action_dim]
    elite_value_cpu = elite_value.squeeze().cpu().numpy()  # shape: [num_elites]
    score_cpu = score.squeeze().cpu().numpy()  # shape: [num_elites]
    
    for i in range(elite_actions_cpu.shape[1]):  # for each elite trajectory
        row_data = {}
        
        # Add time steps (t+1 to t+horizon)
        for h in range(horizon):
            time_step = time + h + 1
            for action_dim in range(cfg.action_dim):
                row_data[f't{time_step}_action{action_dim}'] = elite_actions_cpu[h, i, action_dim]
        
        # Add value and score
        row_data['value'] = elite_value_cpu[i]
        row_data['score'] = score_cpu[i]
        
        data.append(row_data)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    filename = f"planned-actions-time{time}.csv"
    filepath = os.path.join(test_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved planned actions to {filepath}")


def store_trajectory(actions, states, values, time, trajectory_cache, state_to_key_func):
    """
    Store trajectory into cache.
    """
    trajectory_cache.clear() 
    
    # Store all trajectories
    num_store = actions.shape[1]  # Store all trajectories
    
    for i in range(num_store):
        traj_data = {
            'actions': actions[:, i].cpu().numpy(),  # [horizon, action_dim]
            'states': states[i].cpu().numpy(),  # 修正1: 直接存储第i个轨迹的完整状态序列 [horizon+1, state_dim]
            'value': values[i].item(),
            'time': time,
        }
        
        # 修正2: 使用第i个轨迹的初始状态(时间步0)作为键
        initial_state_key = state_to_key_func(states[i][0])  # 或者 states[i, 0]
        
        if initial_state_key not in trajectory_cache:
            trajectory_cache[initial_state_key] = []
        
        trajectory_cache[initial_state_key].append(traj_data)
            

def find_matching_action(current_state, env_type, trajectory_cache, step_in_trajectory, device):
    """
    Find the matching action from cached trajectories for the current state.
    """
    if not trajectory_cache:
        return None
        
    best_match = None
    min_distance = float('inf')
    current_state_np = current_state.squeeze().cpu().numpy()  # Remove batch dimension
    
    # Search through cached trajectories
    for state_key, trajectories in trajectory_cache.items():
        for traj in trajectories:
            if step_in_trajectory < len(traj['states']):
                # Compare the state at the corresponding step in the trajectory
                traj_state = traj['states'][step_in_trajectory]
                distance = np.linalg.norm(current_state_np - traj_state)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = traj
    
    # If a sufficiently similar trajectory is found
    if best_match is not None:
        if step_in_trajectory < len(best_match['actions']):
            action = torch.tensor(best_match['actions'][step_in_trajectory], 
                                dtype=torch.float32, device=device)
            return action
    
    return None


def find_matching_action_with_threshold(current_state, env_type, trajectory_cache, step_in_trajectory, device):
    """
    Find the matching action from cached trajectories for the current state.
    Uses an adaptive threshold based on environment type and state dimension.
    
    Args:
        current_state: Current state tensor.
        env_type: Environment type (e.g., 'cartpole-swingup', 'humanoid-run', etc.)
        trajectory_cache: Cached trajectories.
        step_in_trajectory: Step index in the trajectory.
        device: Device (CPU/GPU).
    
    Returns:
        Matching action tensor or None.
    """
    if not trajectory_cache:
        return None
    
    # Infer the state dimension from current_state
    state_dim = current_state.squeeze().shape[-1]
    
    # Calculate adaptive threshold based on environment type and state dimension
    def get_adaptive_threshold(state_dim, env_type):
        """Calculate threshold based on state dimension and environment type."""
        # Extract base name from environment type
        env_base = env_type.split('-')[0] if '-' in env_type else env_type
        
        base_thresholds = {
            'cartpole': 0.1,     # 4D state space
            'acrobot': 0.1,      # 6D state space
            'humanoid': 0.05,    # 108D state space, stricter
            'walker': 0.08,      # 17D state space
            'cheetah': 0.08,     # 17D state space
            'cup': 0.1,          # 8D state space
            'dog': 0.06,         # High-dimensional state space
            'finger': 0.1,       # 9D state space
            'fish': 0.08,        # 24D state space
            'hopper': 0.08,      # 15D state space
            'quadruped': 0.06,   # High-dimensional state space
            'reacher': 0.1,      # 4-6D state space
        }
        
        # Base threshold
        base = base_thresholds.get(env_base, 0.1)
        
        # Dimensionality adjustment: higher-dimensional spaces need a smaller threshold
        dim_factor = max(0.1, 1.0 / (state_dim ** 0.5))
        
        return base * dim_factor
    
    # Calculate threshold
    threshold = get_adaptive_threshold(state_dim, env_type)
    
    # Perform matching search
    best_match = None
    min_distance = float('inf')
    current_state_np = current_state.squeeze().cpu().numpy()
    
    # Search through cached trajectories
    for state_key, trajectories in trajectory_cache.items():
        for traj in trajectories:
            if step_in_trajectory < len(traj['states']):
                # Compare the state at the corresponding step in the trajectory
                traj_state = traj['states'][step_in_trajectory]
                distance = np.linalg.norm(current_state_np - traj_state)
                
                if distance < min_distance and distance < threshold:
                    min_distance = distance
                    best_match = traj
    
    # If a sufficiently similar trajectory is found
    if best_match is not None:
        if step_in_trajectory < len(best_match['actions']):
            action = torch.tensor(best_match['actions'][step_in_trajectory], 
                                dtype=torch.float32, device=device)
            print(f"find action = {action}")
            return action
    
    return None
