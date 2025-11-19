import os
import re
import numpy as np
import pandas as pd
import torch
import torch.linalg


def natural_sort_key(s):
    """
    Sort key function for natural sorting of strings with numbers.
    Converts 't164_action10' to be sorted after 't164_action9'.
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]


def save_planned_actions(cfg, elite_actions, elite_value, score, time, horizon, test_dir):
    """Save planned actions to CSV file."""
    # Ensure test directory exists
    os.makedirs(test_dir, exist_ok=True)
    
    # Prepare data for CSV
    data = []
    all_headers = set()  # Set to track all unique headers
    
    actions_cpu = elite_actions.cpu().numpy()  # shape: [horizon, num_elites, action_dim]
    value_cpu = elite_value.squeeze().cpu().numpy()  # shape: [num_elites]
    score_cpu = score.squeeze().cpu().numpy()  # shape: [num_elites]
    
    for i in range(actions_cpu.shape[1]):  # for each elite trajectory
        row_data = {}
        
        # Add time steps (t+1 to t+horizon)
        for h in range(actions_cpu.shape[0]):
            time_step = time + h
            for action_dim in range(cfg.action_dim):
                header = f't{time_step}_action{action_dim}'
                row_data[header] = actions_cpu[h, i, action_dim]
                all_headers.add(header)
        
        # Add value and score
        row_data['value'] = value_cpu[i]
        row_data['score'] = score_cpu[i]
        all_headers.add('value')
        all_headers.add('score')
        
        data.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure all rows have the same columns (align to headers)
    for header in all_headers:
        if header not in df.columns:
            df[header] = np.nan  # Add missing columns with NaN values
    
    # Reorder columns with natural sorting, but keep score and value at the end
    action_headers = sorted([h for h in all_headers if h not in ['value', 'score']], 
                           key=natural_sort_key)
    column_order = action_headers + ['value', 'score']
    df = df[column_order]
    
    # Save the DataFrame to CSV
    filename = f"planned-actions-time{time}.csv"
    filepath = os.path.join(test_dir, filename)
    df.to_csv(filepath, index=False)


def save_planned_states(cfg, elite_states, elite_value, score, time, horizon, test_dir):
    """Save planned states to CSV file."""
    # Ensure test directory exists
    os.makedirs(test_dir, exist_ok=True)
    
    # Prepare data for CSV
    data = []
    all_headers = set()  # Set to track all unique headers
    
    states_cpu = elite_states.cpu().numpy()  # shape: [num_trajectories, horizon, state_dim]
    value_cpu = elite_value.squeeze().cpu().numpy()  # shape: [num_trajectories]
    score_cpu = score.squeeze().cpu().numpy()  # shape: [num_trajectories]
    
    # Infer state_dim from the shape of elite_states tensor
    actual_state_dim = states_cpu.shape[2]
    
    for i in range(states_cpu.shape[0]):  # for each elite trajectory
        row_data = {}
        
        # Add time steps (t+1 to t+horizon)
        for h in range(states_cpu.shape[1]):
            time_step = time + h 
            for state_dim in range(actual_state_dim):
                header = f't{time_step}_state{state_dim}'
                row_data[header] = states_cpu[i, h, state_dim]
                all_headers.add(header)
        
        # Add value and score
        row_data['value'] = value_cpu[i]
        row_data['score'] = score_cpu[i]
        all_headers.add('value')
        all_headers.add('score')
        
        data.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure all rows have the same columns (align to headers)
    for header in all_headers:
        if header not in df.columns:
            df[header] = np.nan  # Add missing columns with NaN values
    
    # Reorder columns with natural sorting, but keep score and value at the end
    state_headers = sorted([h for h in all_headers if h not in ['value', 'score']], 
                          key=natural_sort_key)
    column_order = state_headers + ['value', 'score']
    df = df[column_order]
    
    # Save the DataFrame to CSV
    filename = f"planned-states-time{time}.csv"
    filepath = os.path.join(test_dir, filename)
    df.to_csv(filepath, index=False)


def store_trajectory(actions, states, values, time, trajectory_cache):
    """
    Store trajectory into cache as a list.
    
    Args:
        actions: Actions tensor [horizon, num_trajectories, action_dim]
        states: States tensor [num_trajectories, horizon+1, state_dim]
        values: Values tensor [num_trajectories]
        time: Current time step
        trajectory_cache: List to store trajectories
    """
    trajectory_cache.clear()  # Clear existing cache
    
    # Store all trajectories
    num_store = actions.shape[1]  # Store all trajectories
    
    for i in range(num_store):
        traj_data = {
            'actions': actions[:, i].detach(),  # 保持為 Tensor (GPU)
            'states': states[i].detach(),      # 保持為 Tensor (GPU)
            'value': values[i],         # 'value' 可能仍需 .item() 
            'time': time,
            'initial_state': states[i][0].detach(), # 保持為 Tensor (GPU)
            'trajectory_id': i
        }
        trajectory_cache.append(traj_data)


def find_matching_action_with_threshold(current_state, env_type, trajectory_cache, step_in_trajectory, device):
    """
    Find the matching action from cached trajectories for the current state.
    Uses an adaptive threshold based on environment type and state dimension.
    
    Args:
        current_state: Current state tensor.
        env_type: Environment type (e.g., 'cartpole-swingup', 'humanoid-run', etc.)
        trajectory_cache: List of cached trajectories.
        step_in_trajectory: Step index in the trajectory.
        device: Device (CPU/GPU).
    
    Returns:
        tuple: (action tensor or None, trajectory_index or None, distance or None)
    """
    if not trajectory_cache:
        return None, None, None
    
    # Infer the state dimension from current_state
    state_dim = current_state.squeeze().shape[-1]
    
    # Calculate adaptive threshold based on environment type and state dimension
    def get_adaptive_threshold(state_dim, env_type):
        """Calculate threshold based on state dimension and environment type."""
        # Extract base name from environment type
        env_base = env_type.split('-')[0] if '-' in env_type else env_type
        
        base_thresholds = {
            'cartpole': 5,     # 4D state space
            'acrobot': 5,      # 6D state space
            'humanoid': 150,    # 108D state space, stricter
            'walker': 15,      # 17D state space
            'cheetah': 25,     # 17D state space
            'cup': 5,          # 8D state space
            'dog': 5,         # High-dimensional state space
            'finger': 5,       # 9D state space
            'fish': 5,        # 24D state space
            'hopper': 5,      # 15D state space
            'quadruped': 5,   # High-dimensional state space
            'reacher': 5,      # 4-6D state space
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
    best_traj_idx = None
    min_distance = float('inf')
    current_state_np = current_state.squeeze().cpu().numpy()
    
    # Search through cached trajectories (now a list)
    for traj_idx, traj in enumerate(trajectory_cache):
        if step_in_trajectory < len(traj['states']):
            # Compare the state at the corresponding step in the trajectory
            traj_state = traj['states'][step_in_trajectory]
            distance = np.linalg.norm(current_state_np - traj_state)
            
            if distance < min_distance and distance < threshold:
                min_distance = distance
                best_match = traj
                best_traj_idx = traj_idx
    
    # If a sufficiently similar trajectory is found
    if best_match is not None:
        if step_in_trajectory < len(best_match['actions']):
            action = torch.tensor(best_match['actions'][step_in_trajectory], 
                                dtype=torch.float32, device=device)
            print(f"选择轨迹 #{best_traj_idx}, "
                  f"距离: {min_distance:.4f}")
            trajectory_cache.clear()
            return action
    
    return None


def find_matching_action(current_state, env_type, trajectory_cache, step_in_trajectory, device):
    """
    Find the matching action from cached trajectories for the current state (without threshold).
    
    Args:
        current_state: Current state tensor.
        env_type: Environment type.
        trajectory_cache: List of cached trajectories.
        step_in_trajectory: Step index in the trajectory.
        device: Device (CPU/GPU).
    
    Returns:
        tuple: (action tensor or None, trajectory_index or None, distance or None)
    """
    if not trajectory_cache:
        return None
        
    best_match = None
    best_traj_idx = None
    min_distance = float('inf')
    current_state_np = current_state.squeeze().cpu().numpy()  # Remove batch dimension
    
    # Search through cached trajectories (now a list)
    for traj_idx, traj in enumerate(trajectory_cache):
        if step_in_trajectory < len(traj['states']):
            # Compare the state at the corresponding step in the trajectory
            traj_state = traj['states'][step_in_trajectory]
            distance = np.linalg.norm(current_state_np - traj_state)
            
            if distance < min_distance:
                min_distance = distance
                best_match = traj
                best_traj_idx = traj_idx
    
    # If a matching trajectory is found
    if best_match is not None:
        if step_in_trajectory < len(best_match['actions']):
            action = torch.tensor(best_match['actions'][step_in_trajectory], 
                                dtype=torch.float32, device=device)
            # print(f"选择轨迹 #{best_traj_idx}, "
            #       f"距离: {min_distance:.4f}")
            trajectory_cache.clear()
            return action
    
    return None


def store_trajectory_vectorized(actions, states, values, time, trajectory_cache):
    """Stores elite trajectories (Actions:[E, H, A], States:[E, H+1, D]) and converts Values from [E, 1] to [E]."""
    trajectory_cache.clear()
    
    # E = num_trajectories
    num_trajectories = actions.shape[0] 
    
    batched_traj_data = {
        'actions': actions.detach(),               # [E, H, A]
        'states': states.detach(),                 # [E, H+1, D]
        'values': values.squeeze(1).detach(),      # [E]  <-- Squeeze from [E, 1]
        'time': time,                              # Python int
        'initial_states': states[:, 0].detach(),   # [E, D]
        'num_trajectories': num_trajectories
    }
    
    trajectory_cache.append(batched_traj_data)

    
def find_matching_action_vectorized_gpu(current_state, 
                                        env_type, 
                                        trajectory_cache, 
                                        step_in_trajectory,
                                        device):
    """Finds best action by L2-matching state [D] in batch [E, D] and correctly indexes action [E, H, A] as [E_idx, H_idx]."""
    if not trajectory_cache:
        return None
    
    batched_traj = trajectory_cache[0]

    # --- 1. Data Extraction and Validation ---
    current_state = current_state.squeeze() # [D]
    
    try:
        # Extract states at the current step: [E, H+1, D] -> [E, D]
        all_cached_states = batched_traj['states'][:, step_in_trajectory]
        
        # Actions shape: [E, H, A]. Check step_in_trajectory index against H (dim 1)
        _ = batched_traj['actions'][:, step_in_trajectory] 
    except IndexError:
        return None

    # --- 2. Vectorized L2 Distance Calculation on GPU ---
    diff = all_cached_states - current_state 
    distances = torch.linalg.norm(diff, dim=1) # [E]
    
    # --- 3. Find Minimum Distance ---
    min_distance, min_traj_idx = torch.min(distances, dim=0) 
    
    # --- 4. Extract Best Action and Cleanup ---

    # Actions shape: [E, H, A]. Correct indexing: [trajectory_index, step_index]
    best_action = batched_traj['actions'][min_traj_idx, step_in_trajectory] 
    # print(f"Match found in trajectory #{min_traj_idx.item()}. Distance: {min_distance.item():.4f}")
    trajectory_cache.clear()
    
    return best_action


def can_reuse(mathcing_fn_name, clock, reuse_interval, matching_fn, v_std, v_std_thresh):
    function_map = {
        "find_matching_action_with_interval": clock >= reuse_interval 
             and matching_fn is not None and v_std < v_std_thresh,
        "find_matching_action_with_threshold": clock >= reuse_interval 
            and matching_fn is not None  and v_std < v_std_thresh,
        "find_vectorized_action_with_interval": clock >= reuse_interval
            and matching_fn is not None  and v_std < v_std_thresh,
    }

    return function_map.get(mathcing_fn_name, False)
    