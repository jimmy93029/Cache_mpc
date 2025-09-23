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
	存储轨迹到缓存
	"""
	trajectory_cache.clear() 
	
	# 存储所有轨迹
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
			

def find_matching_action(current_state, trajectory_cache, step_in_trajectory, device):
	"""
	找到与当前状态最匹配的缓存轨迹，返回对应的动作
	"""
	if not trajectory_cache:
		return None
		
	best_match = None
	min_distance = float('inf')
	current_state_np = current_state.squeeze().cpu().numpy()  # 移除batch维度
	
	# 搜索所有缓存轨迹
	for state_key, trajectories in trajectory_cache.items():
		for traj in trajectories:
			if step_in_trajectory < len(traj['states']):
				# 比较轨迹中对应步骤的状态
				traj_state = traj['states'][step_in_trajectory]
				distance = np.linalg.norm(current_state_np - traj_state)
				
				if distance < min_distance:
					min_distance = distance
					best_match = traj
	
	# 如果找到足够相似的轨迹
	if best_match is not None:
		if step_in_trajectory < len(best_match['actions']):
			action = torch.tensor(best_match['actions'][step_in_trajectory], 
								dtype=torch.float32, device=device)
			return action
	
	return None


def find_matching_action_with_adaptive_threshold(current_state, env_type, trajectory_cache, step_in_trajectory, device):
    """
    找到与当前状态最匹配的缓存轨迹，返回对应的动作
    使用基于环境类型和状态维度的自适应阈值
    
    Args:
        current_state: 当前状态张量
        env_type: 环境类型 (如 'cartpole-swingup', 'humanoid-run' 等)
        trajectory_cache: 轨迹缓存
        step_in_trajectory: 轨迹中的步骤索引
        device: 设备 (CPU/GPU)
    
    Returns:
        匹配的动作张量或None
    """
    if not trajectory_cache:
        return None
    
    # 从current_state推断状态维度
    state_dim = current_state.squeeze().shape[-1]
    
    # 根据环境类型和状态维度计算自适应阈值
    def get_adaptive_threshold(state_dim, env_type):
        """根据状态维度和环境类型计算阈值"""
        # 提取环境基础名称
        env_base = env_type.split('-')[0] if '-' in env_type else env_type
        
        base_thresholds = {
            'cartpole': 0.1,     # 4维状态空间
            'acrobot': 0.1,      # 6维状态空间
            'humanoid': 0.05,    # 108维状态空间，需要更严格
            'walker': 0.08,      # 17维状态空间
            'cheetah': 0.08,     # 17维状态空间
            'cup': 0.1,          # 8维状态空间
            'dog': 0.06,         # 高维状态空间
            'finger': 0.1,       # 9维状态空间
            'fish': 0.08,        # 24维状态空间
            'hopper': 0.08,      # 15维状态空间
            'quadruped': 0.06,   # 高维状态空间
            'reacher': 0.1,      # 4-6维状态空间
        }
        
        # 基础阈值
        base = base_thresholds.get(env_base, 0.1)
        
        # 维度调整：高维空间需要更小的阈值
        dim_factor = max(0.1, 1.0 / (state_dim ** 0.5))
        
        return base * dim_factor
    
    # 计算阈值
    threshold = get_adaptive_threshold(state_dim, env_type)
    
    # 执行匹配搜索
    best_match = None
    min_distance = float('inf')
    current_state_np = current_state.squeeze().cpu().numpy()
    
    # 搜索所有缓存轨迹
    for state_key, trajectories in trajectory_cache.items():
        for traj in trajectories:
            if step_in_trajectory < len(traj['states']):
                # 比较轨迹中对应步骤的状态
                traj_state = traj['states'][step_in_trajectory]
                distance = np.linalg.norm(current_state_np - traj_state)
                
                if distance < min_distance and distance < threshold:
                    min_distance = distance
                    best_match = traj
    
    # 如果找到足够相似的轨迹
    if best_match is not None:
        if step_in_trajectory < len(best_match['actions']):
            action = torch.tensor(best_match['actions'][step_in_trajectory], 
                                dtype=torch.float32, device=device)
            return action
    
    return None