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