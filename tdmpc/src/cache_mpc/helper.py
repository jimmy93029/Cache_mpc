import pandas as pd
import os

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