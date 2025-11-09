import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored
import time
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)

	# --- Start of Timing Modifications ---
	
	# Initialize timers and counters
	total_cpu_time = 0.0
	total_gpu_time = 0.0
	total_actions = 0
	
	# Create CUDA events for precise GPU timing if available
	if torch.cuda.is_available():
		start_event = torch.cuda.Event(enable_timing=True)
		end_event = torch.cuda.Event(enable_timing=True)
	
	# --- End of Timing Modifications ---
	
	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		ep_rewards, ep_successes = [], []
		for i in range(cfg.eval_episodes):
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done:

				# --- Start of Timing Modifications ---
				
				# 1. Measure CPU (Wall-Clock) Time
				# This measures the total time spent in the function call from the CPU's perspective.
				cpu_start_time = time.perf_counter()

				# 2. Measure GPU Time
				if torch.cuda.is_available():
					start_event.record()

				# THIS IS THE LINE WE ARE TIMING
				action = agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
				
				if torch.cuda.is_available():
					end_event.record()
					# Synchronize to wait for the GPU operations to complete
					torch.cuda.synchronize()
					# elapsed_time returns time in milliseconds, convert to seconds
					total_gpu_time += start_event.elapsed_time(end_event) / 1000.0

				cpu_end_time = time.perf_counter()
				total_cpu_time += (cpu_end_time - cpu_start_time)
				total_actions += 1

				# --- End of Timing Modifications ---

				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if cfg.save_video:
					frames.append(env.render())
				# print(f"obs = {obs}")
				# print(f"reward = {reward}")
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if cfg.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))
	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))

	# --- Start of Timing Modifications ---
	
	# 4. Report the timing results at the end
	print(colored('--- Timing Results ---', 'cyan', attrs=['bold']))
	print(colored(f'Total Actions Timed: {total_actions}', 'cyan'))
	
	# CPU Timing Report
	avg_cpu_time_per_action = total_cpu_time / total_actions if total_actions > 0 else 0
	print(colored(f'CPU Time:', 'cyan'))
	print(colored(f'  - Total: {total_cpu_time:.3f} seconds', 'cyan'))
	print(colored(f'  - Average per action: {avg_cpu_time_per_action*1000:.2f} ms', 'cyan'))
	
	# GPU Timing Report
	if torch.cuda.is_available() and total_actions > 0:
		avg_gpu_time_per_action = total_gpu_time / total_actions
		print(colored(f'GPU Time:', 'cyan'))
		print(colored(f'  - Total: {total_gpu_time:.3f} seconds', 'cyan'))
		print(colored(f'  - Average per action: {avg_gpu_time_per_action*1000:.2f} ms', 'cyan'))

	# --- End of Timing Modifications ---


if __name__ == '__main__':
	evaluate()
