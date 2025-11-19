# Usage of Cache-based MPC

## Overview
This implementation extends TD-MPC with trajectory caching and reuse capabilities. The system can store successful trajectories and reuse similar actions when encountering similar states, potentially improving sample efficiency and performance.

## Configuration Parameters

### Core Cache Settings
- `reuse`: Enable/disable trajectory reuse (boolean)
- `reuse_interval`: Frequency of cache lookups (every N steps)
- `matching_fn`: Function used for trajectory matching
- `store_traj`: Enable trajectory storage during planning

### Available Matching Functions
- `find_matching_action_with_interval`: Simple state-based matching
- `find_matching_action_with_threshold`: Fixed threshold matching

### Supported Environments
- `acrobot-swingup`
- `cartpole-swingup`, `cartpole-swingup-sparse`
- `cheetah-run`
- `cup-catch`
- `dog-run`, `dog-trot`, `dog-walk`
- `finger-spin`, `finger-turn-hard`
- `fish-swim`
- `hopper-hop`
- `humanoid-run`, `humanoid-stand`, `humanoid-walk`
- `quadruped-run`, `quadruped-walk`
- `reacher-easy`, `reacher-hard`
- `walker-run`, `walker-walk`

## Instructions

### Training (train.py)
Train a model with trajectory caching enabled:

```bash
# Basic training with caching
nohup python src/train.py task=dog-run exp_name=dog-run reuse=true store_traj=true > logs/dog-run-horizon5.log &

# Training with specific cache settings
nohup python src/train.py task=humanoid-run exp_name=humanoid-run reuse=true reuse_interval=2 matching_fn=find_matching_action_with_threshold > logs/humanoid-run.log &

# Baseline training (no caching)
nohup python src/train.py task=cartpole-swingup exp_name=cartpole-swingup reuse=false > logs/cartpole-baseline.log &
```

### Testing (test.py)
Test a trained model with trajectory reuse:

```bash
# Basic testing with model auto-loading
nohup python -m src.cache_mpc.test task=humanoid-run exp_name=with_interval reuse=true reuse_interval=2 store_traj=false > tests/humanoid-run-reuse.log &

# Testing with specific cache configuration
nohup python -m src.cache_mpc.test task=dog-run exp_name=with_threshold reuse=true store_traj=false matching_fn=find_matching_action_with_threshold > tests/dog-run.log &

# Baseline testing (no reuse) 
nohup python -m src.cache_mpc.test task=cartpole-swingup exp_name=no-reuse store_traj=false reuse=false > tests/cartpole-swingup-no-reuse.log &
```

### Prepare Guide Cache (post_training.py)
Prepare guide cache
```bash
nohup python -m src.cache_mpc.post_training task=humanoid-run > logs/guide-humanoid-run.log &
```
Test
```bash
nohup python -m src.cache_mpc.test task=dog-run exp_name=with_guide reuse=true store_traj=false matching_fn=find_matching_action_with_guide > tests/dog-run.log &
```

### MPPI
```bash
nohup python -m src.cache_mpc.test task=cartpole-swingup exp_name=mppi > tests/cartpole-swingup-mppi.log &
```

## Model Path Structure
The system automatically loads trained models from:
```
tdmpc/logs/{task}/{modality}/{seed}/models/model.pt
```

For example:
- Model: `tdmpc/logs/cartpole-swingup/state/1/models/model.pt`
- Test command: `task=cartpole-swingup`

Make sure that the exp name match the name of the training dir, logs

## Experimental Configurations

### Configuration Examples

#### Adaptive Threshold Experiment
```yaml
# cfgs/test.yaml
store_traj: False
reuse: False
reuse_interval: 2
matching_fn: find_matching_action_with_interval
```

## Output Structure

### Test Results
```
tests/{task}/{test_number}/
├── episode_1/
│   └── planned_actions_t*.csv
├── episode_2/
│   └── planned_actions_t*.csv
├── videos/
│   ├── episode_1.mp4
│   └── episode_2.mp4
└── test_summary.txt
```

### Training Logs
```
logs/{task}/{modality}/{seed}/
├── models/
│   └── model.pt
├── config.yaml
└── training.log
```

## Performance Monitoring

### Key Metrics to Track
- Cache hit rate (printed during execution)
- Episode rewards
- Sample efficiency
- Planning time

### Log Analysis
Monitor cache effectiveness:
```bash
# Check cache hit rate
grep "Cache hit" tests/your-experiment.log | wc -l

# Monitor episode rewards
grep "Episode.*reward" tests/your-experiment.log
```

## Tips for Effective Usage

1. **Environment-specific tuning**: Use threshold matching for complex, high-dimensional environments
2. **Reuse interval**: Start with `reuse_interval=2`, adjust based on cache hit rates
3. **Baseline comparison**: Always run baseline experiments without caching for proper evaluation
4. **Multiple seeds**: Run experiments with different random seeds for statistical significance
5. **Video analysis**: Use generated videos to qualitatively assess behavior quality

## Troubleshooting

### Common Issues
- **Model not found**: Ensure the model path matches the expected structure
- **Low cache hit rate**: Try different matching functions or adjust reuse intervals
- **Memory issues**: Large trajectory caches may consume significant memory

### Debug Commands
```bash
# Check model exists
ls tdmpc/logs/{task}/state/{exp_name}/1/models/model.pt

# Monitor GPU usage
nvidia-smi

# Check cache performance in real-time
tail -f tests/your-experiment.log | grep "Cache"
```



Record 
```
nohup python tdmpc2/train.py task=dog-run > train_dog_run.log 2>&1 &
```