# Usage of Cache-based MPC

## Instructions
### test.py
example usage
if the model weight is in tdmpc/logs/cartpole-swingup/state/cartpole-swingup-horizon5/1/models
```
nohup python -m src.cache_mpc.test task=humanoid-run exp_name=humanoid-run-horizon5 horizon=5 > tests/humanoid-run-horizon5-reuse.log &
```

### train.py (the version here is different the one in the original repo)
```
nohup python src/train.py task=dog-run exp_name=dog-run horizon=5 > logs/dog-run-horizon5.log &
```