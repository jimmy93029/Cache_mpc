# Usage of Cache-based MPC

## test.py
### usage
example usage
if the model weight is in tdmpc/logs/cartpole-swingup/state/cartpole-swingup-horizon5/1/models
```
nohup python -m src.cache_mpc.test task=humanoid-run exp_name=humanoid-id-run-horizon5 horizon=5 > tests/humanoid-run-horizon5-reuse.log &
```