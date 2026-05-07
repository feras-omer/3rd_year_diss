# Diffusion Checkpoint Comparison

- Dataset: `/home/feras/sim_ws/diffusion_f110/data/expert_merged.npz`
- Split mode: `full`
- Evaluation fraction: `0.1`
- Samples evaluated: `1024`

## Overall

| Model | Steering MAE | Steering RMSE | Speed MAE | Speed RMSE | Combined RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| old | 0.020486 | 0.030565 | 0.055729 | 0.148713 | 0.151822 |
| new | 0.030083 | 0.042019 | 0.019308 | 0.024199 | 0.048489 |

## Improvement

- Steering MAE improvement from old to new: `-0.009597`
- Speed MAE improvement from old to new: `0.036421`
- Combined RMSE improvement from old to new: `0.103333`

Positive values above mean the new checkpoint is better on that metric.

## Subsets

| Model | Subset | Samples | Steering MAE | Speed MAE | Combined RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| old | all | 1024 | 0.020486 | 0.055729 | 0.151822 |
| old | hairpin | 20 | 0.045427 | 0.047132 | 0.073657 |
| old | non_hairpin | 1004 | 0.019989 | 0.055900 | 0.152974 |
| old | left_turn | 34 | 0.039481 | 0.049750 | 0.071770 |
| old | right_turn | 8 | 0.031631 | 0.042964 | 0.055631 |
| old | straight | 873 | 0.017539 | 0.053197 | 0.121713 |
| old | high_steer | 12 | 0.031557 | 0.041431 | 0.058658 |
| new | all | 1024 | 0.030083 | 0.019308 | 0.048489 |
| new | hairpin | 20 | 0.090839 | 0.022441 | 0.110861 |
| new | non_hairpin | 1004 | 0.028872 | 0.019245 | 0.046402 |
| new | left_turn | 34 | 0.068647 | 0.018162 | 0.091704 |
| new | right_turn | 8 | 0.048889 | 0.028501 | 0.066039 |
| new | straight | 873 | 0.028392 | 0.019461 | 0.046285 |
| new | high_steer | 12 | 0.105762 | 0.024425 | 0.121628 |
