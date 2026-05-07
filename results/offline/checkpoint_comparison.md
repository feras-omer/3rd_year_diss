# Diffusion Checkpoint Comparison

- Dataset: `/home/feras/sim_ws/diffusion_f110/data/expert_merged.npz`
- Split mode: `full`
- Evaluation fraction: `0.1`
- Samples evaluated: `20891`

## Overall

| Model | Steering MAE | Steering RMSE | Speed MAE | Speed RMSE | Combined RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| old | 0.033714 | 0.048417 | 0.077185 | 0.112321 | 0.122312 |
| new | 0.034414 | 0.049482 | 0.037546 | 0.323138 | 0.326904 |

## Improvement

- Steering MAE improvement from old to new: `-0.000700`
- Speed MAE improvement from old to new: `0.039639`
- Combined RMSE improvement from old to new: `-0.204592`

Positive values above mean the new checkpoint is better on that metric.

## Subsets

| Model | Subset | Samples | Steering MAE | Speed MAE | Combined RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| old | all | 20891 | 0.033714 | 0.077185 | 0.122312 |
| old | hairpin | 861 | 0.055537 | 0.073741 | 0.198486 |
| old | non_hairpin | 20030 | 0.032776 | 0.077333 | 0.117940 |
| old | left_turn | 2303 | 0.043625 | 0.065429 | 0.114499 |
| old | right_turn | 1880 | 0.034690 | 0.058940 | 0.138845 |
| old | straight | 13236 | 0.032326 | 0.083511 | 0.115594 |
| old | high_steer | 602 | 0.054644 | 0.070108 | 0.224546 |
| new | all | 20891 | 0.034414 | 0.037546 | 0.326904 |
| new | hairpin | 861 | 0.097042 | 0.016327 | 0.119163 |
| new | non_hairpin | 20030 | 0.031722 | 0.038458 | 0.332941 |
| new | left_turn | 2303 | 0.058582 | 0.015950 | 0.076079 |
| new | right_turn | 1880 | 0.072858 | 0.018015 | 0.093333 |
| new | straight | 13236 | 0.025341 | 0.049370 | 0.407244 |
| new | high_steer | 602 | 0.121386 | 0.015777 | 0.138171 |
