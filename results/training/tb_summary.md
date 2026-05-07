# Diffusion TensorBoard Comparison

- Old log: `/home/feras/sim_ws/old stuff/events.out.tfevents.1767726472.feras-Dell-G15-5511.249812.0`
- New log: `/home/feras/sim_ws/diffusion_f110/tb_logs/diffusion/events.out.tfevents.1775770825.feras-Dell-G15-5511.163756.0`

## Available Scalars

- Old: `loss/train`
- New: `loss/train_step, loss/train_epoch, loss/val_epoch, lr`

## Step-Level Training Loss

| Run | Points | First Step | Last Step | First Loss | Last Loss | Minimum Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old `loss/train` | 9000 | 0 | 8999 | 3.035051 | 0.074333 | 0.029409 |
| New `loss/train_step` | 8880 | 0 | 8879 | 6.912366 | 5.034998 | 2.447289 |

## New Trainer Epoch Metrics

| Scalar | Points | First Value | Last Value | Minimum Value |
| --- | ---: | ---: | ---: | ---: |
| `loss/train_epoch` | 120 | 5.267924 | 4.319051 | 4.288140 |
| `loss/val_epoch` | 120 | 2.731122 | 2.294306 | 2.083176 |

## Method Note

- The old and new trainers do not log identical scalars, so TensorBoard comparison is used as supporting evidence about optimisation behaviour, not as the sole measure of model quality.
- The more defensible primary comparisons remain the offline checkpoint evaluation and the runtime benchmark.
