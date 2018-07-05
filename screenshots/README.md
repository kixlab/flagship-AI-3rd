# Screenshots

## 01 - First Training

### Conditions
| #. of ... | Value |
|:---|:-------------|
| Trainer version | v0.1 |
| learned sentences | 100,000 |
| window size | 1 |
| batch size | 20 |
| sample size (for loss) | 15 |
| embedding size | 50 |
| learning rate | 0.1 |
| training epoch | 60,000 |

### Special Note

- Too slow training (9.5 mins per 2,000 epoch)
- More iterations, more losses due to (total #. of words > #. of trained words)
- GPU acceleration & Multi-thread training is needed