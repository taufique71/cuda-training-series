# 1. Comparing Reductions
+-----------+--------------+---------------+----------------+
| `N`       | `atomic_red` | `reduction_a` | `reduction_ws` |
+-----------+--------------+---------------+----------------+
| 8M        | 20.09 ms     | 50.72 us      | 50.75 us       |
+-----------+--------------+---------------+----------------+
| 256 * 640 | 395.62 us    | 5.98 us       | 5.41 us        |
+-----------+--------------+---------------+----------------+
| 32M       | 80.37 ms     | 171.39 us     | 173.76 us      |
+-----------+--------------+---------------+----------------+
### 1a. With N = 8M
Need to think about the reason behind numbers

### 1b. Changing N from ~8M to 163840 (=640*256)
Need to think about the reason behind numbers

### 1c. Changing N from ~8M to ~32M
Need to think about the reason behind numbers

