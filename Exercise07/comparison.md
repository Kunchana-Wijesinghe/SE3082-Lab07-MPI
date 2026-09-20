# Exercise 7: Summary Comparison

**Student Name:** Kunchana Wijesinghe
**Student ID:** IT24102587

## 1. Comparison of the Six MPI Programs

| Program | Communication Operations Used | Array Allocation | Manual Summation on Root | Final Result Availability |
|---|---|---|---|---|
| `sum_bcast.c` | `MPI_Bcast`, `MPI_Send`, `MPI_Recv` | Every process allocates the complete array | Yes. Root receives and adds each partial sum | Root only |
| `sum_scatter.c` | `MPI_Scatter`, `MPI_Send`, `MPI_Recv` | Root allocates the complete array; every process allocates only its chunk | Yes. Root receives and adds each partial sum | Root only |
| `sum_gather.c` | `MPI_Scatter`, `MPI_Gather` | Root allocates the complete array; every process allocates only its chunk | Yes. Root manually adds the gathered values | Root only |
| `sum_reduce.c` | `MPI_Scatter`, `MPI_Reduce` | Root allocates the complete array; every process allocates only its chunk | No. `MPI_Reduce` performs the summation | Root only |
| `sum_allreduce.c` | `MPI_Scatter`, `MPI_Allreduce` | Root allocates the complete array; every process allocates only its chunk | No. `MPI_Allreduce` performs the summation | All processes |
| `sum_scan.c` | `MPI_Scatter`, `MPI_Scan` | Root allocates the complete array; every process allocates only its chunk | No. `MPI_Scan` performs prefix summation | A different prefix result is available on each process; the last rank receives the global total |

## 2. Execution-Time Results

All programs produced the correct total:

`500,000,500,000`

| Program | 2 Processes (sec) | 4 Processes (sec) | 8 Processes (sec) |
|---|---:|---:|---:|
| Broadcast + Send/Recv | 0.006642 | 0.020796 | 0.025040 |
| Scatter + Send/Recv | 0.006834 | 0.008801 | 0.007703 |
| Scatter + Gather | 0.005472 | 0.014258 | 0.016902 |
| Scatter + Reduce | **0.003336** | 0.005293 | 0.006281 |
| Scatter + Allreduce | 0.008989 | **0.003125** | **0.006280** |
| Scatter + Scan | 0.027455 | 0.007079 | 0.012069 |

### Fastest Approaches

- With 2 processes, `MPI_Reduce` was the fastest at 0.003336 seconds.
- With 4 processes, `MPI_Allreduce` was the fastest at 0.003125 seconds.
- With 8 processes, `MPI_Allreduce` produced the lowest measured time at 0.006280 seconds. However, it was effectively tied with `MPI_Reduce`, which took 0.006281 seconds.

`MPI_Bcast` becomes less efficient because it sends the complete one-million-element array to every process, even though each process uses only one part of it. `MPI_Scatter` sends only the required chunk to each process, reducing memory use and communication.

`MPI_Reduce` and `MPI_Allreduce` use efficient collective algorithms instead of requiring root to receive and process each result using a manual loop. Therefore, these approaches generally provided the best performance in this experiment.

The exact execution times can change between runs because of system load, caching, process scheduling, and WSL oversubscription. In this test, 4-process and 8-process executions used `--oversubscribe`, so the results demonstrate the collective operations but do not represent the performance of a real multi-node cluster.

## 3. When to Choose MPI_Scan Instead of MPI_Allreduce

`MPI_Allreduce` should be used when every process needs the same global result. `MPI_Scan` should be used when each process needs a cumulative result containing the values from rank 0 up to its own rank.

A concrete example is calculating global offsets for parts of a distributed array. Each process can calculate the size or sum of its local section. `MPI_Scan` then gives each process a cumulative total. By subtracting its local value, a process obtains the total belonging to all earlier processes. This value can be used as the starting global offset for that process without requiring additional communication.

Other uses include parallel prefix sums, cumulative distributions, global index assignment, and load-balanced work partitioning.
