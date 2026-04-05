# HPC Data Locality Optimization

**Data Locality Optimization Through Cache-Friendly Data Structure Selection in High-Performance Computing Applications**

## Overview

This project demonstrates how replacing cache-unfriendly data structures (linked lists with pointer chasing) with cache-friendly contiguous structures (arrays, Structure of Arrays) can dramatically improve performance in HPC workloads.

Based on the empirical study: *"An Empirical Study of High Performance Computing (HPC) Performance Bugs"* by Azad et al. (2023), which found that data locality optimization was the most frequently applied fix category (21% of all performance fixes) across 23 open-source HPC projects.

## Benchmark: Pairwise Force Computation

The benchmark simulates a core molecular dynamics operation — pairwise Lennard-Jones force computation — comparing three data structure layouts:

| Approach | Description | Cache Behavior |
|----------|-------------|----------------|
| **Linked List** | Individually heap-allocated nodes, pointer chasing | Cache-unfriendly (scattered memory) |
| **Array of Structures (AoS)** | Contiguous array of structs | Cache-friendly (sequential access) |
| **Structure of Arrays (SoA)** | Separate contiguous arrays per field | Optimal (maximum useful data per cache line) |

## Results (Apple M5 Pro)

| N | Linked (ms) | AoS (ms) | SoA (ms) | AoS Speedup | SoA Speedup |
|---|---|---|---|---|---|
| 500 | 0.24 | 0.17 | 0.15 | 1.44x | 1.65x |
| 1,000 | 1.06 | 0.58 | 0.54 | 1.82x | 1.96x |
| 2,000 | 6.86 | 1.40 | 1.29 | 4.90x | 5.30x |
| 5,000 | 42.18 | 8.06 | 6.85 | 5.23x | 6.16x |
| 10,000 | 201.69 | 33.14 | 27.62 | 6.09x | **7.30x** |

## Files

- `hpc_benchmark.c` — Primary C benchmark (linked list vs AoS vs SoA)
- `hpc_data_locality.py` — Supplementary Python benchmark demonstrating that interpreter overhead masks cache effects

## How to Run

### C Benchmark
```bash
gcc -O2 -o hpc_benchmark hpc_benchmark.c -lm
./hpc_benchmark
```

### Python Benchmark
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
python3 hpc_data_locality.py
```

## Key Findings

1. **SoA achieves up to 7.30x speedup** over linked lists from data layout alone — no algorithmic changes
2. **Speedup increases with scale** — at small N data fits in cache regardless of layout; at large N, scattered access causes severe cache thrashing
3. **Language matters** — Python interpreter overhead completely masks cache effects, making this optimization only impactful in compiled languages (C/C++/Fortran)

## Author

Aashish Sapkota — University of the Cumberlands, MSCS Program

## Reference

Azad, M. A. K., Iqbal, N., Hassan, F., & Roy, P. (2023). An empirical study of high performance computing (HPC) performance bugs. *Proceedings of the IEEE/ACM 20th International Conference on Mining Software Repositories (MSR)*, 1–13.
