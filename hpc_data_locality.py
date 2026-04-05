"""
==============================================================================
Data Locality Optimization Through Cache-Friendly Data Structure Selection
==============================================================================
Based on: Azad et al. (2023) "An Empirical Study of HPC Performance Bugs"

Demonstrates how data structure layout affects performance in HPC workloads.
Three approaches compared for pairwise force computation (molecular dynamics):

1. UNOPTIMIZED: Python objects (scattered heap memory, pointer chasing)
2. OPTIMIZED (SoA): NumPy separate arrays with Python loops (contiguous)
3. OPTIMIZED (Vectorized): Fully vectorized NumPy (contiguous + SIMD)

Author: Aashish Sapkota | MSCS - University of the Cumberlands | April 2026
==============================================================================
"""

import time
import random
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class ParticleObject:
    """Heap-allocated Python object. Each access involves pointer dereferencing."""
    def __init__(self, pid, x, y, z):
        self.id = pid
        self.x = x; self.y = y; self.z = z
        self.fx = 0.0; self.fy = 0.0; self.fz = 0.0


def compute_forces_objects(particles, cutoff):
    """
    Force computation using Python objects stored in a list.
    INEFFICIENT pattern from the paper:
    - Each particle is a heap-allocated object at a random memory address
    - Accessing p.x, p.y, p.z involves pointer chasing through object dict
    - Similar to: TileDB-d51b082 (forward_list), CGAL-8855eb5 (std::list)
    """
    cutoff_sq = cutoff * cutoff
    n = len(particles)
    interactions = 0
    for i in range(n):
        particles[i].fx = 0.0; particles[i].fy = 0.0; particles[i].fz = 0.0
    for i in range(n):
        pi = particles[i]
        for j in range(i + 1, n):
            pj = particles[j]
            dx = pi.x - pj.x
            dy = pi.y - pj.y
            dz = pi.z - pj.z
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq < cutoff_sq and dist_sq > 0:
                inv_dist = 1.0 / (dist_sq ** 0.5)
                force = inv_dist * inv_dist * inv_dist
                fx = force * dx; fy = force * dy; fz = force * dz
                pi.fx += fx; pi.fy += fy; pi.fz += fz
                pj.fx -= fx; pj.fy -= fy; pj.fz -= fz
                interactions += 1
    return interactions


def compute_forces_numpy_loops(px, py, pz, cutoff):
    """
    Same algorithm, data in contiguous NumPy arrays (SoA layout).
    OPTIMIZED data layout from the paper:
    - Contiguous float64 arrays enable CPU prefetching
    - Corresponds to: CGAL-8855eb5 (lists -> vectors)
    """
    cutoff_sq = cutoff * cutoff
    n = len(px)
    fx = np.zeros(n); fy = np.zeros(n); fz = np.zeros(n)
    interactions = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = px[i] - px[j]
            dy = py[i] - py[j]
            dz = pz[i] - pz[j]
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq < cutoff_sq and dist_sq > 0:
                inv_dist = 1.0 / (dist_sq ** 0.5)
                force = inv_dist * inv_dist * inv_dist
                ffx = force * dx; ffy = force * dy; ffz = force * dz
                fx[i] += ffx; fy[i] += ffy; fz[i] += ffz
                fx[j] -= ffx; fy[j] -= ffy; fz[j] -= ffz
                interactions += 1
    return interactions


def compute_forces_vectorized(px, py, pz, cutoff):
    """
    Fully vectorized NumPy (contiguous + SIMD).
    MAXIMUM OPTIMIZATION combining:
    (a) Cache-friendly data layout (data locality optimization)
    (b) SIMD parallelism (vector parallelism)
    (c) Elimination of Python interpreter overhead
    """
    cutoff_sq = cutoff * cutoff
    dx = px[:, np.newaxis] - px[np.newaxis, :]
    dy = py[:, np.newaxis] - py[np.newaxis, :]
    dz = pz[:, np.newaxis] - pz[np.newaxis, :]
    dist_sq = dx*dx + dy*dy + dz*dz
    mask = (dist_sq < cutoff_sq) & (dist_sq > 0)
    interactions = int(np.sum(np.triu(mask, k=1)))
    safe = np.where(mask, dist_sq, 1.0)
    inv_dist = np.where(mask, 1.0 / np.sqrt(safe), 0.0)
    force = inv_dist ** 3
    fx = np.sum(force * dx, axis=1)
    fy = np.sum(force * dy, axis=1)
    fz = np.sum(force * dz, axis=1)
    return interactions


def benchmark(func, *args, trials=3):
    times = []
    result = None
    for _ in range(trials):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return np.median(times), result


def run_benchmarks():
    print("=" * 75)
    print("DATA LOCALITY OPTIMIZATION: Cache-Friendly Data Structure Selection")
    print("Workload: Pairwise Force Computation (Molecular Dynamics)")
    print("=" * 75)

    BOX_SIZE = 20.0
    CUTOFF = 5.0
    SIZES = [100, 300, 500, 800, 1000]
    results = {'sizes': SIZES, 'objects': [], 'numpy_loop': [], 'vectorized': []}

    for n in SIZES:
        print(f"\n--- {n} Particles (cutoff={CUTOFF}) ---")
        random.seed(42)
        particles = [ParticleObject(i, random.uniform(0, BOX_SIZE),
                     random.uniform(0, BOX_SIZE), random.uniform(0, BOX_SIZE))
                     for i in range(n)]
        px = np.array([p.x for p in particles])
        py = np.array([p.y for p in particles])
        pz = np.array([p.z for p in particles])

        t1, int1 = benchmark(compute_forces_objects, particles, CUTOFF, trials=2)
        results['objects'].append(t1 * 1000)
        print(f"  Python Objects (heap, scattered):   {t1*1000:>10.2f} ms  [{int1} interactions]")

        t2, int2 = benchmark(compute_forces_numpy_loops, px, py, pz, CUTOFF, trials=2)
        results['numpy_loop'].append(t2 * 1000)
        sp2 = t1 / t2 if t2 > 0 else 0
        print(f"  NumPy Arrays + Python Loop (SoA):   {t2*1000:>10.2f} ms  (Speedup: {sp2:.2f}x)")

        t3, int3 = benchmark(compute_forces_vectorized, px, py, pz, CUTOFF, trials=2)
        results['vectorized'].append(t3 * 1000)
        sp3 = t1 / t3 if t3 > 0 else 0
        print(f"  Vectorized NumPy (SoA + SIMD):      {t3*1000:>10.2f} ms  (Speedup: {sp3:.2f}x)")

    return results


def generate_charts(results):
    sizes = results['sizes']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Data Locality Optimization: Pairwise Force Computation',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(sizes, results['objects'], 'o-', color='#F44336',
            label='Python Objects (Scattered)', linewidth=2, markersize=7)
    ax.plot(sizes, results['numpy_loop'], 's-', color='#FF9800',
            label='NumPy + Loops (Contiguous)', linewidth=2, markersize=7)
    ax.plot(sizes, results['vectorized'], '^-', color='#4CAF50',
            label='Vectorized (Contiguous + SIMD)', linewidth=2, markersize=7)
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Execution Time Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sp_loop = [o/l if l > 0 else 0 for o, l in zip(results['objects'], results['numpy_loop'])]
    sp_vec = [o/v if v > 0 else 0 for o, v in zip(results['objects'], results['vectorized'])]
    ax.plot(sizes, sp_loop, 's-', color='#FF9800', label='Contiguous Layout', linewidth=2, markersize=7)
    ax.plot(sizes, sp_vec, '^-', color='#4CAF50', label='Contiguous + Vectorized', linewidth=2, markersize=7)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Speedup over Python Objects')
    ax.set_title('Speedup from Data Locality Optimization')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    approaches = ['Python Objects\n(Pointer-Heavy)', 'NumPy Arrays\n(Contiguous)', 'Vectorized\n(Contiguous+SIMD)']
    largest = [results['objects'][-1], results['numpy_loop'][-1], results['vectorized'][-1]]
    normalized = [t / largest[0] * 100 for t in largest]
    colors = ['#F44336', '#FF9800', '#4CAF50']
    bars = ax.bar(approaches, normalized, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Relative Execution Time (%)')
    ax.set_title(f'Normalized Time at N={sizes[-1]}')
    for bar, val in zip(bars, normalized):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('hpc_optimization_results.png', dpi=150, bbox_inches='tight')
    print(f"\n[Chart saved: hpc_optimization_results.png]")

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    ax2.plot(sizes, results['objects'], 'o-', color='#F44336', label='Python Objects', linewidth=2)
    ax2.plot(sizes, results['numpy_loop'], 's-', color='#FF9800', label='NumPy + Loops', linewidth=2)
    ax2.plot(sizes, results['vectorized'], '^-', color='#4CAF50', label='Vectorized', linewidth=2)
    base = results['objects'][0]; n0 = sizes[0]
    ref_n2 = [base * (s/n0)**2 for s in sizes]
    ax2.plot(sizes, ref_n2, ':', color='gray', alpha=0.5, label='O(n^2) reference')
    ax2.set_xlabel('Number of Particles')
    ax2.set_ylabel('Execution Time (ms)')
    ax2.set_title('Scaling Behavior: Impact of Data Structure on Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hpc_scaling_analysis.png', dpi=150, bbox_inches='tight')
    print(f"[Chart saved: hpc_scaling_analysis.png]")


if __name__ == '__main__':
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}\n")
    results = run_benchmarks()
    print("\n" + "=" * 75)
    print("SUMMARY TABLE")
    print("=" * 75)
    print(f"{'N':>6} {'Objects(ms)':>12} {'NpLoop(ms)':>12} {'Vector(ms)':>12} {'Loop Spdup':>11} {'Vec Spdup':>11}")
    print("-" * 75)
    for i, n in enumerate(results['sizes']):
        o = results['objects'][i]; l = results['numpy_loop'][i]; v = results['vectorized'][i]
        print(f"{n:>6} {o:>12.2f} {l:>12.2f} {v:>12.2f} {o/l:>10.2f}x {o/v:>10.2f}x")
    print("=" * 75)
    generate_charts(results)
    print("\n[All benchmarks complete]")