# Tabu-Novelty Search with Binary Hamming Signatures

A multi-platform C/CUDA reference implementation for discovering extreme states in chaotic fitness landscapes.

**Author:** Luciano Nieto (Argentina)  
**License:** The Unlicense (public domain)  
**Hardware tested on:** ARM SBC (Raspberry Pi-class), x86-64 laptop

---

## The idea

Evolutionary algorithms exploring extreme states (Collatz anomalies, coupled logistic maps) stall in local attractors. This project implements a **Tabu-Memory-assisted Phase Hijack** that archives stagnant attractors and forces the population to diversify toward the **farthest point from all known failures**, measured in Hamming space over binary signatures.

**Key difference from standard Novelty Search:**
- Lehman & Stanley use **average** k-NN distance over a behavior archive.
- We use **max-min** distance over a **failure archive** (stagnant attractors).

This pushes the search into the "cracks" of the landscape where singularities emerge.

**The speed trick:** instead of Euclidean distance in R^d, we discretize each dimension to 8 bits, pack the vector into a single `uint64_t`, and measure novelty with **Hamming distance** (`popcount` in 1 CPU cycle). The Tabu scan becomes cache-bound, not compute-bound.

---

## Repository structure

```
.
├── Makefile
├── LICENSE
├── README.md
├── .gitignore
└── src/
    ├── tns_cpu_pure.c      # Portable C99 (any CPU)
    ├── tns_arm_neon.c      # ARMv8-A+NEON (Raspberry Pi, Apple Silicon, etc.)
    └── tns_gpu.cu          # CUDA SM_70+ (Volta/Turing/Ampere/Hopper)
```

All three targets share the **exact same** mathematical core: identical Collatz anomaly evaluator, identical Hamming metric, identical stagnation logic. Only the hardware exploitation changes.

---

## Quick start

```bash
# Clone
git clone https://github.com/TU_USUARIO/tabu-novelty-search.git
cd tabu-novelty-search

# Build available targets
make all

# Or individually
make cpu    # gcc
make arm    # gcc + ARM NEON
make gpu    # nvcc

# Run
./bin/tns_cpu
./bin/tns_arm
./bin/tns_gpu
```

### One-liner compilation (no Makefile)

```bash
# Any CPU
gcc -O3 -std=c99 -march=native -ffast-math -o tns_cpu src/tns_cpu_pure.c -lm

# ARM with NEON
gcc -O3 -std=c99 -march=armv8-a+fp+simd -o tns_arm src/tns_arm_neon.c -lm

# CUDA
nvcc -O3 -arch=sm_70 -o tns_gpu src/tns_gpu.cu -lm
```

---

## Target details

| Target | Key optimization | Best for |
|---|---|---|
| **CPU Pure** | `xoshiro256+` PRNG, `__builtin_prefetch`, compiler `hot`/`restrict` hints, insertion sort | Any CPU with a C compiler. Zero dependencies. |
| **ARM NEON** | Vectorized Hamming: `uint64x2_t` + `veorq_u64` processes **2 Tabu signatures per iteration** | ARM SBCs (Pi 4/5, Apple Silicon). 2× hijack throughput without extra DRAM bandwidth. |
| **GPU CUDA** | Tabu history in `__constant__` memory (broadcast cache), race-free shared reduction, async streams with pinned memory | Workstations / servers with NVIDIA GPUs. 256 trials per individual in parallel. |

---

## Parameters (compile-time)

All targets respect the same tunable macros via `-D`:

| Macro | Default | Description |
|---|---|---|
| `VECTOR_DIM` | 8 | State dimensions |
| `POPULATION_SIZE` | 30 | Individuals per era |
| `MEMORY_SIZE` | 16 | Tabu archive capacity |
| `STAGNATION_LIMIT` | 50 | Eras before triggering hijack |
| `MAX_ERAS` | 2000 | Total iterations |
| `ANOMALY_THRESHOLD` | 1.95 | Valence that triggers anomaly report |

Example:
```bash
gcc -O3 -DVECTOR_DIM=16 -DPOPULATION_SIZE=100 -o tns_cpu src/tns_cpu_pure.c -lm
```

---

## What this is / what this isn't

This is a **reference implementation** designed to be:
- **Correct:** race-free, memory-safe, deterministic given a seed.
- **Portable:** compiles on anything from a $35 ARM board to a DGX.
- **Hackable:** single-file targets, no external dependencies, obvious hot paths.

This is **not** a production framework. It does not include:
- Distributed MPI coordination
- Checkpointing / resume
- Automatic hyperparameter tuning
- Python bindings

If you want those, the code is public domain — build on it.

---

## Open questions

I can prove this works on limited hardware. I **cannot** prove:
- What happens with 512-bit signatures and 10k population size.
- Whether there are formal convergence proofs for Tabu-assisted max-min search in purely chaotic environments.
- How this behaves on multi-GPU or distributed ARM clusters.

If you have access to those resources or that theoretical background, I'd genuinely like to know what breaks and what scales.

---

## Contact

Luciano Nieto — independent researcher, Argentina.  
If you run it, extend it, or find something interesting, open an issue or drop a note.
