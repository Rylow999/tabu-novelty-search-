/* ============================================================================
 *  Tabu-Novelty Search — SuperPC / CUDA
 *  Compilación: nvcc -O3 -arch=sm_70 -o tns_gpu tns_gpu.cu -lm
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#ifndef VECTOR_DIM
  #define VECTOR_DIM 8
#endif
#ifndef POPULATION_SIZE
  #define POPULATION_SIZE 30
#endif
#ifndef MEMORY_SIZE
  #define MEMORY_SIZE 16
#endif
#ifndef STAGNATION_LIMIT
  #define STAGNATION_LIMIT 50
#endif
#ifndef MAX_ERAS
  #define MAX_ERAS 2000
#endif
#ifndef ANOMALY_THRESHOLD
  #define ANOMALY_THRESHOLD 1.95
#endif
#ifndef EPSILON
  #define EPSILON 1e-6
#endif
#ifndef TRIALS_PER_BLOCK
  #define TRIALS_PER_BLOCK 256
#endif

/* -------------------------------------------------------------------------- */
/*  Estructuras (layout natural)                                              */
/* -------------------------------------------------------------------------- */
typedef struct {
    uint64_t signature;
    double   state[VECTOR_DIM];
    double   valence;
} Candidate;

typedef struct {
    uint64_t signature;
    double   valence;
} TabuRecord;

/* -------------------------------------------------------------------------- */
/*  Memoria Constante GPU                                                     */
/* -------------------------------------------------------------------------- */
__constant__ uint64_t d_tabu_const[MEMORY_SIZE];
__constant__ int      d_tabu_count_const;

/* -------------------------------------------------------------------------- */
/*  Helpers Device                                                            */
/* -------------------------------------------------------------------------- */
__device__ __forceinline__ uint64_t discretize_word_device(const double *state)
{
    uint64_t sig = 0ULL;
    #pragma unroll
    for (int j = 0; j < VECTOR_DIM; j++) {
        unsigned int bits = (unsigned int)(state[j] * 40.7436654315252);
        if (bits > 255) bits = 255;
        sig |= ((uint64_t)bits) << (j * 8);
    }
    return sig;
}

__device__ __forceinline__ int hamming_device(uint64_t a, uint64_t b)
{
    return __popcll((unsigned long long)(a ^ b));
}

__device__ __forceinline__ double evaluate_fitness_device(const double *seed)
{
    double max_v = 0.0;
    #pragma unroll
    for (int j = 0; j < VECTOR_DIM; j++) {
        unsigned long long n = (unsigned long long)(seed[j] * 18000000ULL) + 3ULL;
        unsigned long long n0 = n, peak = n;
        unsigned int steps = 0;
        while (n > 1) {
            if (n & 1) { n = (3*n+1) >> 1; steps += 2; }
            else       { n >>= 1; steps++; }
            if (n > peak) peak = n;
            if (steps > 2000) break;
        }
        double v = (double)steps * 0.001 * (log((double)peak) / log((double)n0));
        if (v > max_v) max_v = v;
    }
    return max_v;
}

/* RNG device */
__device__ __forceinline__ unsigned int d_lcg(unsigned int *s)
{
    *s = (*s * 1103515245u + 12345u) & 0x7fffffffu;
    return *s;
}
__device__ __forceinline__ double d_rand(unsigned int *s)
{
    return (double)d_lcg(s) / (double)0x7fffffff;
}

/* -------------------------------------------------------------------------- */
/*  Kernel 1: Evaluación masiva                                               */
/* -------------------------------------------------------------------------- */
__global__ void eval_kernel(const double *d_states, double *d_valences)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        d_valences[idx] = evaluate_fitness_device(&d_states[idx * VECTOR_DIM]);
    }
}

/* -------------------------------------------------------------------------- */
/*  Kernel 2: Phase Hijack — Two-pass shared memory reduction (race-free)     */
/*  Cada bloque = 1 individuo. Cada hilo = 1 trial.                         */
/* -------------------------------------------------------------------------- */
__global__ void hijack_kernel(
    const double *d_champion,
    double       *d_new_states,
    uint64_t     *d_new_sigs)
{
    int tid = threadIdx.x;
    int individual = blockIdx.x;
    int tabu_count = d_tabu_count_const;

    /* Shared arrays: cada hilo escribe su candidato */
    __shared__ int    s_dmin[TRIALS_PER_BLOCK];
    __shared__ uint64_t s_sig[TRIALS_PER_BLOCK];
    __shared__ double s_state[TRIALS_PER_BLOCK * VECTOR_DIM];

    unsigned int rng = (unsigned int)(individual * 7919u + tid * 104729u + clock64());

    /* 1. Generar candidato */
    double cand[VECTOR_DIM];
    #pragma unroll
    for (int j = 0; j < VECTOR_DIM; j++) {
        double noise = d_rand(&rng) * 2.0 - 1.0;
        cand[j] = fmod(fabs(d_champion[j] + noise), 2.0 * M_PI);
    }

    /* 2. Cuantizar */
    uint64_t sig = discretize_word_device(cand);

    /* 3. Min Hamming vs memoria constante */
    int dmin = 64;
    for (int m = 0; m < tabu_count; m++) {
        int d = hamming_device(sig, d_tabu_const[m]);
        if (d < dmin) dmin = d;
    }

    /* 4. Escribir en shared memory */
    s_dmin[tid] = dmin;
    s_sig[tid] = sig;
    for (int j = 0; j < VECTOR_DIM; j++) {
        s_state[tid * VECTOR_DIM + j] = cand[j];
    }
    __syncthreads();

    /* 5. Reducción secuencial por hilo 0 (race-free) */
    if (tid == 0) {
        int best_idx = 0;
        int best_dmin = s_dmin[0];
        for (int i = 1; i < blockDim.x; i++) {
            if (s_dmin[i] > best_dmin) {
                best_dmin = s_dmin[i];
                best_idx = i;
            }
        }
        for (int j = 0; j < VECTOR_DIM; j++) {
            d_new_states[individual * VECTOR_DIM + j] = s_state[best_idx * VECTOR_DIM + j];
        }
        d_new_sigs[individual] = s_sig[best_idx];
    }
}

/* -------------------------------------------------------------------------- */
/*  Host: xoshiro256+ (unificado con CPU/ARM)                                  */
/* -------------------------------------------------------------------------- */
static uint64_t h_rng_s[4];

static inline uint64_t h_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
static inline uint64_t h_xoshiro256plus(void)
{
    uint64_t result = h_rng_s[0] + h_rng_s[3];
    uint64_t t = h_rng_s[1] << 17;
    h_rng_s[2] ^= h_rng_s[0]; h_rng_s[3] ^= h_rng_s[1];
    h_rng_s[1] ^= h_rng_s[2]; h_rng_s[0] ^= h_rng_s[3];
    h_rng_s[2] ^= t;
    h_rng_s[3] = h_rotl(h_rng_s[3], 45);
    return result;
}
static inline double h_randf(void)  { return (h_xoshiro256plus() >> 11) * (1.0 / (1ULL << 53)); }
static inline double h_randf_sym(void) { return h_randf() * 2.0 - 1.0; }

/* -------------------------------------------------------------------------- */
/*  Host Helpers                                                              */
/* -------------------------------------------------------------------------- */
static Candidate  h_population[POPULATION_SIZE];
static TabuRecord h_tabu_memory[MEMORY_SIZE];
static int        h_memory_count = 0;

static inline __attribute__((always_inline)) void evolve_classic_host(void)
{
    for (int i = 5; i < POPULATION_SIZE; i++) {
        int p1 = (int)(h_randf() * 5.0);
        int p2 = (int)(h_randf() * 5.0);
        for (int j = 0; j < VECTOR_DIM; j++) {
            h_population[i].state[j] = (h_randf() < 0.5)
                ? h_population[p1].state[j] : h_population[p2].state[j];
            if (h_randf() < 0.1) {
                h_population[i].state[j] += h_randf_sym() * 0.125;
                h_population[i].state[j] = fmod(fabs(h_population[i].state[j]), 2.0 * M_PI);
            }
        }
        h_population[i].signature = 0;
    }
}

static inline __attribute__((always_inline)) void sort_population_host(void)
{
    for (int i = 1; i < POPULATION_SIZE; i++) {
        Candidate key = h_population[i];
        int j = i - 1;
        while (j >= 0 && h_population[j].valence < key.valence) {
            h_population[j + 1] = h_population[j];
            j--;
        }
        h_population[j + 1] = key;
    }
}

static inline uint64_t discretize_host(const double *state)
{
    uint64_t sig = 0ULL;
    for (int j = 0; j < VECTOR_DIM; j++) {
        unsigned int bits = (unsigned int)(state[j] * 40.7436654315252);
        if (bits > 255) bits = 255;
        sig |= ((uint64_t)bits) << (j * 8);
    }
    return sig;
}

/* -------------------------------------------------------------------------- */
/*  MAIN                                                                      */
/* -------------------------------------------------------------------------- */
int main(void)
{
    /* Seed xoshiro256+ host */
    uint64_t z = (uint64_t)time(NULL) + 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 4; i++) {
        z += 0x9e3779b97f4a7c15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        h_rng_s[i] = x ^ (x >> 31);
    }

    printf("=== Tabu-Novelty Search | SuperPC / CUDA ===\n");
    printf("ARCH    : CUDA SM_70+ | Constant Mem | Race-free Shared Reduction\n");
    printf("VECTOR  : %d dims | POP: %d | MEM: %d | STAG: %d | ERAS: %d\n",
           VECTOR_DIM, POPULATION_SIZE, MEMORY_SIZE, STAGNATION_LIMIT, MAX_ERAS);
    printf("METRICA : Hamming 64-bit (__popcll) en Constant Cache\n");
    printf("BLOCK   : %d trials/individuo\n", TRIALS_PER_BLOCK);
    printf("-----------------------------------------------------------\n\n");

    for (int i = 0; i < POPULATION_SIZE; i++) {
        for (int j = 0; j < VECTOR_DIM; j++)
            h_population[i].state[j] = h_randf() * 2.0 * M_PI;
        h_population[i].signature = discretize_host(h_population[i].state);
        h_population[i].valence = 0.0;
    }

    /* Streams y pinned memory */
    cudaStream_t stream_eval, stream_hijack;
    cudaStreamCreate(&stream_eval);
    cudaStreamCreate(&stream_hijack);

    double *h_states_pinned, *h_valences_pinned;
    cudaMallocHost(&h_states_pinned,   POPULATION_SIZE * VECTOR_DIM * sizeof(double));
    cudaMallocHost(&h_valences_pinned, POPULATION_SIZE * sizeof(double));

    double   *d_states, *d_valences, *d_champion;
    uint64_t *d_sigs;
    cudaMalloc(&d_states,   POPULATION_SIZE * VECTOR_DIM * sizeof(double));
    cudaMalloc(&d_valences,  POPULATION_SIZE * sizeof(double));
    cudaMalloc(&d_champion,   VECTOR_DIM * sizeof(double));
    cudaMalloc(&d_sigs,      POPULATION_SIZE * sizeof(uint64_t));

    double best_ever = 0.0, v_best = 0.0;
    int stagnation = 0, hijack_count = 0;

    for (unsigned long long era = 1; era <= MAX_ERAS; era++) {

        for (int i = 0; i < POPULATION_SIZE; i++)
            memcpy(&h_states_pinned[i * VECTOR_DIM], h_population[i].state, VECTOR_DIM * sizeof(double));
        cudaMemcpyAsync(d_states, h_states_pinned,
                        POPULATION_SIZE * VECTOR_DIM * sizeof(double),
                        cudaMemcpyHostToDevice, stream_eval);

        eval_kernel<<<(POPULATION_SIZE + 255) / 256, 256, 0, stream_eval>>>(d_states, d_valences);

        cudaMemcpyAsync(h_valences_pinned, d_valences,
                        POPULATION_SIZE * sizeof(double),
                        cudaMemcpyDeviceToHost, stream_eval);
        cudaStreamSynchronize(stream_eval);

        for (int i = 0; i < POPULATION_SIZE; i++) h_population[i].valence = h_valences_pinned[i];
        sort_population_host();

        double champion = h_population[0].valence;
        if (champion > best_ever) best_ever = champion;

        if (fabs(champion - v_best) < EPSILON) stagnation++;
        else { v_best = champion; stagnation = 0; }

        if (stagnation >= STAGNATION_LIMIT) {
            hijack_count++;

            int slot = h_memory_count % MEMORY_SIZE;
            h_tabu_memory[slot].signature = discretize_host(h_population[0].state);
            h_tabu_memory[slot].valence = champion;
            h_memory_count++;
            int limit = (h_memory_count < MEMORY_SIZE) ? h_memory_count : MEMORY_SIZE;

            uint64_t h_tabu_sigs[MEMORY_SIZE];
            for (int m = 0; m < limit; m++) h_tabu_sigs[m] = h_tabu_memory[m].signature;
            cudaMemcpyToSymbol(d_tabu_const, h_tabu_sigs, MEMORY_SIZE * sizeof(uint64_t));
            cudaMemcpyToSymbol(d_tabu_count_const, &limit, sizeof(int));

            cudaMemcpy(d_champion, h_population[0].state, VECTOR_DIM * sizeof(double), cudaMemcpyHostToDevice);

            hijack_kernel<<<POPULATION_SIZE, TRIALS_PER_BLOCK, 0, stream_hijack>>>(
                d_champion, d_states, d_sigs);
            cudaStreamSynchronize(stream_hijack);

            cudaMemcpyAsync(h_states_pinned, d_states,
                            POPULATION_SIZE * VECTOR_DIM * sizeof(double),
                            cudaMemcpyDeviceToHost, stream_hijack);
            cudaStreamSynchronize(stream_hijack);
            for (int i = 0; i < POPULATION_SIZE; i++) {
                memcpy(h_population[i].state, &h_states_pinned[i * VECTOR_DIM], VECTOR_DIM * sizeof(double));
                h_population[i].signature = discretize_host(h_population[i].state);
                h_population[i].valence = 0.0;
            }

            printf("[Era %04llu] HIJACK #%d | blocked=%.5f | mem=%d/%d (GPU)\n",
                   era, hijack_count, champion, limit, MEMORY_SIZE);
            stagnation = 0;
            continue;
        }

        if (era % 200 == 0 || era == 1)
            printf("[Era %04llu] champion=%.5f | stagnation=%d/%d | hijacks=%d\n",
                   era, champion, stagnation, STAGNATION_LIMIT, hijack_count);

        if (champion >= ANOMALY_THRESHOLD) {
            printf("\n[!!!] ANOMALY at era %llu | valence=%.5f\n", era, champion);
            printf("[>] State: ");
            for (int j = 0; j < VECTOR_DIM; j++) printf("%.4f ", h_population[0].state[j]);
            printf("\n[>] Sig  : 0x%016llX\n", (unsigned long long)h_population[0].signature);
            break;
        }

        evolve_classic_host();
    }

    cudaFreeHost(h_states_pinned);
    cudaFreeHost(h_valences_pinned);
    cudaFree(d_states); cudaFree(d_valences); cudaFree(d_champion); cudaFree(d_sigs);
    cudaStreamDestroy(stream_eval); cudaStreamDestroy(stream_hijack);

    printf("\n=== Summary ===\n");
    printf("Hijacks executed : %d\n", hijack_count);
    printf("Best valence     : %.5f\n", best_ever);
    printf("Memory used      : %d/%d slots\n",
           (h_memory_count < MEMORY_SIZE ? h_memory_count : MEMORY_SIZE), MEMORY_SIZE);
    return 0;
}
