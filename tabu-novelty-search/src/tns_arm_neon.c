/* ============================================================================
 *  Tabu-Novelty Search — ARM NEON (AArch64, SIMD-Vectorized)
 *  Compilación: gcc -O3 -std=c99 -march=armv8-a+fp+simd -o tns_arm tns_arm_neon.c -lm
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <arm_neon.h>

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
#ifndef TRIALS_PER_INDIVIDUAL
  #define TRIALS_PER_INDIVIDUAL 8
#endif

/* -------------------------------------------------------------------------- */
/*  xoshiro256+                                                               */
/* -------------------------------------------------------------------------- */
static uint64_t rng_s[4];
static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
static inline uint64_t xoshiro256plus(void)
{
    uint64_t result = rng_s[0] + rng_s[3];
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t;
    rng_s[3] = rotl(rng_s[3], 45);
    return result;
}
static inline double randf(void)  { return (xoshiro256plus() >> 11) * (1.0 / (1ULL << 53)); }
static inline double randf_sym(void) { return randf() * 2.0 - 1.0; }

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

static Candidate  population[POPULATION_SIZE] __attribute__((aligned(64)));
static TabuRecord tabu_memory[MEMORY_SIZE]    __attribute__((aligned(64)));
static int        memory_count = 0;

/* -------------------------------------------------------------------------- */
/*  Helpers inline                                                            */
/* -------------------------------------------------------------------------- */
static inline __attribute__((always_inline, hot, target("arch=armv8-a+fp+simd")))
uint64_t discretize_word(const double *restrict state)
{
    uint64_t sig = 0ULL;
    #pragma GCC unroll 8
    for (int j = 0; j < VECTOR_DIM; j++) {
        unsigned int bits = (unsigned int)(state[j] * 40.7436654315252);
        bits = (bits > 255) ? 255 : bits;
        sig |= ((uint64_t)bits) << (j * 8);
    }
    return sig;
}

static inline __attribute__((always_inline, hot))
int hamming_scalar(uint64_t a, uint64_t b)
{
    return __builtin_popcountll(a ^ b);
}

/* Vectorizado: 2 firmas de tabú por iteración */
static inline __attribute__((always_inline, hot, target("arch=armv8-a+fp+simd")))
int hamming_vec2(uint64_t sig, uint64_t t0, uint64_t t1)
{
    uint64x2_t vsig = vdupq_n_u64(sig);
    uint64_t   pair[2] = { t0, t1 };
    uint64x2_t vtabu = vld1q_u64(pair);
    uint64x2_t vxor  = veorq_u64(vsig, vtabu);
    uint64_t   xr[2];
    vst1q_u64(xr, vxor);
    int d0 = __builtin_popcountll(xr[0]);
    int d1 = __builtin_popcountll(xr[1]);
    return (d0 < d1) ? d0 : d1;
}

/* FLE: Collatz anomaly scorer */
static inline __attribute__((always_inline, hot))
double evaluate_fitness(const double *restrict seed)
{
    double max_v = 0.0;
    #pragma GCC unroll 8
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

/* -------------------------------------------------------------------------- */
/*  TSC: Phase Hijack (ARM NEON-optimized)                                    */
/* -------------------------------------------------------------------------- */
static __attribute__((hot, target("arch=armv8-a+fp+simd"))) void
 tsc_hijack(const double *restrict champion_state, double blocked_v)
{
    int slot = memory_count % MEMORY_SIZE;
    tabu_memory[slot].signature = discretize_word(champion_state);
    tabu_memory[slot].valence = blocked_v;
    memory_count++;

    int limit = (memory_count < MEMORY_SIZE) ? memory_count : MEMORY_SIZE;
    int limit_even = limit & ~1;

    for (int i = 0; i < POPULATION_SIZE; i++) {
        double best_state[VECTOR_DIM];
        uint64_t best_sig = 0;
        int best_dmin = -1;

        for (int trial = 0; trial < TRIALS_PER_INDIVIDUAL; trial++) {
            double cand[VECTOR_DIM];
            for (int j = 0; j < VECTOR_DIM; j++) {
                cand[j] = fmod(fabs(champion_state[j] + randf_sym()), 2.0 * M_PI);
            }
            uint64_t sig = discretize_word(cand);

            int dmin = 64;
            int m = 0;
            /* Vectorizado: 2 firmas por iteración */
            for (; m < limit_even; m += 2) {
                int d2 = hamming_vec2(sig,
                                       tabu_memory[m].signature,
                                       tabu_memory[m+1].signature);
                if (d2 < dmin) dmin = d2;
                __builtin_prefetch(&tabu_memory[m+2], 0, 3);
            }
            /* Scalar tail */
            for (; m < limit; m++) {
                int d = hamming_scalar(sig, tabu_memory[m].signature);
                if (d < dmin) dmin = d;
            }

            if (dmin > best_dmin || trial == 0) {
                best_dmin = dmin;
                best_sig = sig;
                memcpy(best_state, cand, sizeof(cand));
            }
        }
        memcpy(population[i].state, best_state, sizeof(best_state));
        population[i].signature = best_sig;
        population[i].valence = 0.0;
    }
}

static inline __attribute__((always_inline)) void evolve_classic(void)
{
    for (int i = 5; i < POPULATION_SIZE; i++) {
        int p1 = (int)(randf() * 5.0);
        int p2 = (int)(randf() * 5.0);
        for (int j = 0; j < VECTOR_DIM; j++) {
            population[i].state[j] = (randf() < 0.5)
                ? population[p1].state[j] : population[p2].state[j];
            if (randf() < 0.1) {
                population[i].state[j] += randf_sym() * 0.125;
                population[i].state[j] = fmod(fabs(population[i].state[j]), 2.0 * M_PI);
            }
        }
        population[i].signature = discretize_word(population[i].state);
    }
}

static inline __attribute__((always_inline)) void sort_population(void)
{
    for (int i = 1; i < POPULATION_SIZE; i++) {
        Candidate key = population[i];
        int j = i - 1;
        while (j >= 0 && population[j].valence < key.valence) {
            population[j + 1] = population[j];
            j--;
        }
        population[j + 1] = key;
    }
}

/* -------------------------------------------------------------------------- */
/*  MAIN                                                                      */
/* -------------------------------------------------------------------------- */
int main(void)
{
    uint64_t z = (uint64_t)time(NULL) + 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 4; i++) {
        z += 0x9e3779b97f4a7c15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = x ^ (x >> 31);
    }

    printf("=== Tabu-Novelty Search | ARM NEON ===\n");
    printf("ARCH    : ARMv8-A+NEON | SIMD-vectorized Hamming (2x/iter)\n");
    printf("VECTOR  : %d dims | POP: %d | MEM: %d | STAG: %d | ERAS: %d\n",
           VECTOR_DIM, POPULATION_SIZE, MEMORY_SIZE, STAGNATION_LIMIT, MAX_ERAS);
    printf("METRICA : Hamming 64-bit (NEON veorq + popcount)\n");
    printf("RNG     : xoshiro256+\n");
    printf("-----------------------------------------------------------\n\n");

    for (int i = 0; i < POPULATION_SIZE; i++) {
        for (int j = 0; j < VECTOR_DIM; j++)
            population[i].state[j] = randf() * 2.0 * M_PI;
        population[i].signature = discretize_word(population[i].state);
        population[i].valence = 0.0;
    }

    double best_ever = 0.0, v_best = 0.0;
    int stagnation = 0, hijack_count = 0;

    for (unsigned long long era = 1; era <= MAX_ERAS; era++) {
        for (int i = 0; i < POPULATION_SIZE; i++)
            population[i].valence = evaluate_fitness(population[i].state);
        sort_population();

        double champion = population[0].valence;
        if (champion > best_ever) best_ever = champion;

        if (fabs(champion - v_best) < EPSILON) stagnation++;
        else { v_best = champion; stagnation = 0; }

        if (stagnation >= STAGNATION_LIMIT) {
            hijack_count++;
            printf("[Era %04llu] HIJACK #%d | blocked=%.5f | mem=%d/%d\n",
                   era, hijack_count, champion,
                   (memory_count < MEMORY_SIZE ? memory_count : MEMORY_SIZE), MEMORY_SIZE);
            tsc_hijack(population[0].state, champion);
            stagnation = 0;
            continue;
        }

        if (era % 200 == 0 || era == 1)
            printf("[Era %04llu] champion=%.5f | stagnation=%d/%d | hijacks=%d\n",
                   era, champion, stagnation, STAGNATION_LIMIT, hijack_count);

        if (champion >= ANOMALY_THRESHOLD) {
            printf("\n[!!!] ANOMALY at era %llu | valence=%.5f\n", era, champion);
            printf("[>] State: ");
            for (int j = 0; j < VECTOR_DIM; j++) printf("%.4f ", population[0].state[j]);
            printf("\n[>] Sig  : 0x%016llX\n", (unsigned long long)population[0].signature);
            break;
        }
        evolve_classic();
    }

    printf("\n=== Summary ===\n");
    printf("Hijacks executed : %d\n", hijack_count);
    printf("Best valence     : %.5f\n", best_ever);
    printf("Memory used      : %d/%d slots\n",
           (memory_count < MEMORY_SIZE ? memory_count : MEMORY_SIZE), MEMORY_SIZE);
    return 0;
}
