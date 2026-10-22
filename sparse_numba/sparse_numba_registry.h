/* sparse_numba_registry.h - the handle registry of the sparse_numba shims
 * and the build facts they report for cache_key(),
 * shared by the SuperLU, UMFPACK and Eigen shims (C and C++).
 *
 * The including file defines SN_REGISTRY_FIRST and SN_REGISTRY_LAST, the
 * backend ids its registry hands out (sparse_numba_codes.h: one id for
 * SuperLU and for UMFPACK, EIGEN_LU .. EIGEN_LDLT for the Eigen shim), and
 * includes this header once. Its state is static: each extension module
 * holds a registry of its own, and a handle of another module's backend
 * reads as SN_HANDLE_WRONG_BACKEND. Nothing here prints. */
#ifndef SPARSE_NUMBA_REGISTRY_H
#define SPARSE_NUMBA_REGISTRY_H

#include <stdint.h>
#include <stdlib.h>
#include "sparse_numba_codes.h"

#if !defined(SN_REGISTRY_FIRST) || !defined(SN_REGISTRY_LAST)
#error "define SN_REGISTRY_FIRST and SN_REGISTRY_LAST before sparse_numba_registry.h"
#endif


/* ================================================================
 * The handle registry: a process-global table of slots in fixed chunks
 * that never move. A slot holds the generation of its live occupant and
 * the factors pointer; the handle the caller sees is
 * (generation << 32) | (backend << SN_SLOT_BITS) | slot. Lookups are
 * lock-free reads of a fixed slot (one load and one compare); taking and
 * releasing a slot, and adding a chunk, run under a spin lock. free
 * advances the generation, so a stale handle reads as freed by name.
 * ================================================================ */

#define SN_CHUNK_BITS 12
#define SN_CHUNK_SIZE (1 << SN_CHUNK_BITS)
#define SN_MAX_SLOTS (1 << SN_SLOT_BITS)
#define SN_MAX_CHUNKS (SN_MAX_SLOTS / SN_CHUNK_SIZE)
#define SN_GEN_MAX 0x7FFFFFFFLL

typedef struct {
    int64_t generation;   /* the live occupant's; advanced by every release */
    void *factors;        /* NULL while the slot is free */
    int32_t next_free;
} sn_slot_t;

static sn_slot_t *sn_chunks[SN_MAX_CHUNKS];
static int32_t sn_high_water = 0;
static int32_t sn_free_head = -1;
static char sn_lock_flag = 0;

static void sn_lock(void) {
    while (__atomic_test_and_set(&sn_lock_flag, __ATOMIC_ACQUIRE)) {
    }
}

static void sn_unlock(void) {
    __atomic_clear(&sn_lock_flag, __ATOMIC_RELEASE);
}

/* a slot for factors under a backend id of this registry */
static int sn_register(void *factors, int backend, int64_t *handle_out) {
    int32_t idx;
    sn_slot_t *slot;
    int64_t gen;

    if (backend < SN_REGISTRY_FIRST || backend > SN_REGISTRY_LAST) return SN_HANDLE_INVALID;
    sn_lock();
    if (sn_free_head >= 0) {
        idx = sn_free_head;
        slot = &sn_chunks[idx >> SN_CHUNK_BITS][idx & (SN_CHUNK_SIZE - 1)];
        sn_free_head = slot->next_free;
    } else {
        if (sn_high_water >= SN_MAX_SLOTS) {
            sn_unlock();
            return SN_ALLOC_FAILED;
        }
        idx = sn_high_water;
        if ((idx & (SN_CHUNK_SIZE - 1)) == 0) {
            sn_slot_t *chunk = (sn_slot_t*)calloc(SN_CHUNK_SIZE, sizeof(sn_slot_t));
            if (!chunk) {
                sn_unlock();
                return SN_ALLOC_FAILED;
            }
            for (int i = 0; i < SN_CHUNK_SIZE; i++) chunk[i].generation = 1;
            __atomic_store_n(&sn_chunks[idx >> SN_CHUNK_BITS], chunk, __ATOMIC_RELEASE);
        }
        sn_high_water = idx + 1;
        slot = &sn_chunks[idx >> SN_CHUNK_BITS][idx & (SN_CHUNK_SIZE - 1)];
    }
    gen = slot->generation;
    __atomic_store_n(&slot->factors, factors, __ATOMIC_RELEASE);
    sn_unlock();

    *handle_out = (gen << 32) | ((int64_t)backend << SN_SLOT_BITS) | (int64_t)idx;
    return SN_OK;
}

static int sn_decode(int64_t handle, sn_slot_t **slot_out) {
    int backend;
    int64_t gen, live;
    int32_t idx;
    sn_slot_t *chunk;

    if (handle <= 0) return SN_HANDLE_INVALID;
    backend = (int)((handle >> SN_SLOT_BITS) & 0xFF);
    if (backend < SN_REGISTRY_FIRST || backend > SN_REGISTRY_LAST) {
        if (backend >= SN_BACKEND_SUPERLU && backend <= SN_BACKEND_EIGEN_LDLT)
            return SN_HANDLE_WRONG_BACKEND;
        return SN_HANDLE_INVALID;
    }
    gen = handle >> 32;
    idx = (int32_t)(handle & (SN_MAX_SLOTS - 1));
    if (gen <= 0) return SN_HANDLE_INVALID;
    chunk = __atomic_load_n(&sn_chunks[idx >> SN_CHUNK_BITS], __ATOMIC_ACQUIRE);
    if (!chunk) return SN_HANDLE_INVALID;
    *slot_out = &chunk[idx & (SN_CHUNK_SIZE - 1)];
    live = __atomic_load_n(&(*slot_out)->generation, __ATOMIC_ACQUIRE);
    if (gen < live) return SN_HANDLE_FREED;
    if (gen > live) return SN_HANDLE_INVALID;
    return SN_OK;
}

static int sn_lookup(int64_t handle, void **factors_out) {
    sn_slot_t *slot = NULL;
    void *factors;
    int code = sn_decode(handle, &slot);
    if (code != SN_OK) return code;
    factors = __atomic_load_n(&slot->factors, __ATOMIC_ACQUIRE);
    if (!factors) return SN_HANDLE_INVALID;  /* a generation never handed out */
    *factors_out = factors;
    return SN_OK;
}

static int sn_release(int64_t handle, void **factors_out) {
    sn_slot_t *slot = NULL;
    int code;

    sn_lock();
    code = sn_decode(handle, &slot);
    if (code == SN_OK) {
        if (!slot->factors) {
            code = SN_HANDLE_INVALID;
        } else {
            int64_t next = slot->generation + 1;
            *factors_out = slot->factors;
            __atomic_store_n(&slot->factors, (void*)NULL, __ATOMIC_RELEASE);
            __atomic_store_n(&slot->generation, next, __ATOMIC_RELEASE);
            if (next < SN_GEN_MAX) {  /* else the slot retires */
                slot->next_free = sn_free_head;
                sn_free_head = (int32_t)(handle & (SN_MAX_SLOTS - 1));
            }
        }
    }
    sn_unlock();
    return code;
}


/* ================================================================
 * The build facts for cache_key(): the compiler and its code-generation
 * flags as the compiler reports them, and SN_BUILD_FLAGS, the text of
 * setup.py's compile flags for the extension (passed as -DSN_BUILD_FLAGS;
 * "unknown" when the shim is built without it)
 * ================================================================ */

#define SN_STR2(x) #x
#define SN_STR(x) SN_STR2(x)

#if defined(__clang__)
#define SN_CC "clang " __clang_version__
#elif defined(__GNUC__)
#define SN_CC "gcc " __VERSION__
#else
#define SN_CC "unknown"
#endif

#ifdef __OPTIMIZE__
#define SN_OPT "1"
#else
#define SN_OPT "0"
#endif

#ifdef __FAST_MATH__
#define SN_FAST_MATH "1"
#else
#define SN_FAST_MATH "0"
#endif

#if defined(__AVX512F__)
#define SN_SIMD "avx512f"
#elif defined(__AVX2__)
#define SN_SIMD "avx2"
#elif defined(__AVX__)
#define SN_SIMD "avx"
#elif defined(__SSE2__)
#define SN_SIMD "sse2"
#else
#define SN_SIMD "none"
#endif

#ifdef __FMA__
#define SN_FMA "1"
#else
#define SN_FMA "0"
#endif

#ifndef SN_BUILD_FLAGS
#define SN_BUILD_FLAGS "unknown"
#endif

/* "cc=...,optimize=...,fast_math=...,simd=...,fma=...,flags=..." */
#define SN_BUILD_FACTS \
    "cc=" SN_CC ",optimize=" SN_OPT ",fast_math=" SN_FAST_MATH \
    ",simd=" SN_SIMD ",fma=" SN_FMA ",flags=" SN_BUILD_FLAGS

#endif /* SPARSE_NUMBA_REGISTRY_H */
