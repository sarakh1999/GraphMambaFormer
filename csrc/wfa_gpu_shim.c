/*
 * Stable C-ABI shim over WFA-GPU's public aligner API.
 *
 * WFA-GPU (https://github.com/quim0/WFA-GPU, MIT) is the maintained,
 * CUDA-12-capable stand-in for the archived GenomeWorks ``cudaaligner`` module:
 * it computes batched gap-affine pairwise alignments *with the CIGAR on the
 * GPU*. This shim keeps every WFA-GPU struct on the C side and exposes a flat
 * handle + scalar accessors so the Python binding
 * (graphmambaformer/accel/wfa_gpu_ops.py) can drive it through ctypes without
 * knowing any struct layouts.
 *
 * Built into libgmf_wfa_gpu.so by scripts/build_wfa_gpu.sh, which links it
 * against WFA-GPU's libwfagpu.so and WFA2-lib's libwfa.so.
 */
#include <stdlib.h>
#include "include/wfa_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate + initialize an aligner. Returns an opaque handle or NULL. */
void* gmf_wfagpu_create(void) {
    wfagpu_aligner_t* aligner = (wfagpu_aligner_t*) calloc(1, sizeof(wfagpu_aligner_t));
    if (!aligner) {
        return NULL;
    }
    if (!wfagpu_initialize_aligner(aligner)) {
        free(aligner);
        return NULL;
    }
    return (void*) aligner;
}

/* Queue one (query, target) ASCII pair. Returns 0 on success, -1 on failure. */
int gmf_wfagpu_add(void* handle, const char* query, const char* target) {
    if (!handle || !query || !target) {
        return -1;
    }
    return wfagpu_add_sequences((wfagpu_aligner_t*) handle, query, target) ? 0 : -1;
}

/* Number of sequence pairs currently queued. */
long gmf_wfagpu_num_pairs(void* handle) {
    if (!handle) {
        return 0;
    }
    return (long) ((wfagpu_aligner_t*) handle)->num_sequence_pairs;
}

/*
 * Initialize parameters (must be called *after* all sequences are added, as the
 * library derives default error bounds from the first pair) and run the
 * alignment. ``compute_cigar`` != 0 produces the ASCII CIGAR; a positive
 * ``batch_size`` / ``max_error`` / ``band`` overrides the library default.
 * Returns 0 on success, -1 on failure.
 */
int gmf_wfagpu_run(void* handle,
                   int x, int o, int e,
                   int compute_cigar,
                   long batch_size,
                   int max_error,
                   int band) {
    if (!handle) {
        return -1;
    }
    wfagpu_aligner_t* aligner = (wfagpu_aligner_t*) handle;
    affine_penalties_t penalties;
    penalties.x = x;
    penalties.o = o;
    penalties.e = e;
    if (!wfagpu_initialize_parameters(aligner, penalties)) {
        return -1;
    }
    if (batch_size > 0) {
        if (!wfagpu_set_batch_size(aligner, (size_t) batch_size)) {
            return -1;
        }
    }
    if (max_error > 0) {
        aligner->alignment_options.max_error = max_error;
    }
    if (band > 0) {
        aligner->alignment_options.band = band;
    }
    aligner->alignment_options.compute_cigar = compute_cigar ? true : false;
    return wfagpu_align(aligner) ? 0 : -1;
}

/* Alignment score / edit-distance of pair ``i`` (valid after a successful run). */
unsigned int gmf_wfagpu_error(void* handle, long i) {
    return ((wfagpu_aligner_t*) handle)->results[i].error;
}

/*
 * ASCII CIGAR buffer of pair ``i`` (valid until destroy). NULL if the CIGAR was
 * not computed. The caller must copy it before gmf_wfagpu_destroy().
 */
const char* gmf_wfagpu_cigar(void* handle, long i) {
    return (const char*) ((wfagpu_aligner_t*) handle)->results[i].cigar.buffer;
}

/* Free all aligner + result memory. */
void gmf_wfagpu_destroy(void* handle) {
    if (!handle) {
        return;
    }
    wfagpu_destroy_aligner((wfagpu_aligner_t*) handle);
    free(handle);
}

#ifdef __cplusplus
}
#endif
