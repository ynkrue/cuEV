/**
 * @file   cuev_mp.h
 * @brief  Public API for cuEV — distributed multi-GPU symmetric dense eigensolver.
 *
 * Requires building with -DCUEV_ENABLE_MP=ON (links libcuev_mp).
 * For single-GPU usage see cuev.h.
 *
 * Matrix distribution: 2D block-cyclic, BLACS-compatible process grid p×q.
 * Distributed operations via cuBLASMp + NCCL.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#ifdef CUEV_ENABLE_MP

#include <cuda_runtime.h>

namespace cuev {
namespace mp {

// TODO: decide on matrix descriptor (BLACS int[9], cublasMpMatrixDescriptor_t, or custom)
// TODO: decide on communicator handle (NCCL ncclComm_t, cuBLASMp handle, or both)

// symm_eig_solve declaration goes here once descriptors are settled.

} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
