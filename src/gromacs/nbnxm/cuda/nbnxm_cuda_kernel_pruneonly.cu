/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017,2019,2020,2021, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#include "gmxpre.h"

#include "nbnxm_cuda_kernel_pruneonly.cuh"

#include "gromacs/nbnxm/cuda/nbnxm_cuda.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/nbnxm/nbnxm_gpu_internal.h"

#ifndef FUNCTION_DECLARATION_ONLY
/* Instantiate external template functions */
template __global__ void
nbnxn_kernel_prune_cuda<false>(const NBAtomDataGpu, const NBParamGpu, const Nbnxm::gpu_plist, int, int);
template __global__ void
nbnxn_kernel_prune_cuda<true>(const NBAtomDataGpu, const NBParamGpu, const Nbnxm::gpu_plist, int, int);
#endif

namespace Nbnxm
{

/*! Calculates the amount of shared memory required by the CUDA kernel in use. */
static inline int calc_shmem_required_prune(const int num_threads_z)
{
    int shmem;

    /* i-atom x in shared memory */
    shmem = c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(float4);
    /* cj in shared memory, for each warp separately */
    shmem += num_threads_z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(int);

    return shmem;
}

void launchNbnxmKernelPruneOnly(NbnxmGpu*                      nb,
                                const gmx::InteractionLocality iloc,
                                const int                      numParts,
                                const int                      part,
                                const int                      numSciInPart,
                                CommandEvent*                  timingEvent)
{
    NBAtomDataGpu*      adat         = nb->atdat;
    NBParamGpu*         nbp          = nb->nbparam;
    gpu_plist*          plist        = nb->plist[iloc];
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    /* Kernel launch config:
     * - The thread block dimensions match the size of i-clusters, j-clusters,
     *   and j-cluster concurrency, in x, y, and z, respectively.
     * - The 1D block-grid contains as many blocks as super-clusters.
     */
    int num_threads_z = c_pruneKernelJ4Concurrency;
    int nblock        = calc_nb_kernel_nblock(numSciInPart, &nb->deviceContext_->deviceInfo());
    KernelLaunchConfig config;
    config.blockSize[0]     = c_clSize;
    config.blockSize[1]     = c_clSize;
    config.blockSize[2]     = num_threads_z;
    config.gridSize[0]      = nblock;
    config.sharedMemorySize = calc_shmem_required_prune(num_threads_z);

    if (debug)
    {
        fprintf(debug,
                "Pruning GPU kernel launch configuration:\n\tThread block: %zux%zux%zu\n\t"
                "\tGrid: %zux%zu\n\t#Super-clusters/clusters: %d/%d (%d)\n"
                "\tShMem: %zu\n",
                config.blockSize[0],
                config.blockSize[1],
                config.blockSize[2],
                config.gridSize[0],
                config.gridSize[1],
                numSciInPart * c_nbnxnGpuNumClusterPerSupercluster,
                c_nbnxnGpuNumClusterPerSupercluster,
                plist->na_c,
                config.sharedMemorySize);
    }

    constexpr char kernelName[] = "k_pruneonly";
    const auto     kernel =
            plist->haveFreshList ? nbnxn_kernel_prune_cuda<true> : nbnxn_kernel_prune_cuda<false>;
    const auto kernelArgs = prepareGpuKernelArguments(kernel, config, adat, nbp, plist, &numParts, &part);
    launchGpuKernel(kernel, config, deviceStream, timingEvent, kernelName, kernelArgs);
}

} // namespace Nbnxm
