/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016 by the GROMACS development team.
 * Copyright (c) 2017,2018,2019,2020,2021, by the GROMACS development team, led by
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
/*! \internal \file
 *  \brief Define OpenCL implementation of nbnxm_gpu.h
 *
 *  \author Anca Hamuraru <anca@streamcomputing.eu>
 *  \author Teemu Virolainen <teemu@streamcomputing.eu>
 *  \author Dimitrios Karkoulis <dimitris.karkoulis@gmail.com>
 *  \author Szilárd Páll <pall.szilard@gmail.com>
 *  \ingroup module_nbnxm
 *
 *  TODO (psz):
 *  - Add a static const cl_uint c_pruneKernelWorkDim / c_nbnxnKernelWorkDim = 3;
 *  - Rework the copying of OCL data structures done before every invocation of both
 *    nb and prune kernels (using fillin_ocl_structures); also consider at the same
 *    time calling clSetKernelArg only on the updated parameters (if tracking changed
 *    parameters is feasible);
 *  - Consider using the event_wait_list argument to clEnqueueNDRangeKernel to mark
 *    dependencies on the kernel launched: e.g. the non-local nb kernel's dependency
 *    on the misc_ops_and_local_H2D_done event could be better expressed this way.
 *
 *  - Consider extracting common sections of the OpenCL and CUDA nbnxn logic, e.g:
 *    - in nbnxn_gpu_launch_kernel_pruneonly() the pre- and post-kernel launch logic
 *      is identical in the two implementations, so a 3-way split might allow sharing
 *      code;
 *    -
 *
 */
#include "gmxpre.h"

#include <assert.h>
#include <stdlib.h>

#if defined(_MSVC)
#    include <limits>
#endif

#include "gromacs/gpu_utils/oclutils.h"
#include "gromacs/nbnxm/nbnxm_gpu_internal.h"
#include "gromacs/nbnxm/opencl/nbnxm_ocl.h"

#include "nbnxm_ocl_types.h"

namespace Nbnxm
{

/*! \brief Convenience constants */
//@{
static constexpr int c_clSize = c_nbnxnGpuClusterSize;
//@}

/*! \brief Return a pointer to the prune kernel version to be executed at the current invocation.
 *
 * \param[in] kernel_pruneonly  array of prune kernel objects
 * \param[in] firstPrunePass    true if the first pruning pass is being executed
 */
static inline cl_kernel selectPruneKernel(cl_kernel kernel_pruneonly[], bool firstPrunePass)
{
    cl_kernel* kernelPtr;

    if (firstPrunePass)
    {
        kernelPtr = &(kernel_pruneonly[epruneFirst]);
    }
    else
    {
        kernelPtr = &(kernel_pruneonly[epruneRolling]);
    }
    // TODO: consider creating the prune kernel object here to avoid a
    // clCreateKernel for the rolling prune kernel if this is not needed.
    return *kernelPtr;
}

/*! \brief Calculates the amount of shared memory required by the prune kernel.
 *
 *  Note that for the sake of simplicity we use the CUDA terminology "shared memory"
 *  for OpenCL local memory.
 *
 * \param[in] num_threads_z cj4 concurrency equal to the number of threads/work items in the 3-rd
 * dimension. \returns   the amount of local memory in bytes required by the pruning kernel
 */
static inline int calc_shmem_required_prune(const int num_threads_z)
{
    int shmem;

    /* i-atom x in shared memory (for convenience we load all 4 components including q) */
    shmem = c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(float) * 4;
    /* cj in shared memory, for each warp separately
     * Note: only need to load once per wavefront, but to keep the code simple,
     * for now we load twice on AMD.
     */
    shmem += num_threads_z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(int);
    /* Warp vote, requires one uint per warp/32 threads per block. */
    shmem += sizeof(cl_uint) * 2 * num_threads_z;

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
    /* kernel launch config */
    KernelLaunchConfig config;
    config.sharedMemorySize = calc_shmem_required_prune(num_threads_z);
    config.blockSize[0]     = c_clSize;
    config.blockSize[1]     = c_clSize;
    config.blockSize[2]     = num_threads_z;
    config.gridSize[0]      = numSciInPart;

    validate_global_work_size(config, 3, &nb->deviceContext_->deviceInfo());

    if (debug)
    {
        fprintf(debug,
                "Pruning GPU kernel launch configuration:\n\tLocal work size: %zux%zux%zu\n\t"
                "\tGlobal work size: %zux%zu\n\t#Super-clusters/clusters: %d/%d (%d)\n"
                "\tShMem: %zu\n",
                config.blockSize[0],
                config.blockSize[1],
                config.blockSize[2],
                config.blockSize[0] * config.gridSize[0],
                config.blockSize[1] * config.gridSize[1],
                plist->nsci * c_nbnxnGpuNumClusterPerSupercluster,
                c_nbnxnGpuNumClusterPerSupercluster,
                plist->na_c,
                config.sharedMemorySize);
    }

    cl_nbparam_params_t nbparams_params;
    fillin_ocl_structures(nbp, &nbparams_params);

    constexpr char kernelName[] = "k_pruneonly";
    const auto     pruneKernel  = selectPruneKernel(nb->kernel_pruneonly, plist->haveFreshList);
    const auto     kernelArgs   = prepareGpuKernelArguments(pruneKernel,
                                                      config,
                                                      &nbparams_params,
                                                      &adat->xq,
                                                      &adat->shiftVec,
                                                      &plist->sci,
                                                      &plist->cj4,
                                                      &plist->imask,
                                                      &numParts,
                                                      &part);
    launchGpuKernel(pruneKernel, config, deviceStream, timingEvent, kernelName, kernelArgs);
}

} // namespace Nbnxm
