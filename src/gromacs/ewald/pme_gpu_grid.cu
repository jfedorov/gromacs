/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021, by the GROMACS development team, led by
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
 * \brief Implements PME GPU halo exchange and PME GPU - Host FFT grid conversion
 * functions. These functions are used for PME decomposition in mixed-mode
 *
 * \author Gaurav Garg <gaugarg@nvidia.com>
 *
 * \ingroup module_ewald
 */

#include "gmxpre.h"

#include "pme_gpu_grid.h"

#include "config.h"

#include <cstdlib>

#include "gromacs/math/vec.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/devicebuffer.cuh"

#include "pme.cuh"
#include "pme_gpu_types_host.h"
#include "pme_gpu_types.h"
#include "pme_gpu_types_host_impl.h"
#include "gromacs/fft/parallel_3dfft.h"

/*! \brief
 * A CUDA kernel which packs non-contiguous overlap data in Y-dimension
 *
 * \param[in] gm_realGrid          local grid
 * \param[in] gm_transferGrid      device array used to pack data
 * \param[in] offset               offset of y-overlap region
 * \param[in] overlapSize          overlap Size in y-overlap region
 * \param[in] pmeSize              Local PME grid size
 */
static __global__ void pmeGpuPackHaloY(const float* __restrict__ gm_realGrid,
                                       float* __restrict__ gm_transferGrid,
                                       int  offset,
                                       int  overlapSize,
                                       int3 pmeSize)
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int ix = threadIdx.z + blockIdx.z * blockDim.z;

    // we might get iz greather than pmeSize.z when pmeSize.z is not
    // multiple of threadsAlongZDim(see below)
    if (iz >= pmeSize.z)
    {
        return;
    }

    int pmeIndex    = ix * pmeSize.y * pmeSize.z + (iy + offset) * pmeSize.z + iz;
    int packedIndex = ix * overlapSize * pmeSize.z + iy * pmeSize.z + iz;

    gm_transferGrid[packedIndex] = gm_realGrid[pmeIndex];
}

/*! \brief
 * A CUDA kernel which adds/puts grid overlap data received from neighboring rank in Y-dim
 *
 * \param[in] gm_realGrid          local grid
 * \param[in] gm_transferGrid      overlapping region from neighboring rank
 * \param[in] starty               offset of y-overlap region
 * \param[in] overlapSize          overlap Size in y-overlap region
 * \param[in] pmeSize              Local PME grid size
 */
template<bool forward>
static __global__ void pmeGpuAddHaloY(float* __restrict__ gm_realGrid,
                                      const float* __restrict__ gm_transferGrid,
                                      int  offset,
                                      int  overlapSize,
                                      int3 pmeSize)
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int ix = threadIdx.z + blockIdx.z * blockDim.z;

    // we might get iz greather than pmeSize.z when pmeSize.z is not
    // multiple of threadsAlongZDim(see below)
    if (iz >= pmeSize.z)
    {
        return;
    }

    int pmeIndex    = ix * pmeSize.y * pmeSize.z + (iy + offset) * pmeSize.z + iz;
    int packedIndex = ix * overlapSize * pmeSize.z + iy * pmeSize.z + iz;

    if (forward)
    {
        gm_realGrid[pmeIndex] += gm_transferGrid[packedIndex];
    }
    else
    {
        gm_realGrid[pmeIndex] = gm_transferGrid[packedIndex];
    }
}

/*! \brief
 * A CUDA kernel which adds grid overlap data received from neighboring rank
 *
 * \param[in] gm_realGrid          local grid
 * \param[in] gm_transferGrid      overlapping region from neighboring rank
 * \param[in] size                 Number of elements in overlap region
 */
static __global__ void pmeGpuAddHalo(float* __restrict__ gm_realGrid,
                                     const float* __restrict__ gm_transferGrid,
                                     int size)
{
    int val = threadIdx.x + blockIdx.x * blockDim.x;
    if (val < size)
    {
        gm_realGrid[val] += gm_transferGrid[val];
    }
}

/*! \brief
 * A CUDA kernel which copies data from pme grid to FFT grid and back
 *
 * \param[in] gm_pmeGrid          local PME grid
 * \param[in] gm_fftGrid          local FFT grid
 * \param[in] fft_ndata           local FFT grid size without padding
 * \param[in] fft_size            local FFT grid padded size
 * \param[in] pme_size            local PME grid padded size
 */
template<bool forward>
static __global__ void pmegrid_to_fftgrid(float* __restrict__ gm_realGrid,
                                          float* __restrict__ gm_fftGrid,
                                          int3 fft_ndata,
                                          int3 fft_size,
                                          int3 pme_size)
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int ix = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= fft_ndata.x || iy >= fft_ndata.y || iz >= fft_ndata.z)
    {
        return;
    }

    int fftidx   = ix * fft_size.y * fft_size.z + iy * fft_size.z + iz;
    int pmeIndex = ix * pme_size.y * pme_size.z + iy * pme_size.z + iz;

    if (forward)
    {
        gm_fftGrid[fftidx] = gm_realGrid[pmeIndex];
    }
    else
    {
        gm_realGrid[pmeIndex] = gm_fftGrid[fftidx];
    }
}

/*! \brief
 * Launches CUDA kernel to pack non-contiguous overlap data in Y-dimension
 *
 * \param[in]  pmeGpu              The PME GPU structure.
 * \param[in] overlapSize          overlap Size in y-overlap region
 * \param[in] yOffset              offset of y-overlap region
 * \param[in] localXSize           Local x size
 * \param[in] pmeSize              PME grid size
 * \param[in] realGrid             local grid
 * \param[in] packrdGrid           device array used to pack data
 */
static void packYData(const PmeGpu* pmeGpu,
                      int           overlapSize,
                      int           yOffset,
                      int           localXSize,
                      const ivec&   pmeSize,
                      float*        realGrid,
                      float*        packrdGrid)
{
    // keeping same as warp size for better coalescing
    // Not keeping to higher value such as 64 to avoid high masked out
    // inactive threads as FFT grid sizes tend to be quite small
    const int threadsAlongZDim = 32;

    // right grid
    KernelLaunchConfig config;
    config.blockSize[0]     = threadsAlongZDim;
    config.blockSize[1]     = overlapSize;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (pmeSize[ZZ] + threadsAlongZDim - 1) / threadsAlongZDim;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = localXSize;
    config.sharedMemorySize = 0;


    auto kernelFn = pmeGpuPackHaloY;

    auto kernelArgs = prepareGpuKernelArguments(
            kernelFn, config, &realGrid, &packrdGrid, &yOffset, &overlapSize, &pmeSize);

    launchGpuKernel(kernelFn,
                    config,
                    pmeGpu->archSpecific->pmeStream_,
                    nullptr,
                    "PME Domdec GPU Pack Grid Halo Exchange",
                    kernelArgs);
}

/*! \brief
 * Launches CUDA kernel to reduce/unpack overlap data in Y-dimension
 *
 * \param[in]  pmeGpu              The PME GPU structure.
 * \param[in] overlapSize          overlap Size in y-overlap region
 * \param[in] yOffset              offset of y-overlap region
 * \param[in] localXSize           Local x size
 * \param[in] pmeSize              PME grid size
 * \param[in] realGrid             local grid
 * \param[in] packrdGrid           device array used to pack data
 */
template<bool forward>
static void reduceYData(const PmeGpu* pmeGpu,
                        int           overlapSize,
                        int           yOffset,
                        int           localXSize,
                        const ivec&   pmeSize,
                        float*        realGrid,
                        float*        packrdGrid)
{
    // keeping same as warp size for better coalescing
    // Not keeping to higher value such as 64 to avoid high masked out
    // inactive threads as FFT grid sizes tend to be quite small
    const int threadsAlongZDim = 32;

    // right grid
    KernelLaunchConfig config;
    config.blockSize[0]     = threadsAlongZDim;
    config.blockSize[1]     = overlapSize;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (pmeSize[ZZ] + threadsAlongZDim - 1) / threadsAlongZDim;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = localXSize;
    config.sharedMemorySize = 0;

    auto kernelFn = pmeGpuAddHaloY<forward>;

    auto kernelArgs = prepareGpuKernelArguments(
            kernelFn, config, &realGrid, &packrdGrid, &yOffset, &overlapSize, &pmeSize);

    launchGpuKernel(kernelFn,
                    config,
                    pmeGpu->archSpecific->pmeStream_,
                    nullptr,
                    "PME Domdec GPU Pack Grid Halo Exchange",
                    kernelArgs);
}

/*! \brief
 * Launches CUDA kernel to reduce overlap data in X-dimension
 *
 * \param[in]  pmeGpu              The PME GPU structure.
 * \param[in] overlapSize          overlap Size in y-overlap region
 * \param[in] realGrid             local grid
 * \param[in] packrdGrid           device array used to pack data
 */
static void reduceXData(const PmeGpu* pmeGpu, int overlapSize, float* realGrid, float* packrdGrid)
{
    // launch reduction kernel
    const int threadsPerBlock = 64;

    KernelLaunchConfig config;
    config.blockSize[0]     = threadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (overlapSize + threadsPerBlock - 1) / threadsPerBlock;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    auto kernelFn = pmeGpuAddHalo;

    auto kernelArgs = prepareGpuKernelArguments(kernelFn, config, &realGrid, &packrdGrid, &overlapSize);

    launchGpuKernel(kernelFn,
                    config,
                    pmeGpu->archSpecific->pmeStream_,
                    nullptr,
                    "PME Domdec GPU Apply Grid Halo Exchange",
                    kernelArgs);
}

void pmeGpuGridHaloExchange(const PmeGpu* pmeGpu)
{
    // Note here we are assuming that width of the chunks is not so small that we need to
    // transfer to/from multiple ranks i.e. that the distributed grid is at least order-1 points wide.

    auto* kernelParamsPtr = pmeGpu->kernelParams.get();
    ivec  local_pme_size;
    local_pme_size[XX] = kernelParamsPtr->grid.realGridSizePadded[XX];
    local_pme_size[YY] = kernelParamsPtr->grid.realGridSizePadded[YY];
    local_pme_size[ZZ] = kernelParamsPtr->grid.realGridSizePadded[ZZ];

    int extraGridLines = ceil(pmeGpu->common->rlist / pmeGpu->common->spacing);
    int overlapSize    = pmeGpu->common->pme_order - 1 + extraGridLines;

    // minor dimension
    if (pmeGpu->common->nnodes_minor > 1)
    {
        int rank  = pmeGpu->common->nodeid_minor;
        int size  = pmeGpu->common->nnodes_minor;
        int right = (rank + 1) % size;
        int left  = (rank + size - 1) % size;

        // Note that s2g0[size] is the grid size (array is allocated to size+1)
        int myGrid    = pmeGpu->common->s2g0y[rank + 1] - pmeGpu->common->s2g0y[rank];
        int rightGrid = pmeGpu->common->s2g0y[right + 1] - pmeGpu->common->s2g0y[right];
        int leftGrid  = pmeGpu->common->s2g0y[left + 1] - pmeGpu->common->s2g0y[left];

        // current implementation transfers from/to only immediate neighbours
        // in case overlap size is > slab width, we need to transfer data to multiple neighbours
        // Or, we should put a release assert which will NOT allow runs if overlapSize > slab size
        int overlapRecv  = std::min(overlapSize, myGrid);
        int overlapRight = std::min(overlapSize, rightGrid);
        int overlapLeft  = std::min(overlapSize, leftGrid);

        int pmegrid_nkx = pmeGpu->common->pmegrid_nk[XX];

        for (int gridIndex = 0; gridIndex < pmeGpu->common->ngrids; gridIndex++)
        {
            // launch packing kernel
            float* realGrid = pmeGpu->kernelParams->grid.d_realGrid[gridIndex];

            // Pack data that needs to be sent to right rank
            packYData(pmeGpu,
                      overlapRight,
                      myGrid,
                      pmegrid_nkx,
                      local_pme_size,
                      realGrid,
                      pmeGpu->archSpecific->d_sendGridRighty);

            // Pack data that needs to be sent to left rank
            packYData(pmeGpu,
                      overlapLeft,
                      local_pme_size[YY] - overlapLeft,
                      pmegrid_nkx,
                      local_pme_size,
                      realGrid,
                      pmeGpu->archSpecific->d_sendGridLefty);

            // synchronize before starting halo exchange
            pme_gpu_synchronize(pmeGpu);

            int        tag = 403; // Arbitrarily chosen
            MPI_Status status;

            // send data to right rank and recv from left rank
            MPI_Sendrecv(pmeGpu->archSpecific->d_sendGridRighty,
                         overlapRight * pmegrid_nkx * local_pme_size[ZZ],
                         MPI_FLOAT,
                         right,
                         tag,
                         pmeGpu->archSpecific->d_recvGridLefty,
                         overlapRecv * pmegrid_nkx * local_pme_size[ZZ],
                         MPI_FLOAT,
                         left,
                         tag,
                         pmeGpu->common->mpi_commy,
                         &status);

            // send data to left rank and recv from right rank
            MPI_Sendrecv(pmeGpu->archSpecific->d_sendGridLefty,
                         overlapLeft * pmegrid_nkx * local_pme_size[ZZ],
                         MPI_FLOAT,
                         left,
                         tag,
                         pmeGpu->archSpecific->d_recvGridRighty,
                         overlapRecv * pmegrid_nkx * local_pme_size[ZZ],
                         MPI_FLOAT,
                         right,
                         tag,
                         pmeGpu->common->mpi_commy,
                         &status);

            // reduce data received from left rank
            reduceYData<true>(pmeGpu,
                              overlapRecv,
                              0,
                              pmegrid_nkx,
                              local_pme_size,
                              realGrid,
                              pmeGpu->archSpecific->d_recvGridLefty);

            // reduce data received from right rank
            reduceYData<true>(pmeGpu,
                              overlapRecv,
                              myGrid - overlapRecv,
                              pmegrid_nkx,
                              local_pme_size,
                              realGrid,
                              pmeGpu->archSpecific->d_recvGridRighty);
        }
    }

    // major dimension
    if (pmeGpu->common->nnodes_major > 1)
    {
        int rank  = pmeGpu->common->nodeid_major;
        int size  = pmeGpu->common->nnodes_major;
        int right = (rank + 1) % size;
        int left  = (rank + size - 1) % size;

        // Note that s2g0[size] is the grid size (array is allocated to size+1)
        int myGrid    = pmeGpu->common->s2g0x[rank + 1] - pmeGpu->common->s2g0x[rank];
        int rightGrid = pmeGpu->common->s2g0x[right + 1] - pmeGpu->common->s2g0x[right];
        int leftGrid  = pmeGpu->common->s2g0x[left + 1] - pmeGpu->common->s2g0x[left];

        // current implementation transfers from/to only immediate neighbours
        // in case overlap size is > slab width, we need to transfer data to multiple neighbours
        // Or, we should put a release assert which will NOT allow runs if overlapSize > slab size
        int overlapRecv  = std::min(overlapSize, myGrid);
        int overlapRight = std::min(overlapSize, rightGrid);
        int overlapLeft  = std::min(overlapSize, leftGrid);

        int transferStartRight = myGrid * local_pme_size[YY] * local_pme_size[ZZ];
        int transferStartLeft =
                (local_pme_size[XX] - overlapLeft) * local_pme_size[YY] * local_pme_size[ZZ];

        // Current implementation transfers the whole grid along y, an optimization is
        // possible where only local y-length can be trasnferred
        // But, this will require executing packing kernel
        int transferSizeSendRight = overlapRight * local_pme_size[YY] * local_pme_size[ZZ];
        int transferSizeSendLeft  = overlapLeft * local_pme_size[YY] * local_pme_size[ZZ];
        int transferSizeRecv      = overlapRecv * local_pme_size[YY] * local_pme_size[ZZ];

        for (int gridIndex = 0; gridIndex < pmeGpu->common->ngrids; gridIndex++)
        {
            float* realGrid = pmeGpu->kernelParams->grid.d_realGrid[gridIndex];

            // synchronize before starting halo exchange
            pme_gpu_synchronize(pmeGpu);

            int tag = 403; // Arbitrarily chosen

            MPI_Status status;
            // send data to right rank and recv from left rank
            MPI_Sendrecv(&realGrid[transferStartRight],
                         transferSizeSendRight,
                         MPI_FLOAT,
                         right,
                         tag,
                         pmeGpu->archSpecific->d_recvGridLeftx,
                         transferSizeRecv,
                         MPI_FLOAT,
                         left,
                         tag,
                         pmeGpu->common->mpi_commx,
                         &status);

            // send data to left rank and recv from right rank
            MPI_Sendrecv(&realGrid[transferStartLeft],
                         transferSizeSendLeft,
                         MPI_FLOAT,
                         left,
                         tag,
                         pmeGpu->archSpecific->d_recvGridRightx,
                         transferSizeRecv,
                         MPI_FLOAT,
                         right,
                         tag,
                         pmeGpu->common->mpi_commx,
                         &status);

            // reduce data received from left rank
            reduceXData(pmeGpu, transferSizeRecv, realGrid, pmeGpu->archSpecific->d_recvGridLeftx);

            // reduce data received from right rank
            int    offset       = (myGrid - overlapRecv) * local_pme_size[YY] * local_pme_size[ZZ];
            float* offsetedGrid = realGrid + offset;
            reduceXData(pmeGpu, transferSizeRecv, offsetedGrid, pmeGpu->archSpecific->d_recvGridRightx);
        }
    }
}

void pmeGpuGridHaloExchangeReverse(const PmeGpu* pmeGpu)
{
    auto* kernelParamsPtr = pmeGpu->kernelParams.get();
    ivec  local_pme_size;
    local_pme_size[XX] = kernelParamsPtr->grid.realGridSizePadded[XX];
    local_pme_size[YY] = kernelParamsPtr->grid.realGridSizePadded[YY];
    local_pme_size[ZZ] = kernelParamsPtr->grid.realGridSizePadded[ZZ];

    int extraGridLines = ceil(pmeGpu->common->rlist / pmeGpu->common->spacing);
    int overlapSize    = pmeGpu->common->pme_order - 1 + extraGridLines;

    // minor dimension
    if (pmeGpu->common->nnodes_minor > 1)
    {
        int rank  = pmeGpu->common->nodeid_minor;
        int size  = pmeGpu->common->nnodes_minor;
        int right = (rank + 1) % size;
        int left  = (rank + size - 1) % size;

        int myGrid    = pmeGpu->common->s2g0y[rank + 1] - pmeGpu->common->s2g0y[rank];
        int rightGrid = pmeGpu->common->s2g0y[right + 1] - pmeGpu->common->s2g0y[right];
        int leftGrid  = pmeGpu->common->s2g0y[left + 1] - pmeGpu->common->s2g0y[left];

        // current implementation transfers from/to only immediate neighbours
        // in case overlap size is > slab width, we need to transfer data to multiple neighbours
        // Or, we should put a release assert which will NOT allow runs if overlapSize > slab size
        int overlapSend  = std::min(overlapSize, myGrid);
        int overlapRight = std::min(overlapSize, rightGrid);
        int overlapLeft  = std::min(overlapSize, leftGrid);

        int pmegrid_nkx = pmeGpu->common->pmegrid_nk[XX];

        for (int gridIndex = 0; gridIndex < pmeGpu->common->ngrids; gridIndex++)
        {
            // launch packing kernel
            float* realGrid = pmeGpu->kernelParams->grid.d_realGrid[gridIndex];

            // Pack data that needs to be sent to left rank
            packYData(pmeGpu,
                      overlapSend,
                      0,
                      pmegrid_nkx,
                      local_pme_size,
                      realGrid,
                      pmeGpu->archSpecific->d_sendGridLefty);

            // Pack data that needs to be sent to right rank
            packYData(pmeGpu,
                      overlapSend,
                      (myGrid - overlapSend),
                      pmegrid_nkx,
                      local_pme_size,
                      realGrid,
                      pmeGpu->archSpecific->d_sendGridRighty);

            // synchronize before starting halo exchange
            pme_gpu_synchronize(pmeGpu);

            int        tag = 403; // Arbitrarily chosen
            MPI_Status status;

            // send data to left rank and recv from right rank
            MPI_Sendrecv(pmeGpu->archSpecific->d_sendGridLefty,
                         overlapSend * pmegrid_nkx * local_pme_size[ZZ],
                         MPI_FLOAT,
                         left,
                         tag,
                         pmeGpu->archSpecific->d_recvGridRighty,
                         overlapRight * pmegrid_nkx * local_pme_size[ZZ],
                         MPI_FLOAT,
                         right,
                         tag,
                         pmeGpu->common->mpi_commy,
                         &status);

            // send data to right rank and recv from left rank
            MPI_Sendrecv(pmeGpu->archSpecific->d_sendGridRighty,
                         overlapSend * pmegrid_nkx * local_pme_size[ZZ],
                         MPI_FLOAT,
                         right,
                         tag,
                         pmeGpu->archSpecific->d_recvGridLefty,
                         overlapLeft * pmegrid_nkx * local_pme_size[ZZ],
                         MPI_FLOAT,
                         left,
                         tag,
                         pmeGpu->common->mpi_commy,
                         &status);

            // unpack data received from right rank
            reduceYData<false>(pmeGpu,
                               overlapRight,
                               myGrid,
                               pmegrid_nkx,
                               local_pme_size,
                               realGrid,
                               pmeGpu->archSpecific->d_recvGridRighty);

            // unpack data received from left rank
            reduceYData<false>(pmeGpu,
                               overlapLeft,
                               local_pme_size[YY] - overlapLeft,
                               pmegrid_nkx,
                               local_pme_size,
                               realGrid,
                               pmeGpu->archSpecific->d_recvGridLefty);
        }
    }

    // major dimension
    if (pmeGpu->common->nnodes_major > 1)
    {
        int rank  = pmeGpu->common->nodeid_major;
        int size  = pmeGpu->common->nnodes_major;
        int right = (rank + 1) % size;
        int left  = (rank + size - 1) % size;

        int myGrid    = pmeGpu->common->s2g0x[rank + 1] - pmeGpu->common->s2g0x[rank];
        int rightGrid = pmeGpu->common->s2g0x[right + 1] - pmeGpu->common->s2g0x[right];
        int leftGrid  = pmeGpu->common->s2g0x[left + 1] - pmeGpu->common->s2g0x[left];

        // current implementation transfers from/to only immediate neighbours
        // in case overlap size is > slab width, we need to transfer data to multiple neighbours
        // Or, we should put a release assert which will NOT allow runs if overlapSize > slab size
        int overlapSend  = std::min(overlapSize, myGrid);
        int overlapRight = std::min(overlapSize, rightGrid);
        int overlapLeft  = std::min(overlapSize, leftGrid);

        int transferstartRight = myGrid * local_pme_size[YY] * local_pme_size[ZZ];
        int transferstartLeft =
                (local_pme_size[XX] - overlapLeft) * local_pme_size[YY] * local_pme_size[ZZ];

        // Current implementation transfers the whole grid along y, an optimization is
        // possible where only local y-length can be trasnferred
        // But, this will require executing packing kernel
        int transferSizeSend      = overlapSend * local_pme_size[YY] * local_pme_size[ZZ];
        int transferSizeRecvRight = overlapRight * local_pme_size[YY] * local_pme_size[ZZ];
        int transferSizeRecvLeft  = overlapLeft * local_pme_size[YY] * local_pme_size[ZZ];

        for (int gridIndex = 0; gridIndex < pmeGpu->common->ngrids; gridIndex++)
        {
            float* realGrid = pmeGpu->kernelParams->grid.d_realGrid[gridIndex];

            pme_gpu_synchronize(pmeGpu);
            const int  tag = 403; // Arbitrarily chosen
            MPI_Status status;

            // send data to left rank and recv from right rank
            MPI_Sendrecv(&realGrid[0],
                         transferSizeSend,
                         MPI_FLOAT,
                         left,
                         tag,
                         &realGrid[transferstartRight],
                         transferSizeRecvRight,
                         MPI_FLOAT,
                         right,
                         tag,
                         pmeGpu->common->mpi_commx,
                         &status);

            // send data to right rank and recv from left rank
            int offset = (myGrid - overlapSend) * local_pme_size[YY] * local_pme_size[ZZ];
            MPI_Sendrecv(&realGrid[offset],
                         transferSizeSend,
                         MPI_FLOAT,
                         right,
                         tag,
                         &realGrid[transferstartLeft],
                         transferSizeRecvLeft,
                         MPI_FLOAT,
                         left,
                         tag,
                         pmeGpu->common->mpi_commx,
                         &status);
        }
    }
}

template<bool forward>
void convertPmeGridToFftGrid(const PmeGpu* pmeGpu, float* h_grid, gmx_parallel_3dfft_t* pfft_setup, const int gridIndex)
{
    ivec local_fft_ndata, local_fft_offset, local_fft_size;
    ivec local_pme_size;

    gmx_parallel_3dfft_real_limits(pfft_setup[gridIndex], local_fft_ndata, local_fft_offset, local_fft_size);

    local_pme_size[XX] = pmeGpu->kernelParams->grid.realGridSizePadded[XX];
    local_pme_size[YY] = pmeGpu->kernelParams->grid.realGridSizePadded[YY];
    local_pme_size[ZZ] = pmeGpu->kernelParams->grid.realGridSizePadded[ZZ];

    // this should be true in case of slab decomposition
    if (local_pme_size[ZZ] == local_fft_size[ZZ] && local_pme_size[YY] == local_fft_size[YY])
    {
        int fftSize = local_fft_size[ZZ] * local_fft_size[YY] * local_fft_ndata[XX];
        if (forward)
        {
            copyFromDeviceBuffer(h_grid,
                                 &pmeGpu->kernelParams->grid.d_realGrid[gridIndex],
                                 0,
                                 fftSize,
                                 pmeGpu->archSpecific->pmeStream_,
                                 pmeGpu->settings.transferKind,
                                 nullptr);
        }
        else
        {
            copyToDeviceBuffer(&pmeGpu->kernelParams->grid.d_realGrid[gridIndex],
                               h_grid,
                               0,
                               fftSize,
                               pmeGpu->archSpecific->pmeStream_,
                               pmeGpu->settings.transferKind,
                               nullptr);
        }
    }
    else
    {
        // launch copy kernel
        KernelLaunchConfig config;
        config.blockSize[0] = 32;
        config.blockSize[1] = 4;
        config.blockSize[2] = 1;
        config.gridSize[0]  = (local_fft_ndata[ZZ] + config.blockSize[0] - 1) / config.blockSize[0];
        config.gridSize[1]  = (local_fft_ndata[YY] + config.blockSize[1] - 1) / config.blockSize[1];
        config.gridSize[2]  = local_fft_ndata[XX];
        config.sharedMemorySize = 0;

        auto kernelFn = pmegrid_to_fftgrid<forward>;

        const auto kernelArgs =
                prepareGpuKernelArguments(kernelFn,
                                          config,
                                          &pmeGpu->kernelParams->grid.d_realGrid[gridIndex],
                                          &h_grid,
                                          &local_fft_ndata,
                                          &local_fft_size,
                                          &local_pme_size);

        launchGpuKernel(kernelFn,
                        config,
                        pmeGpu->archSpecific->pmeStream_,
                        nullptr,
                        "Convert PME grid to FFT grid",
                        kernelArgs);
    }

    if (forward)
    {
        pmeGpu->archSpecific->syncSpreadGridD2H.markEvent(pmeGpu->archSpecific->pmeStream_);
    }
}

template void convertPmeGridToFftGrid<true>(const PmeGpu*         pmeGpu,
                                            float*                h_grid,
                                            gmx_parallel_3dfft_t* pfft_setup,
                                            const int             gridIndex);

template void convertPmeGridToFftGrid<false>(const PmeGpu*         pmeGpu,
                                             float*                h_grid,
                                             gmx_parallel_3dfft_t* pfft_setup,
                                             const int             gridIndex);
