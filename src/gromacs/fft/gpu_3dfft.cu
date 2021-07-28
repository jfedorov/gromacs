/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017,2018,2019,2020,2021, by the GROMACS development team, led by
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
 *  \brief Implements GPU 3D FFT routines for CUDA.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \ingroup module_fft
 */

#include "gmxpre.h"

#include "gpu_3dfft.h"

#include <cufft.h>

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/devicebuffer.cuh"
#include "gromacs/ewald/pme_gpu_types_host.h"
#include "gromacs/ewald/pme_gpu_internal.h"
#include "gromacs/ewald/pme.cuh"

#define UCX_MPIALLTOALLV_BUG_HACK 1

namespace gmx
{

class Gpu3dFft::Impl
{
public:
    Impl(const PmeGpu*        pmeGpu,
         ivec                 realGridSize,
         ivec                 realGridSizePadded,
         ivec                 complexGridSize,
         ivec                 complexGridSizePadded,
         bool                 useDecomposition,
         bool                 performOutOfPlaceFFT,
         const DeviceContext& context,
         const DeviceStream&  pmeStream,
         DeviceBuffer<float>  realGrid,
         DeviceBuffer<float>  complexGrid,
         DeviceBuffer<float>  complexGrid2);
    ~Impl();

    cufftHandle   planR2C_;
    cufftHandle   planC2R_;
    cufftReal*    realGrid_;
    cufftComplex* complexGrid_;
    cufftComplex* complexGrid2_;
    ivec complexGridSizePadded_;

    const PmeGpu* pmeGpu_;

    /*! \brief
     * CUDA stream used for PME computation
     */
    const DeviceStream& stream_;

    /*! \brief
     * 2D and 1D cufft plans used for distributed fft implementation
     */
    cufftHandle planR2C2D_;
    cufftHandle planC2R2D_;
    cufftHandle planC2C1D_;

    /*! \brief
     * MPI complex type
     */
    MPI_Datatype complexType_;

    /*! \brief
     * MPI communicator for PME ranks
     */
    MPI_Comm mpi_comm_;

    /*! \brief
     * total ranks within PME group
     */
    int mpiSize_;

    /*! \brief
     * current local mpi rank within PME group
     */
    int mpiRank_;

    /*! \brief
     * Max local grid size in X-dim (used during transposes in forward pass)
     */
    int xMax_;

    /*! \brief
     * Max local grid size in Y-dim (used during transposes in reverse pass)
     */
    int yMax_;

    /*! \brief
     * device array containing 1D decomposition size in X-dim (forwarad pass)
     */
    DeviceBuffer<int> d_xBlockSizes_;

    /*! \brief
     * device array containing 1D decomposition size in Y-dim (reverse pass)
     */
    DeviceBuffer<int> d_yBlockSizes_;

    /*! \brief
     * device arrays for local interpolation grid start values in X-dim
     * (used during transposes in forward pass)
     */
    DeviceBuffer<int> d_s2g0x_;

    /*! \brief
     * device arrays for local interpolation grid start values in Y-dim
     * (used during transposes in reverse pass)
     */
    DeviceBuffer<int> d_s2g0y_;

    /*! \brief
     * host array containing 1D decomposition size in X-dim (forwarad pass)
     */
    gmx::HostVector<int> h_xBlockSizes_;

    /*! \brief
     * host array containing 1D decomposition size in Y-dim (reverse pass)
     */
    gmx::HostVector<int> h_yBlockSizes_;

    /*! \brief
     * host array for local interpolation grid start values in Y-dim
     */
    gmx::HostVector<int> h_s2g0y_;

    /*! \brief
     * device array big enough to hold grid overlapping region
     * used during grid halo exchange
     */
    DeviceBuffer<float> d_transferGrid_;

    /*! \brief
     * count and displacement arrays used in MPI_Alltoall call
     *
     */
    int *sendCount_, *sendDisp_;
    int *recvCount_, *recvDisp_;

#        if UCX_MPIALLTOALLV_BUG_HACK
    /*! \brief
     * count arrays used in MPI_Alltoall call which has no self copies
     *
     */
    int *sendCountTemp_, *recvCountTemp_;
#        endif
};

static void handleCufftError(cufftResult_t status, const char* msg)
{
    if (status != CUFFT_SUCCESS)
    {
        gmx_fatal(FARGS, "%s (error code %d)\n", msg, status);
    }
}

// CUDA block size x and y-dim
constexpr int c_threads = 16;

/*! \brief
 * A CUDA kernel which converts grid from XYZ to YZX layout in case of forward fft
 * and converts from YZX to XYZ layout in case of reverse fft
 *
 * \tparam[in] forward            Forward pass or reverse pass
 *
 * \param[in] gm_arrayIn          Input local grid
 * \param[in] gm_arrayOut         Output local grid in converted layout
 * \param[in] sizeX               Grid size in X-dim.
 * \param[in] sizeY               Grid size in Y-dim.
 * \param[in] sizeZ               Grid size in Z-dim.
 */
template<bool forward>
static __global__ void transposeXyzToYzxKernel(const cufftComplex* __restrict__ gm_arrayIn,
                                               cufftComplex* __restrict__ gm_arrayOut,
                                               const int sizeX,
                                               const int sizeY,
                                               const int sizeZ)
{
    __shared__ cufftComplex sm_temp[c_threads][c_threads];
    int                     x = blockIdx.x * blockDim.x + threadIdx.x;
    int                     y = blockIdx.y;
    int                     z = blockIdx.z * blockDim.z + threadIdx.z;

    // use threads in other order for xyz (works as blockDim.x == blockDim.z)
    int xt = blockIdx.x * blockDim.x + threadIdx.z;
    int zt = blockIdx.z * blockDim.z + threadIdx.x;

    int  xyzIndex = zt + y * sizeZ + xt * sizeY * sizeZ;
    int  yzxIndex = x + z * sizeX + y * sizeX * sizeZ;
    int  inIndex, outIndex;
    bool validIn, validOut;

    if (forward) // xyz to yzx
    {
        inIndex  = xyzIndex;
        outIndex = yzxIndex;
        validIn  = (xt < sizeX && zt < sizeZ);
        validOut = (x < sizeX && z < sizeZ);
    }
    else // yzx to xyz
    {
        inIndex  = yzxIndex;
        outIndex = xyzIndex;
        validIn  = (x < sizeX && z < sizeZ);
        validOut = (xt < sizeX && zt < sizeZ);
    }

    if (validIn)
    {
        sm_temp[threadIdx.x][threadIdx.z] = gm_arrayIn[inIndex];
    }
    __syncthreads();

    if (validOut)
    {
        gm_arrayOut[outIndex] = sm_temp[threadIdx.z][threadIdx.x];
    }
}

/*! \brief
 * A CUDA kernel which merges multiple blocks in YZX layout from different ranks
 *
 * \param[in] gm_arrayIn          Input local grid
 * \param[in] gm_arrayOut         Output local grid in converted layout
 * \param[in] sizeX               Grid size in X-dim.
 * \param[in] sizeY               Grid size in Y-dim.
 * \param[in] sizeZ               Grid size in Z-dim.
 * \param[in] xBlockSizes         Array containing X-block sizes for each rank
 * \param[in] xOffset             Array containing grid offsets for each rank
 */
static __global__ void convertBlockedYzxToYzxKernel(const cufftComplex* __restrict__ gm_arrayIn,
                                                    cufftComplex* __restrict__ gm_arrayOut,
                                                    const int sizeX,
                                                    const int sizeY,
                                                    const int sizeZ,
                                                    const int* __restrict__ xBlockSizes,
                                                    const int* __restrict__ xOffset)
{
    // no need to cache block unless x_block_size is small
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int region = blockIdx.z;
    int xLocal = thread % xBlockSizes[region];
    int z      = thread / xBlockSizes[region];
    int y      = blockIdx.y;
    int x      = xOffset[region] + xLocal;

    int indexIn  = xLocal + xBlockSizes[region] * (z + sizeZ * y) + xOffset[region] * sizeY * sizeZ;
    int indexOut = x + sizeX * (z + sizeZ * y);

    if (x < xOffset[region + 1] && z < sizeZ)
    {
        gm_arrayOut[indexOut] = gm_arrayIn[indexIn];
    }
}

/*! \brief
 * A CUDA kernel which merges multiple blocks in XYZ layout from different ranks
 *
 * \param[in] gm_arrayIn          Input local grid
 * \param[in] gm_arrayOut         Output local grid in converted layout
 * \param[in] sizeX               Grid size in X-dim.
 * \param[in] sizeY               Grid size in Y-dim.
 * \param[in] sizeZ               Grid size in Z-dim.
 * \param[in] yBlockSizes         Array containing Y-block sizes for each rank
 * \param[in] yOffset             Array containing grid offsets for each rank
 */
static __global__ void convertBlockedXyzToXyzKernel(const cufftComplex* __restrict__ gm_arrayIn,
                                                    cufftComplex* __restrict__ gm_arrayOut,
                                                    const int sizeX,
                                                    const int sizeY,
                                                    const int sizeZ,
                                                    const int* __restrict__ yBlockSizes,
                                                    const int* __restrict__ yOffset)
{
    int x      = blockIdx.y;
    int yz     = blockIdx.x * blockDim.x + threadIdx.x;
    int region = blockIdx.z;
    int z      = yz % sizeZ;
    int yLocal = yz / sizeZ;

    int y        = yLocal + yOffset[region];
    int indexIn  = z + sizeZ * (yLocal + yBlockSizes[region] * x + sizeX * yOffset[region]);
    int indexOut = z + sizeZ * (y + sizeY * x);

    if (y < yOffset[region + 1] && z < sizeZ)
    {
        gm_arrayOut[indexOut] = gm_arrayIn[indexIn];
    }
}

template<bool forward>
static void transposeXyzToYzx(cufftComplex* arrayIn, cufftComplex* arrayOut, int sizeX, int sizeY, int sizeZ, const DeviceStream& stream)
{
    KernelLaunchConfig config;
    config.blockSize[0]     = c_threads;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = c_threads;
    config.gridSize[0]      = (sizeX + c_threads - 1) / c_threads;
    config.gridSize[1]      = sizeY;
    config.gridSize[2]      = (sizeZ + c_threads - 1) / c_threads;
    config.sharedMemorySize = 0;


    auto kernelFn = transposeXyzToYzxKernel<forward>;

    const auto kernelArgs =
            prepareGpuKernelArguments(kernelFn, config, &arrayIn, &arrayOut, &sizeX, &sizeY, &sizeZ);

    launchGpuKernel(kernelFn, config, stream, nullptr, "PME FFT GPU grid transpose", kernelArgs);
}

static void convertBlockedYzxToYzx(cufftComplex* arrayIn,
                                              cufftComplex* arrayOut,
                                              int           sizeX,
                                              int           sizeY,
                                              int           sizeZ,
                                              int*          xBlockSizes,
                                              int*          xOffsets,
                                              int           numRegions,
                                              int           maxRegionSize,
                                              const DeviceStream& stream)
{
    int blockDim = c_threads * c_threads;
    int sizexz   = maxRegionSize * sizeZ;

    KernelLaunchConfig config;
    config.blockSize[0]     = blockDim;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (sizexz + blockDim - 1) / blockDim;
    config.gridSize[1]      = sizeY;
    config.gridSize[2]      = numRegions;
    config.sharedMemorySize = 0;


    auto kernelFn = convertBlockedYzxToYzxKernel;

    const auto kernelArgs = prepareGpuKernelArguments(
            kernelFn, config, &arrayIn, &arrayOut, &sizeX, &sizeY, &sizeZ, &xBlockSizes, &xOffsets);

    launchGpuKernel(kernelFn, config, stream, nullptr, "PME FFT GPU grid rearrange", kernelArgs);
}

static void convertBlockedXyzToXyz(cufftComplex* arrayIn,
                                    cufftComplex* arrayOut,
                                    int           sizeX,
                                    int           sizeY,
                                    int           sizeZ,
                                    int*          yBlockSizes,
                                    int*          yOffsets,
                                    int           numRegions,
                                    int           maxRegionSize,
                                    const DeviceStream& stream)
{
    int blockDim = c_threads * c_threads;
    int sizexz   = maxRegionSize * sizeZ;

    KernelLaunchConfig config;
    config.blockSize[0]     = blockDim;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (sizexz + blockDim - 1) / blockDim;
    config.gridSize[1]      = sizeX;
    config.gridSize[2]      = numRegions;
    config.sharedMemorySize = 0;


    auto kernelFn = convertBlockedXyzToXyzKernel;

    const auto kernelArgs = prepareGpuKernelArguments(
            kernelFn, config, &arrayIn, &arrayOut, &sizeX, &sizeY, &sizeZ, &yBlockSizes, &yOffsets);

    launchGpuKernel(kernelFn, config, stream, nullptr, "PME FFT GPU grid rearrange", kernelArgs);
}

Gpu3dFft::Impl::Impl(const PmeGpu* pmeGpu,
                     ivec       realGridSize,
                     ivec       realGridSizePadded,
                     ivec       complexGridSize,
                     ivec       complexGridSizePadded,
                     const bool useDecomposition,
                     const bool /*performOutOfPlaceFFT*/,
                     const DeviceContext& context,
                     const DeviceStream& pmeStream,
                     DeviceBuffer<float> realGrid,
                     DeviceBuffer<float> complexGrid,
                     DeviceBuffer<float> complexGrid2) :
    realGrid_(reinterpret_cast<cufftReal*>(realGrid)),
    complexGrid_(reinterpret_cast<cufftComplex*>(complexGrid)),
    complexGrid2_(reinterpret_cast<cufftComplex*>(complexGrid2)),
    pmeGpu_(pmeGpu),
    stream_(pmeStream)
{
    for (int i = 0; i < DIM; i++)
    {
        complexGridSizePadded_[i] = complexGridSizePadded[i];
    }
    const int complexGridSizePaddedTotal =
            complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ];
    const int realGridSizePaddedTotal =
            realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ];

    realGrid_ = realGrid;

    GMX_RELEASE_ASSERT(realGrid_, "Bad (null) input real-space grid");
    GMX_RELEASE_ASSERT(complexGrid_, "Bad (null) input complex grid");


    cufftResult_t result;
    /* Commented code for a simple 3D grid with no padding */
    /*
       result = cufftPlan3d(&planR2C_, realGridSize[XX], realGridSize[YY], realGridSize[ZZ],
       CUFFT_R2C); handleCufftError(result, "cufftPlan3d R2C plan failure");

       result = cufftPlan3d(&planC2R_, realGridSize[XX], realGridSize[YY], realGridSize[ZZ],
       CUFFT_C2R); handleCufftError(result, "cufftPlan3d C2R plan failure");
     */

    cudaStream_t stream = pmeStream.stream();
    GMX_RELEASE_ASSERT(stream, "Can not use the default CUDA stream for PME cuFFT");

    if (!useDecomposition)
    {
        int rank  = 3;
        int batch = 1;
        result    = cufftPlanMany(&planR2C_,
                               rank,
                               realGridSize,
                               realGridSizePadded,
                               1,
                               realGridSizePaddedTotal,
                               complexGridSizePadded,
                               1,
                               complexGridSizePaddedTotal,
                               CUFFT_R2C,
                               batch);
        handleCufftError(result, "cufftPlanMany R2C plan failure");
        result = cufftSetStream(planR2C_, stream);
        handleCufftError(result, "cufftSetStream R2C failure");


        result = cufftPlanMany(&planC2R_,
                               rank,
                               realGridSize,
                               complexGridSizePadded,
                               1,
                               complexGridSizePaddedTotal,
                               realGridSizePadded,
                               1,
                               realGridSizePaddedTotal,
                               CUFFT_C2R,
                               batch);
        handleCufftError(result, "cufftPlanMany C2R plan failure");
        result = cufftSetStream(planC2R_, stream);
        handleCufftError(result, "cufftSetStream C2R failure");
    }

    int mpiSize   = 1;
    int mpiRank   = 0;

    // count and displacement arrays used in MPI_Alltoall call
    sendCount_ = sendDisp_ = recvCount_ = recvDisp_ = NULL;
#    if UCX_MPIALLTOALLV_BUG_HACK
    sendCountTemp_ = recvCountTemp_ = NULL;
#    endif

    // local grid size along decmposed dimension
    d_xBlockSizes_ = d_yBlockSizes_ = NULL;

    // device arrays keeping local grid offsets
    d_s2g0x_ = d_s2g0y_ = NULL;

    // device memory to transfer overlapping regions between ranks
    d_transferGrid_ = NULL;
    if (useDecomposition)
    {
        changePinningPolicy(&h_xBlockSizes_, gmx::PinningPolicy::PinnedIfSupported);
        changePinningPolicy(&h_yBlockSizes_, gmx::PinningPolicy::PinnedIfSupported);
        changePinningPolicy(&h_s2g0y_, gmx::PinningPolicy::PinnedIfSupported);

        const int complexGridSizePaddedTotal2D = complexGridSizePadded[YY] * complexGridSizePadded[ZZ];
        const int realGridSizePaddedTotal2D    = realGridSizePadded[YY] * realGridSizePadded[ZZ];

        int localx = realGridSize[XX];
        int localy = realGridSize[YY];

        MPI_Comm_size(pmeGpu->common->mpi_commx, &mpiSize);
        MPI_Comm_rank(pmeGpu->common->mpi_commx, &mpiRank);
        mpi_comm_  = pmeGpu->common->mpi_commx;
        sendCount_ = (int*)malloc(mpiSize * sizeof(int));
        sendDisp_  = (int*)malloc(mpiSize * sizeof(int));
        recvCount_ = (int*)malloc(mpiSize * sizeof(int));
        recvDisp_  = (int*)malloc(mpiSize * sizeof(int));
        h_xBlockSizes_.resize(mpiSize);
        h_yBlockSizes_.resize(mpiSize);
        h_s2g0y_.resize(mpiSize + 1);
        allocateDeviceBuffer(&d_xBlockSizes_, mpiSize, context);
        allocateDeviceBuffer(&d_yBlockSizes_, mpiSize, context);
        allocateDeviceBuffer(&d_s2g0x_, (mpiSize + 1), context);
        allocateDeviceBuffer(&d_s2g0y_, (mpiSize + 1), context);

        localx = pmeGpu_->common->s2g0x[mpiRank + 1] - pmeGpu_->common->s2g0x[mpiRank];

        for (int i = 0; i < mpiSize; i++)
        {
            h_s2g0y_[i] = (i * complexGridSizePadded[YY] + 0) / mpiSize;
        }
        h_s2g0y_[mpiSize] = complexGridSizePadded[YY];

        localy        = h_s2g0y_[mpiRank + 1] - h_s2g0y_[mpiRank];
        int totalSend = 0;
        int totalRecv = 0;
        int xmax      = 0;
        int ymax      = 0;
        for (int i = 0; i < mpiSize; i++)
        {
            int ix            = pmeGpu_->common->s2g0x[i + 1] - pmeGpu_->common->s2g0x[i];
            int iy            = h_s2g0y_[i + 1] - h_s2g0y_[i];
            h_xBlockSizes_[i] = ix;
            h_yBlockSizes_[i] = iy;
            if (xmax < ix)
                xmax = ix;
            if (ymax < iy)
                ymax = iy;
            sendCount_[i] = complexGridSize[ZZ] * localx * iy;
            recvCount_[i] = complexGridSize[ZZ] * localy * ix;
            sendDisp_[i]  = totalSend;
            recvDisp_[i]  = totalRecv;
            totalSend += sendCount_[i];
            totalRecv += recvCount_[i];
        }
        xMax_ = xmax;
        yMax_ = ymax;
        copyToDeviceBuffer(
                &d_s2g0x_, pmeGpu_->common->s2g0x.data(), 0, (mpiSize + 1), stream_, GpuApiCallBehavior::Sync, nullptr);
        copyToDeviceBuffer(
                &d_xBlockSizes_, h_xBlockSizes_.data(), 0, mpiSize, stream_, GpuApiCallBehavior::Async, nullptr);
        copyToDeviceBuffer(
                &d_yBlockSizes_, h_yBlockSizes_.data(), 0, mpiSize, stream_, GpuApiCallBehavior::Async, nullptr);
        copyToDeviceBuffer(
                &d_s2g0y_, h_s2g0y_.data(), 0, (mpiSize + 1), stream_, GpuApiCallBehavior::Async, nullptr);

        allocateDeviceBuffer(
                &d_transferGrid_, xmax * realGridSizePadded[YY] * realGridSizePadded[ZZ], context);

#    if UCX_MPIALLTOALLV_BUG_HACK
        sendCountTemp_ = (int*)malloc(mpiSize * sizeof(int));
        recvCountTemp_ = (int*)malloc(mpiSize * sizeof(int));

        memcpy(sendCountTemp_, sendCount_, mpiSize * sizeof(int));
        memcpy(recvCountTemp_, recvCount_, mpiSize * sizeof(int));

        // don't make any self copies. UCX has perf issues with self copies
        sendCountTemp_[mpiRank] = 0;
        recvCountTemp_[mpiRank] = 0;
#    endif

        int rank  = 2;
        int batch = localx;
        // split 3d fft as 2D fft and 1d fft to implement distributed fft
        result = cufftPlanMany(&planR2C2D_,
                               rank,
                               &realGridSize[YY],
                               &realGridSizePadded[YY],
                               1,
                               realGridSizePaddedTotal2D,
                               &complexGridSizePadded[YY],
                               1,
                               complexGridSizePaddedTotal2D,
                               CUFFT_R2C,
                               batch);
        handleCufftError(result, "cufftPlanMany 2D R2C plan failure");
        result = cufftSetStream(planR2C2D_, stream);
        handleCufftError(result, "cufftSetStream R2C failure");

        result = cufftPlanMany(&planC2R2D_,
                               rank,
                               &realGridSize[YY],
                               &complexGridSizePadded[YY],
                               1,
                               complexGridSizePaddedTotal2D,
                               &realGridSizePadded[YY],
                               1,
                               realGridSizePaddedTotal2D,
                               CUFFT_C2R,
                               batch);
        handleCufftError(result, "cufftPlanMany 2D C2R plan failure");
        result = cufftSetStream(planC2R2D_, stream);
        handleCufftError(result, "cufftSetStream C2R failure");

        rank   = 1;
        batch  = localy * complexGridSize[ZZ];
        result = cufftPlanMany(&planC2C1D_,
                               rank,
                               &complexGridSize[XX], // 1D C2C part of the R2C
                               &complexGridSizePadded[XX],
                               1,
                               complexGridSizePadded[XX],
                               &complexGridSizePadded[XX],
                               1,
                               complexGridSizePadded[XX],
                               CUFFT_C2C,
                               batch);
        handleCufftError(result, "cufftPlanMany  1D C2C plan failure");
        result = cufftSetStream(planC2C1D_, stream);
        handleCufftError(result, "cufftSetStream C2C failure");

        MPI_Type_contiguous(2, MPI_FLOAT, &complexType_);
        MPI_Type_commit(&complexType_);
    }
    mpiSize_ = mpiSize;
    mpiRank_ = mpiRank;
}

Gpu3dFft::Impl::~Impl()
{
    cufftResult_t result;
    if (!pme_gpu_settings(pmeGpu_).useDecomposition)
    {
        result = cufftDestroy(planR2C_);
        handleCufftError(result, "cufftDestroy R2C failure");
        result = cufftDestroy(planC2R_);
        handleCufftError(result, "cufftDestroy C2R failure");
    }
    else
    {
        result = cufftDestroy(planR2C2D_);
        handleCufftError(result, "cufftDestroy R2C failure");
        result = cufftDestroy(planC2R2D_);
        handleCufftError(result, "cufftDestroy C2R failure");
        result = cufftDestroy(planC2C1D_);
        handleCufftError(result, "cufftDestroy C2C failure");

        MPI_Type_free(&complexType_);

        free(sendCount_);
        free(sendDisp_);
        free(recvCount_);
        free(recvDisp_);
        freeDeviceBuffer(&d_xBlockSizes_);
        freeDeviceBuffer(&d_yBlockSizes_);
        freeDeviceBuffer(&d_s2g0x_);
        freeDeviceBuffer(&d_s2g0y_);

#    if UCX_MPIALLTOALLV_BUG_HACK
        free(sendCountTemp_);
        free(recvCountTemp_);

#    endif // UCX_MPIALLTOALLV_BUG_HACK
    }
}

void Gpu3dFft::perform3dFft(gmx_fft_direction dir, CommandEvent* /*timingEvent*/)
{
    cufftResult_t result;
    if (!pme_gpu_settings(impl_->pmeGpu_).useDecomposition)
    {
        if (dir == GMX_FFT_REAL_TO_COMPLEX)
        {
            result = cufftExecR2C(impl_->planR2C_, impl_->realGrid_, impl_->complexGrid_);
            handleCufftError(result, "cuFFT R2C execution failure");
        }
        else
        {
            result = cufftExecC2R(impl_->planC2R_, impl_->complexGrid_, impl_->realGrid_);
            handleCufftError(result, "cuFFT C2R execution failure");
        }
    }
    else
    {
        int                           localx, localy;
        localx = impl_->pmeGpu_->common->s2g0x[impl_->mpiRank_ + 1] - impl_->pmeGpu_->common->s2g0x[impl_->mpiRank_];
        localy = impl_->h_s2g0y_[impl_->mpiRank_ + 1] - impl_->h_s2g0y_[impl_->mpiRank_];

        if (dir == GMX_FFT_REAL_TO_COMPLEX)
        {
            // 2D FFT
            result = cufftExecR2C(impl_->planR2C2D_, impl_->realGrid_, impl_->complexGrid_);
            handleCufftError(result, "cuFFT R2C 2D execution failure");
            // Transpose and communicate
            transposeXyzToYzx<true>(
                    impl_->complexGrid_, impl_->complexGrid2_, localx, impl_->complexGridSizePadded_[YY], impl_->complexGridSizePadded_[ZZ], impl_->stream_);
            pme_gpu_synchronize(impl_->pmeGpu_);

#    if UCX_MPIALLTOALLV_BUG_HACK

            // self copy on the same rank
            cudaMemcpyAsync(impl_->complexGrid_ + impl_->recvDisp_[impl_->mpiRank_],
                            impl_->complexGrid2_ + impl_->sendDisp_[impl_->mpiRank_],
                            impl_->recvCount_[impl_->mpiRank_] * sizeof(cufftComplex),
                            cudaMemcpyDeviceToDevice,
                            impl_->stream_.stream());

            // copy to other ranks. UCX has perf issues if self copies are made in MPI_Alltoallv call
            MPI_Alltoallv(impl_->complexGrid2_,
                          impl_->sendCountTemp_,
                          impl_->sendDisp_,
                          impl_->complexType_,
                          impl_->complexGrid_,
                          impl_->recvCountTemp_,
                          impl_->recvDisp_,
                          impl_->complexType_,
                          impl_->mpi_comm_);

#    else
            // MPI_Alltoallv has perf issues where copy to self is too slow. above implementation takes care of that
            MPI_Alltoallv(impl_->complexGrid2_,
                          impl_->sendCount_,
                          impl_->sendDisp_,
                          impl_->complexType_,
                          impl_->complexGrid_,
                          impl_->recvCount_,
                          impl_->recvDisp_,
                          impl_->complexType_,
                          impl_->mpi_comm_);
#    endif

            // make data in proper layout once different blocks are received from different MPI ranks
            convertBlockedYzxToYzx(impl_->complexGrid_,
                                   impl_->complexGrid2_,
                                   impl_->complexGridSizePadded_[XX],
                                   localy,
                                   impl_->complexGridSizePadded_[ZZ],
                                   impl_->d_xBlockSizes_,
                                   impl_->d_s2g0x_,
                                   impl_->mpiSize_,
                                   impl_->xMax_,
                                   impl_->stream_);
            // 1D FFT
            result = cufftExecC2C(impl_->planC2C1D_, impl_->complexGrid2_, impl_->complexGrid_, CUFFT_FORWARD);
            handleCufftError(result, "cuFFT C2C 1D execution failure");
        }
        else
        {
            // 1D FFT
            result = cufftExecC2C(impl_->planC2C1D_, impl_->complexGrid_, impl_->complexGrid2_, CUFFT_INVERSE);
            handleCufftError(result, "cuFFT C2C 1D execution failure");
            // transpose and communicate
            transposeXyzToYzx<false>(
                    impl_->complexGrid2_, impl_->complexGrid_, impl_->complexGridSizePadded_[XX], localy, impl_->complexGridSizePadded_[ZZ], impl_->stream_);
            pme_gpu_synchronize(impl_->pmeGpu_);

#    if UCX_MPIALLTOALLV_BUG_HACK
            // self copy on the same rank
            cudaMemcpyAsync(impl_->complexGrid2_ + impl_->recvDisp_[impl_->mpiRank_],
                            impl_->complexGrid_ + impl_->sendDisp_[impl_->mpiRank_],
                            impl_->recvCount_[impl_->mpiRank_] * sizeof(cufftComplex),
                            cudaMemcpyDeviceToDevice,
                            impl_->stream_.stream());

            // copy to other ranks. UCX has perf issues if self copies are made in MPI_Alltoallv call
            MPI_Alltoallv(impl_->complexGrid_,
                          impl_->sendCountTemp_,
                          impl_->sendDisp_,
                          impl_->complexType_,
                          impl_->complexGrid2_,
                          impl_->recvCountTemp_,
                          impl_->recvDisp_,
                          impl_->complexType_,
                          impl_->mpi_comm_);

#    else
            MPI_Alltoallv(impl_->complexGrid_,
                          impl_->sendCount_,
                          impl_->sendDisp_,
                          impl_->complexType_,
                          impl_->complexGrid2_,
                          impl_->recvCount_,
                          impl_->recvDisp_,
                          impl_->complexType_,
                          impl_->mpi_comm_);
#    endif

            // make data in proper layout once different blocks are received from different MPI ranks
            convertBlockedXyzToXyz(impl_->complexGrid2_,
                                   impl_->complexGrid_,
                                   localx,
                                   impl_->complexGridSizePadded_[YY],
                                   impl_->complexGridSizePadded_[ZZ],
                                   impl_->d_yBlockSizes_,
                                   impl_->d_s2g0y_,
                                   impl_->mpiSize_,
                                   impl_->yMax_,
                                   impl_->stream_);
            // 2D
            result = cufftExecC2R(impl_->planC2R2D_, impl_->complexGrid_, impl_->realGrid_);
            handleCufftError(result, "cuFFT C2R 2D execution failure");
        }
    }
}

Gpu3dFft::Gpu3dFft(const PmeGpu*        pmeGpu,
                   ivec                 realGridSize,
                   ivec                 realGridSizePadded,
                   ivec                 complexGridSize,
                   ivec                 complexGridSizePadded,
                   const bool           useDecomposition,
                   const bool           performOutOfPlaceFFT,
                   const DeviceContext& context,
                   const DeviceStream&  pmeStream,
                   DeviceBuffer<float>  realGrid,
                   DeviceBuffer<float>  complexGrid,
                   DeviceBuffer<float>  complexGrid2) :
    impl_(std::make_unique<Impl>(pmeGpu,
                                 realGridSize,
                                 realGridSizePadded,
                                 complexGridSize,
                                 complexGridSizePadded,
                                 useDecomposition,
                                 performOutOfPlaceFFT,
                                 context,
                                 pmeStream,
                                 realGrid,
                                 complexGrid,
                                 complexGrid2))
{
}

Gpu3dFft::~Gpu3dFft() = default;

} // namespace gmx
