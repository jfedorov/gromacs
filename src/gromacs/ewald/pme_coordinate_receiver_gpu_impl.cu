/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
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
 *
 * \brief Implements class which recieves coordinates to GPU memory on PME task using CUDA
 *
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "gromacs/ewald/pme_pp_communication.h"
#include "pme_coordinate_receiver_gpu_impl.h"

#include "config.h"

#include "gromacs/ewald/pme_force_sender_gpu.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/utility/gmxmpi.h"

namespace gmx
{

PmeCoordinateReceiverGpu::Impl::Impl(MPI_Comm               comm,
                                     const DeviceContext&   deviceContext,
                                     gmx::ArrayRef<PpRanks> ppRanks) :
    comm_(comm), ppRanks_(ppRanks), deviceContext_(deviceContext)
{
    request_.resize(ppRanks.size());
    ppSync_.resize(ppRanks.size());

    // Create streams to manage pipelining
    DeviceStream* stream;
    size_t        i = 0;
    while (i < ppRanks_.size())
    {
        stream = new DeviceStream(deviceContext_, DeviceStreamPriority::High, false);
        ppCommStream_.push_back(std::make_unique<DeviceStream>(deviceContext_, DeviceStreamPriority::High, false));
        i++;
    }
}

PmeCoordinateReceiverGpu::Impl::~Impl() = default;

void PmeCoordinateReceiverGpu::Impl::reinitCoordinateReceiver(DeviceBuffer<RVec> d_x)
{
    atomStart_.resize(ppRanks_.size());
    atomEnd_.resize(ppRanks_.size());
    int ind_start = 0;
    int ind_end   = 0;
    int i         = 0;
    for (const auto& receiver : ppRanks_)
    {
        ind_start = ind_end;
        ind_end   = ind_start + receiver.numAtoms;

        atomStart_[i] = ind_start;
        atomEnd_[i]   = ind_end;

        // Need to send address to PP rank only for thread-MPI as PP rank pushes data using cudamemcpy
        if (GMX_THREAD_MPI)
        {
            // Data will be transferred directly from GPU.
            void* sendBuf = reinterpret_cast<void*>(&d_x[ind_start]);
#if GMX_MPI
            MPI_Send(&sendBuf, sizeof(void**), MPI_BYTE, receiver.rankId, 0, comm_);
#else
            GMX_UNUSED_VALUE(sendBuf);
#endif
        }
        i++;
    }
}

/*! \brief Receive coordinate synchronizer pointer from the PP ranks. */
void PmeCoordinateReceiverGpu::Impl::receiveCoordinatesSynchronizerFromPpCudaDirect(int ppRank)
{
    GMX_ASSERT(GMX_THREAD_MPI,
               "receiveCoordinatesSynchronizerFromPpCudaDirect is expected to be called only for "
               "Thread-MPI");

    // Data will be pushed directly from PP task

#if GMX_MPI
    // Receive event from PP task
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    MPI_Irecv(&ppSync_[ppRank], sizeof(GpuEventSynchronizer*), MPI_BYTE, ppRank, 0, comm_, &request_[ppRank]);
#else
    GMX_UNUSED_VALUE(ppRank);
#endif
}

/*! \brief Receive coordinate data using CUDA-aware MPI */
void PmeCoordinateReceiverGpu::Impl::launchReceiveCoordinatesFromPpCudaMpi(DeviceBuffer<RVec> recvbuf,
                                                                           int numAtoms,
                                                                           int numBytes,
                                                                           int ppRank)
{
    GMX_ASSERT(GMX_LIB_MPI,
               "launchReceiveCoordinatesFromPpCudaMpi is expected to be called only for Lib-MPI");

#if GMX_MPI
    MPI_Irecv(&recvbuf[numAtoms], numBytes, MPI_BYTE, ppRank, eCommType_COORD_GPU, comm_, &request_[ppRank]);
#else
    GMX_UNUSED_VALUE(recvbuf);
    GMX_UNUSED_VALUE(numAtoms);
    GMX_UNUSED_VALUE(numBytes);
    GMX_UNUSED_VALUE(ppRank);
#endif
}

int PmeCoordinateReceiverGpu::Impl::synchronizeOnCoordinatesFromPpRanks(int pipelineStage,
                                                                        const DeviceStream& deviceStream)
{
#if GMX_MPI
    int senderRank = -1; // Rank of PP task that is associated with this invocation.
#    if (!GMX_THREAD_MPI)
    // Wait on data from any one of the PP sender GPUs
    MPI_Waitany(request_.size(), request_.data(), &senderRank, MPI_STATUS_IGNORE);
    GMX_ASSERT(senderRank >= 0, "Rank of sending PP task must be 0 or greater");
    GMX_UNUSED_VALUE(pipelineStage);
    GMX_UNUSED_VALUE(deviceStream);
#    else
    // MPI_Waitany is not available in thread-MPI. However, the
    // MPI_Wait here is not associated with data but is host-side
    // scheduling code to receive a CUDA event, and will be executed
    // in advance of the actual data transfer. Therefore we can
    // receive in order of pipeline stage, still allowing the
    // scheduled GPU-direct comms to initiate out-of-order in their
    // respective streams. For cases with CPU force computations, the
    // scheduling is less asynchronous (done on a per-step basis), so
    // host-side improvements should be investigated as tracked in
    // issue #4047
    senderRank = pipelineStage;
    MPI_Wait(&request_[senderRank], MPI_STATUS_IGNORE);
    ppSync_[senderRank]->enqueueWaitEvent(deviceStream);
#    endif
    return senderRank;
#endif
}

DeviceStream* PmeCoordinateReceiverGpu::Impl::ppCommStream(int senderIndex)
{
    return ppCommStream_[senderIndex];
}

std::tuple<int, int> PmeCoordinateReceiverGpu::Impl::ppCommAtomRange(int senderIndex)
{
    return std::make_tuple(atomStart_[senderIndex], atomEnd_[senderIndex]);
}

int PmeCoordinateReceiverGpu::Impl::ppCommNumSenderRanks()
{
    return ppRanks_.size();
}

PmeCoordinateReceiverGpu::PmeCoordinateReceiverGpu(MPI_Comm               comm,
                                                   const DeviceContext&   deviceContext,
                                                   gmx::ArrayRef<PpRanks> ppRanks) :
    impl_(new Impl(comm, deviceContext, ppRanks))
{
}

PmeCoordinateReceiverGpu::~PmeCoordinateReceiverGpu() = default;

void PmeCoordinateReceiverGpu::reinitCoordinateReceiver(DeviceBuffer<RVec> d_x)
{
    impl_->reinitCoordinateReceiver(d_x);
}

void PmeCoordinateReceiverGpu::receiveCoordinatesSynchronizerFromPpCudaDirect(int ppRank)
{
    impl_->receiveCoordinatesSynchronizerFromPpCudaDirect(ppRank);
}

void PmeCoordinateReceiverGpu::launchReceiveCoordinatesFromPpCudaMpi(DeviceBuffer<RVec> recvbuf,
                                                                     int                numAtoms,
                                                                     int                numBytes,
                                                                     int                ppRank)
{
    impl_->launchReceiveCoordinatesFromPpCudaMpi(recvbuf, numAtoms, numBytes, ppRank);
}

int PmeCoordinateReceiverGpu::synchronizeOnCoordinatesFromPpRanks(int                 senderIndex,
                                                                  const DeviceStream& deviceStream)
{
    return impl_->synchronizeOnCoordinatesFromPpRanks(senderIndex, deviceStream);
}

DeviceStream* PmeCoordinateReceiverGpu::ppCommStream(int senderIndex)
{
    return impl_->ppCommStream(senderIndex);
}

std::tuple<int, int> PmeCoordinateReceiverGpu::ppCommAtomRange(int senderIndex)
{
    return impl_->ppCommAtomRange(senderIndex);
}

int PmeCoordinateReceiverGpu::ppCommNumSenderRanks()
{
    return impl_->ppCommNumSenderRanks();
}


} // namespace gmx
