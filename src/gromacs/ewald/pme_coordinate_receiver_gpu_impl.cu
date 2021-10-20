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

#include "pme_coordinate_receiver_gpu_impl.h"

#include "config.h"

#include "gromacs/ewald/pme_force_sender_gpu.h"
#include "gromacs/ewald/pme_pp_communication.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/utility/gmxmpi.h"

namespace gmx
{

PmeCoordinateReceiverGpu::Impl::Impl(MPI_Comm                     comm,
                                     const DeviceContext&         deviceContext,
                                     gmx::ArrayRef<const PpRanks> ppRanks) :
    comm_(comm), requests_(ppRanks.size(), MPI_REQUEST_NULL), deviceContext_(deviceContext)
{
    // Create streams to manage pipelining
    ppCommManagers_.reserve(ppRanks.size());
    for (auto& ppRank : ppRanks)
    {
        ppCommManagers_.emplace_back(PpCommManager{
                ppRank,
                std::make_unique<DeviceStream>(deviceContext_, DeviceStreamPriority::High, false),
                nullptr,
                { 0, 0 } });
    }
}

PmeCoordinateReceiverGpu::Impl::~Impl() = default;

void PmeCoordinateReceiverGpu::Impl::reinitCoordinateReceiver(DeviceBuffer<RVec> d_x)
{
    int indEnd = 0;
    for (auto& ppCommManager : ppCommManagers_)
    {
        int indStart = indEnd;
        indEnd       = indStart + ppCommManager.ppRank.numAtoms;

        ppCommManager.atomRange = std::make_tuple(indStart, indEnd);

        // Need to send address to PP rank only for thread-MPI as PP rank pushes data using cudamemcpy
        if (GMX_THREAD_MPI)
        {
            // Data will be transferred directly from GPU.
            void* sendBuf = reinterpret_cast<void*>(&d_x[indStart]);
#if GMX_MPI
            MPI_Send(&sendBuf, sizeof(void**), MPI_BYTE, ppCommManager.ppRank.rankId, 0, comm_);
#else
            GMX_UNUSED_VALUE(sendBuf);
#endif
        }
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
    MPI_Irecv(&ppCommManagers_[ppRank].sync,
              sizeof(GpuEventSynchronizer*),
              MPI_BYTE,
              ppRank,
              0,
              comm_,
              &(requests_[ppRank]));
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
    MPI_Irecv(&recvbuf[numAtoms], numBytes, MPI_BYTE, ppRank, eCommType_COORD_GPU, comm_, &(requests_[ppRank]));
#else
    GMX_UNUSED_VALUE(recvbuf);
    GMX_UNUSED_VALUE(numAtoms);
    GMX_UNUSED_VALUE(numBytes);
    GMX_UNUSED_VALUE(ppRank);
#endif
}

std::pair<int,int> PmeCoordinateReceiverGpu::Impl::prepareForSpread(const bool          canPipelineReceives,
                                                                    const DeviceStream& pmeStream)
{
    const bool usePipeline = canPipelineReceives && ppCommManagers_.size() > 1;
    if (usePipeline)
    {
        return {0, ppCommManagers_.size()};
    }
    // Either we can't pipeline the receives, or there's only one of them
#if GMX_MPI
    MPI_Waitall(requests_.size(), requests_.data(), MPI_STATUSES_IGNORE);
#endif
#if GMX_THREAD_MPI
    for (const auto& ppCommManager : ppCommManagers_)
    {
        ppCommManager.sync->enqueueWaitEvent(pmeStream);
    }
#else
    GMX_UNUSED_VALUE(pmeStream);
#endif
    return {0, 1};
}

PipelinedSpreadManager PmeCoordinateReceiverGpu::Impl::synchronizeOnCoordinatesFromAPpRank(int senderRank)
{
    PipelinedSpreadManager manager;

    GMX_ASSERT(ppCommManagers_.size() > 1, "Multiple PP ranks are required for pipelining");
#if GMX_MPI
    // Ignore the rank passed in
    senderRank = -1; // Rank of PP task that is associated with this invocation.
#    if GMX_LIB_MPI
    // Wait on data from any one of the PP sender GPUs
    MPI_Waitany(requests_.size(), requests_.data(), &senderRank, MPI_STATUS_IGNORE);
    GMX_ASSERT(senderRank >= 0, "Rank of sending PP task must be 0 or greater");
    manager.launchStream = ppCommManagers_[senderRank].stream.get();
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
    MPI_Wait(&requests_[senderRank], MPI_STATUS_IGNORE);
    // Prepare to launch the spread kernel in the stream of the
    // rank whose coordinates have arrived, and synchronize
    // appropriately.
    manager.launchStream = ppCommManagers_[senderRank].stream.get();
    ppCommManagers_[senderRank].sync->enqueueWaitEvent(*manager.launchStream);
#    endif
    manager.atomStart = std::get<0>(ppCommManagers_[senderRank].atomRange);
    manager.atomEnd   = std::get<1>(ppCommManagers_[senderRank].atomRange);
#endif
    return manager;
}

void PmeCoordinateReceiverGpu::Impl::addPipelineDependencies(const DeviceStream& pmeStream)
{
    GMX_ASSERT(ppCommManagers_.size() > 1, "Multiple PP ranks are required for pipelining");
    for (const auto& ppCommManager : ppCommManagers_)
    {
        GpuEventSynchronizer event;
        event.markEvent(*ppCommManager.stream);
        event.enqueueWaitEvent(pmeStream);
    }
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

std::pair<int,int> PmeCoordinateReceiverGpu::prepareForSpread(const bool canPipelineReceives, const DeviceStream& pmeStream)
{
    return impl_->prepareForSpread(canPipelineReceives, pmeStream);
}

PipelinedSpreadManager PmeCoordinateReceiverGpu::synchronizeOnCoordinatesFromAPpRank(int senderRank)
{
    return impl_->synchronizeOnCoordinatesFromAPpRank(senderRank);
}

void PmeCoordinateReceiverGpu::addPipelineDependencies(const DeviceStream& pmeStream)
{
    impl_->addPipelineDependencies(pmeStream);
}

} // namespace gmx
