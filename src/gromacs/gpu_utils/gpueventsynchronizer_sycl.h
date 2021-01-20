/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020,2021, by the GROMACS development team, led by
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
/*! \libinternal \file
 *  \brief Implements a GpuEventSynchronizer class for SYCL.
 *
 *  This implementation relies on SYCL_INTEL_enqueue_barrier proposal,
 *  https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/EnqueueBarrier/enqueue_barrier.asciidoc
 *
 *  Using event-based synchronization is not recommended for SYCL.
 *  SYCL queues are out-of-order and rely on data dependencies, allowing only to wait
 *  for a specific kernel (by capturing the \c event returned from \c queue.submit) or for all
 *  the tasks in the queue (\c queue.wait).
 *
 *  \author Erik Lindahl <erik.lindahl@gmail.com>
 *  \author Andrey Alekseenko <al42and@gmail.com>
 * \inlibraryapi
 */
#ifndef GMX_GPU_UTILS_GPUEVENTSYNCHRONIZER_SYCL_H
#define GMX_GPU_UTILS_GPUEVENTSYNCHRONIZER_SYCL_H

#include <optional>

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"

#ifndef DOXYGEN
/*! \libinternal \brief
 * A class which allows for CPU thread to mark and wait for certain GPU stream execution point.
 * The event can be put into the stream with markEvent() and then later waited on with waitForEvent().
 * This can be repeated as necessary, but the current implementation does not allow waiting on
 * completed event more than once, expecting only exact pairs of markEvent(stream); waitForEvent().
 * The class generally attempts to track the correctness of its state transitions, but
 * please note that calling waitForEvent() right after the construction will fail with OpenCL
 * and SYCL but succeed with CUDA.
 *
 * Another possible mode of operation can be implemented if needed:
 * multiple calls to waitForEvent() after a single markEvent(). For this, event.reset() call
 * from waitForEvent() should instead happen conditionally at the beginning of markEvent(), replacing
 * the GMX_ASSERT(). This was tested to work both with CUDA, NVidia OpenCL, and Intel SYCL,
 * but not with AMD/Intel OpenCl.
 *
 *  \warning This class is offered for uniformity with other GPU implementations, but expect it to
 *  be deprecated in the future.
 *
 */
class GpuEventSynchronizer
{
public:
    //! A constructor.
    GpuEventSynchronizer() = default;
    //! A constructor from an existing event.
    GpuEventSynchronizer(const cl::sycl::event& event) : event_(event) {}
    //! A destructor.
    ~GpuEventSynchronizer() = default;
    //! No copying
    GpuEventSynchronizer(const GpuEventSynchronizer&) = delete;
    //! No assignment
    GpuEventSynchronizer& operator=(GpuEventSynchronizer&&) = delete;
    //! Moving is disabled but can be considered in the future if needed
    GpuEventSynchronizer(GpuEventSynchronizer&&) = delete;

    /*! \brief Marks the synchronization point in the \p deviceStream.
     * Should be called first and then followed by waitForEvent() or enqueueWaitEvent().
     */
    inline void markEvent(const DeviceStream& deviceStream)
    {
        GMX_ASSERT(!event_.has_value(), "Do not call markEvent more than once!");
#    ifdef __HIPSYCL__
        deviceStream.stream().wait_and_throw(); // SYCL-TODO: Use CUDA/HIP-specific solutions
#    else
        // Relies on SYCL_INTEL_enqueue_barrier
        event_ = deviceStream.stream().submit_barrier();
#    endif
    }
    /*! \brief Synchronizes the host thread on the marked event.
     * As in the OpenCL implementation, the event is released.
     */
    inline void waitForEvent()
    {
#    ifndef __HIPSYCL__
        event_->wait_and_throw();
#    endif
        event_.reset();
    }
    /*! \brief Enqueues a wait for the recorded event in stream \p deviceStream.
     * As in the OpenCL implementation, the event is released.
     */
    inline void enqueueWaitEvent(const DeviceStream& deviceStream)
    {
#    ifdef __HIPSYCL__
        deviceStream.stream().wait_and_throw(); // SYCL-TODO: Use CUDA/HIP-specific solutions
#    else
        // Relies on SYCL_INTEL_enqueue_barrier
        const std::vector<cl::sycl::event> waitlist{ event_.value() };
        deviceStream.stream().submit_barrier(waitlist);
        event_.reset();
#    endif
    }

private:
    std::optional<cl::sycl::event> event_ = std::nullopt;
};

#endif // !defined DOXYGEN

#endif // GMX_GPU_UTILS_GPUEVENTSYNCHRONIZER_SYCL_H
