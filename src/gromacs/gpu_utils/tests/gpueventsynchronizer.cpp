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
 * \brief Tests for GpuEventSynchronizer
 *
 * \author Andrey Alekseenko <al42and@gmail.com>
 *
 * \ingroup module_gpu_utils
 */
#include "gmxpre.h"

#include "gromacs/gpu_utils/gpueventsynchronizer.h"

#include <gtest/gtest.h>

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/hardware/device_management.h"

#include "testutils/test_hardware_environment.h"

namespace gmx
{

namespace test
{

namespace
{

#if GMX_GPU
TEST(GpuEventSynchronizerTest, BasicFunctionality)
{
    const auto& testDeviceList = getTestHardwareEnvironment()->getTestDeviceList();
    for (const auto& testDevice : testDeviceList)
    {
        const DeviceInformation& deviceInfo = testDevice->deviceInfo();
        setActiveDevice(deviceInfo);
        DeviceContext context(deviceInfo);
        DeviceStream  streamA(context, DeviceStreamPriority::Normal, false);
        DeviceStream  streamB(context, DeviceStreamPriority::Normal, false);

        {
            SCOPED_TRACE("Constructor");
            GpuEventSynchronizer gpuEventSynchronizer;
        }

        {
            SCOPED_TRACE("Mark and wait");
            GpuEventSynchronizer gpuEventSynchronizer;
            gpuEventSynchronizer.markEvent(streamA);
            gpuEventSynchronizer.waitForEvent(); // Should return immediately
        }

        {
            SCOPED_TRACE("Mark and enqueueWait");
            GpuEventSynchronizer gpuEventSynchronizer;
            gpuEventSynchronizer.markEvent(streamA);
            gpuEventSynchronizer.enqueueWaitEvent(streamB);
            streamB.synchronize(); // Should return immediately
        }

        {
            SCOPED_TRACE("Mark and wait twice");
            GpuEventSynchronizer gpuEventSynchronizer;
            gpuEventSynchronizer.markEvent(streamA);
            gpuEventSynchronizer.waitForEvent();
            gpuEventSynchronizer.markEvent(streamB);
            gpuEventSynchronizer.waitForEvent();
        }

#    if !GMX_GPU_CUDA // CUDA has very lax rules for event consumption. See Issues #2527 and #3988.
        {
            SCOPED_TRACE("Wait before marking");
            GpuEventSynchronizer gpuEventSynchronizer;
            EXPECT_THROW(gpuEventSynchronizer.waitForEvent(), gmx::InternalError);
        }
        {
            SCOPED_TRACE("enqueueWait before marking");
            GpuEventSynchronizer gpuEventSynchronizer;
            EXPECT_THROW(gpuEventSynchronizer.enqueueWaitEvent(streamA), gmx::InternalError);
        }
        {
            SCOPED_TRACE("Wait twice after marking");
            GpuEventSynchronizer gpuEventSynchronizer;
            gpuEventSynchronizer.markEvent(streamA);
            gpuEventSynchronizer.waitForEvent();
            EXPECT_THROW(gpuEventSynchronizer.waitForEvent(), gmx::InternalError);
        }
#    endif
    }
}
#endif // GMX_GPU

} // namespace
} // namespace test
} // namespace gmx
