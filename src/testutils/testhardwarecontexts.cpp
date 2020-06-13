/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2017,2018,2019,2020, by the GROMACS development team, led by
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
 * \brief
 * Implements test environment class which performs hardware enumeration for unit tests.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \ingroup module_ewald
 */

#include "gmxpre.h"

#include "testhardwarecontexts.h"

#include "config.h"

#include <memory>

#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/hardware/detecthardware.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/loggerbuilder.h"
#include "gromacs/utility/physicalnodecommunicator.h"

namespace gmx
{
namespace test
{

/* Implements the "construct on first use" idiom to avoid any static
 * initialization order fiasco.
 *
 * Note that thread-safety of the initialization is guaranteed by the
 * C++11 language standard.
 *
 * The pointer itself (not the memory it points to) has no destructor,
 * so there is no deinitialization issue.  See
 * https://isocpp.org/wiki/faq/ctors for discussion of alternatives
 * and trade-offs. */
const TestHardwareEnvironment* getTestHardwareEnvironment()
{
    static TestHardwareEnvironment* testHardwareEnvironment = nullptr;
    if (testHardwareEnvironment == nullptr)
    {
        printf("Initializing the test hardware environment!\n");
        // Ownership of the TestEnvironment is taken by GoogleTest, so nothing can leak
        testHardwareEnvironment = static_cast<TestHardwareEnvironment*>(
                ::testing::AddGlobalTestEnvironment(new TestHardwareEnvironment));
    }
    return testHardwareEnvironment;
}

void callAddGlobalTestEnvironment()
{
    getTestHardwareEnvironment();
}

//! Simple hardware initialization
static gmx_hw_info_t* hardwareInit()
{
    PhysicalNodeCommunicator physicalNodeComm(MPI_COMM_WORLD, gmx_physicalnode_id_hash());
    return gmx_detect_hardware(MDLogger{}, physicalNodeComm);
}


// These two functions should not be here
static bool deviceBuildIsSupported()
{
    if (GMX_DOUBLE)
    {
        return false;
    }
    if (GMX_GPU == GMX_GPU_NONE)
    {
        return false;
    }
    return true;
}

static bool deviceIsSupported()
{
    if (GMX_GPU == GMX_GPU_OPENCL)
    {
#ifdef __APPLE__
        return false;
#endif
    }
    return true;
}

void TestHardwareEnvironment::SetUp()
{
    hardwareContexts_.emplace_back(std::make_unique<TestHardwareContext>("CPU"));
    hardwareInfo_ = hardwareInit();
    if (!deviceBuildIsSupported() || !deviceIsSupported())
    {
        // The build of hardware do not support the device runs
        return;
    }
    // Constructing contexts for all compatible GPUs - will not add anything for non-GPU builds
    for (int gpuIndex : getCompatibleGpus(hardwareInfo_->gpu_info))
    {
        const DeviceInformation* deviceInfo = getDeviceInfo(hardwareInfo_->gpu_info, gpuIndex);
        init_gpu(deviceInfo);

        char stmp[200] = {};
        get_gpu_device_info_string(stmp, hardwareInfo_->gpu_info, gpuIndex);
        std::string description = "(GPU " + std::string(stmp) + ") ";
        hardwareContexts_.emplace_back(
                std::make_unique<TestHardwareContext>(description.c_str(), *deviceInfo));
    }
}

void TestHardwareEnvironment::TearDown()
{
    for (auto& context : hardwareContexts_)
    {
        context.reset(nullptr);
    }
    hardwareContexts_.clear();
}

} // namespace test
} // namespace gmx
