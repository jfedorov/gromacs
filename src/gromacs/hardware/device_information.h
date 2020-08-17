/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018,2019,2020, by the GROMACS development team, led by
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
 *  \brief Declares the GPU type traits for non-GPU builds.
 *
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \inlibraryapi
 * \ingroup module_hardware
 */
#ifndef GMX_HARDWARE_DEVICE_INFORMATION_H
#define GMX_HARDWARE_DEVICE_INFORMATION_H

#include "config.h"

#if GMX_GPU_CUDA
#    include <cuda_runtime.h>
#endif

#if GMX_GPU_OPENCL
#    include "gromacs/gpu_utils/gmxopencl.h"
#endif
#include "gromacs/utility/enumerationhelpers.h"

//! Possible results of the GPU detection/check.
enum class DeviceStatus : int
{
    //! The device is compatible
    Compatible = 0,
    //! Device does not exist
    Nonexistent = 1,
    //! Device is not compatible
    Incompatible = 2,
    //! OpenCL device has incompatible cluster size for non-bonded kernels.
    IncompatibleClusterSize = 3,
    /*! \brief An error occurred he functionality checks.
     * That indicates malfunctioning of the device, driver, or incompatible driver/runtime.
     */
    NonFunctional = 4,
    /*! \brief CUDA devices are busy or unavailable.
     * typically due to use of \p cudaComputeModeExclusive, \p cudaComputeModeProhibited modes.
     */
    Unavailable = 5,
    //! Enumeration size
    Count = 6
};

/*! \brief Names of the GPU detection/check results
 *
 * Check-source wants to warn about the use of a symbol name that would
 * require an inclusion of config.h. However the use is in a comment, so that
 * is a false warning. So C-style string concatenation is used to fool the
 * naive parser in check-source. That needs a clang-format suppression
 * in order to look reasonable. Also clang-tidy wants to suggest that a comma is
 * missing, so that is suppressed.
 */
static const gmx::EnumerationArray<DeviceStatus, const char*> c_deviceStateString = {
    "compatible", "nonexistent", "incompatible",
    // clang-format off
    // NOLINTNEXTLINE(bugprone-suspicious-missing-comma)
    "incompatible (please recompile with correct GMX" "_OPENCL_NB_CLUSTER_SIZE of 4)",
    // clang-format on
    "non-functional", "unavailable"
};

//! Device vendors
enum class DeviceVendor : int
{
    //! No data
    Unknown = 0,
    //! NVIDIA
    Nvidia = 1,
    //! Advanced Micro Devices
    Amd = 2,
    //! Intel
    Intel = 3,
    //! Enumeration size
    Count = 4
};


/*! \brief Platform-dependent device information.
 *
 * The device information is queried and set at detection and contains
 * both information about the device/hardware returned by the runtime as well
 * as additional data like support status.
 */
struct DeviceInformation
{
    //! Device status.
    DeviceStatus status;
    //! ID of the device.
    int id;

#if GMX_GPU_CUDA
    //! CUDA device properties.
    cudaDeviceProp prop;
#elif GMX_GPU_OPENCL
    cl_platform_id oclPlatformId;       //!< OpenCL Platform ID.
    cl_device_id   oclDeviceId;         //!< OpenCL Device ID.
    char           device_name[256];    //!< Device name.
    char           device_version[256]; //!< Device version.
    char           vendorName[256];     //!< Device vendor name.
    int            compute_units;       //!< Number of compute units.
    int            adress_bits;         //!< Number of address bits the device is capable of.
    DeviceVendor   deviceVendor;        //!< Device vendor.
    size_t         maxWorkItemSizes[3]; //!< Workgroup size limits (CL_DEVICE_MAX_WORK_ITEM_SIZES).
    size_t         maxWorkGroupSize;    //!< Workgroup total size limit (CL_DEVICE_MAX_WORK_GROUP_SIZE).
#endif
};

#endif // GMX_HARDWARE_DEVICE_INFORMATION_H