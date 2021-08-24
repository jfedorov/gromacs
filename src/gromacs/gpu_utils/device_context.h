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
#ifndef GMX_GPU_UTILS_DEVICE_CONTEXT_H
#define GMX_GPU_UTILS_DEVICE_CONTEXT_H

/*! \libinternal \file
 *
 * \brief Declarations for DeviceContext class.
 *
 * Only needed for OpenCL builds. Other platforms will be given a stub class.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_gpu_utils
 * \inlibraryapi
 */

#include "config.h"

#include <memory>

#if GMX_GPU_OPENCL
#    include "gromacs/gpu_utils/gmxopencl.h"
#endif
#if GMX_GPU_SYCL
#    include "gromacs/gpu_utils/gmxsycl.h"
#endif

#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/utility/classhelpers.h"

struct DeviceInformation;

// Stub for device context
class DeviceContext
{
public:
    //! Constructor.
    DeviceContext(const DeviceInformation& deviceInfo);
    //! Destructor
    // NOLINTNEXTLINE(performance-trivially-destructible)
    ~DeviceContext();

    //! Get the associated device information
    const DeviceInformation& deviceInfo() const { return deviceInfo_; }

    //! Activate the device associated with the context
    void setDeviceActive() { setActiveDevice(deviceInfo_); }

    /*! \brief Check if this context is currently active (i.e. corresponding device is set in CUDA)
     *
     * In CUDA, switching the current device is done explicitely, using \c cudaSetDevice(..) function.
     * When e.g. the device streams are created, they are attached to the device that is currently set active.
     * For performance reasons, current implementation of the \c DeviceContext in CUDA does not track
     * this attachment. As a result, when the CUDA active device is changed after e.g. the stream was created,
     * this stream should not be used until the active device is explicetely switched back. Since using
     * this stream is still allowed by the API, it is beneficial to have the function that checks if
     * the \c DeviceContext object corresponds to the device that is currently active.
     *
     * These checks are not needed in SYCL and OpenCL, where the device handle (reference to \c DeviceContext
     * object) is passed to the API calls, which redirects the calls to a specific device.
     *
     * \return Whether this context corresponds to an active device.
     */
    bool isDeviceActive() const { return checkDeviceActive(deviceInfo_); }

private:
    //! A reference to the device information used upon context creation
    const DeviceInformation& deviceInfo_;

#if GMX_GPU_OPENCL
public:
    //! Getter
    cl_context context() const;

private:
    //! OpenCL context object
    cl_context context_ = nullptr;
#endif

#if GMX_GPU_SYCL
public:
    //! Const getter
    const cl::sycl::context& context() const { return context_; }
    //! Getter
    cl::sycl::context& context() { return context_; }

private:
    //! SYCL context object
    cl::sycl::context context_;
#endif

    GMX_DISALLOW_COPY_MOVE_AND_ASSIGN(DeviceContext);
};

#endif // GMX_GPU_UTILS_DEVICE_CONTEXT_H
