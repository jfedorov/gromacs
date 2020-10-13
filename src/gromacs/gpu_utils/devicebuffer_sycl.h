/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020, by the GROMACS development team, led by
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
#ifndef GMX_GPU_UTILS_DEVICEBUFFER_SYCL_H
#define GMX_GPU_UTILS_DEVICEBUFFER_SYCL_H

/*! \libinternal \file
 *  \brief Implements the DeviceBuffer type and routines for SYCL.
 *  Should only be included directly by the main DeviceBuffer file devicebuffer.h.
 *  TODO: the intent is for DeviceBuffer to become a class.
 *
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *  \author Erik Lindahl <erik.lindahl@gmail.com>
 *  \author Andrey Alekseenko <al42and@gmail.com>
 *
 *  \inlibraryapi
 */

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/gpu_utils/gpu_utils.h" //only for GpuApiCallBehavior
#include "gromacs/gpu_utils/gputraits_sycl.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#ifndef DOXYGEN
template<typename T>
class DeviceBuffer<T>::ClSyclBufferWrapper : public cl::sycl::buffer<T, 1>
{
    using cl::sycl::buffer<T, 1>::buffer; // Get all the constructors
};

template<typename T>
using ClSyclBufferWrapper = typename DeviceBuffer<T>::ClSyclBufferWrapper;

//! Constructor.
template<typename T>
DeviceBuffer<T>::DeviceBuffer() : buffer_(nullptr)
{
}

//! Destructor.
template<typename T>
DeviceBuffer<T>::~DeviceBuffer() = default;

//! Copy constructor (references the same underlying SYCL buffer).
template<typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer<T> const& src) :
    buffer_(new ClSyclBufferWrapper(*src.buffer_))
{
}

//! Move constructor.
template<typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer<T>&& src) noexcept = default;

//! Copy assignment (references the same underlying SYCL buffer).
template<typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(DeviceBuffer<T> const& src)
{
    buffer_.reset(new ClSyclBufferWrapper(*src.buffer_));
    return *this;
}

//! Move assignment.
template<typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(DeviceBuffer<T>&& src) noexcept = default;

/*! \brief Dummy assignment operator to allow compilation of some cross-platform code.
 *
 * A hacky way to make SYCL implementation of DeviceBuffer compatible with details of CUDA and
 * OpenCL implementations.
 *
 * \todo Should be removed after DeviceBuffer refactoring.
 *
 * \tparam T Type of buffer content.
 * \param nullPtr \c std::nullptr. Not possible to assign any other pointers.
 */
template<typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(std::nullptr_t nullPtr)
{
    buffer_.reset(nullPtr);
    return *this;
}

#endif // #ifndef DOXYGEN

/*! \brief Check the validity of the device buffer.
 *
 * Checks if the buffer is valid and if its allocation is big enough.
 *
 * \param[in] buffer        Device buffer to be checked.
 * \param[in] requiredSize  Number of elements that the buffer will have to accommodate.
 *
 * \returns Whether the device buffer exists and has enough capacity.
 */
template<typename T>
static gmx_unused bool checkDeviceBuffer(const DeviceBuffer<T>& buffer, int requiredSize)
{
    return buffer.buffer_ && (static_cast<int>(buffer.buffer_->get_count()) >= requiredSize);
}

/*! \libinternal \brief
 * Allocates a device-side buffer.
 * It is currently a caller's responsibility to call it only on not-yet allocated buffers.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] buffer               Pointer to the device-side buffer.
 * \param[in]     numValues            Number of values to accommodate.
 * \param[in]     deviceContext        The buffer's device context-to-be.
 */
template<typename ValueType>
void allocateDeviceBuffer(DeviceBuffer<ValueType>* buffer, size_t numValues, const DeviceContext& deviceContext)
{
    /* SYCL does not require binding buffer to a specific context or device. The ::context_bound
     * property only enforces the use of only given context, and possibly offers some optimizations */
    const cl::sycl::property_list bufferProperties{ cl::sycl::property::buffer::context_bound(
            deviceContext.context()) };
    buffer->buffer_.reset(
            new ClSyclBufferWrapper<ValueType>(cl::sycl::range<1>(numValues), bufferProperties));
}

/*! \brief
 * Frees a device-side buffer.
 * This does not reset separately stored size/capacity integers,
 * as this is planned to be a destructor of DeviceBuffer as a proper class,
 * and no calls on \p buffer should be made afterwards.
 *
 * \param[in] buffer  Pointer to the buffer to free.
 */
template<typename ValueType>
void freeDeviceBuffer(DeviceBuffer<ValueType>* buffer)
{
    buffer->buffer_.reset(nullptr);
}

/*! \brief
 * Performs the host-to-device data copy, synchronous or asynchronously on request.
 *
 * Unlike in CUDA and OpenCL, synchronous call does not guarantee that all previously
 * submitted operations are complete, only the ones that are required for \p buffer consistency.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] buffer               Pointer to the device-side buffer.
 * \param[in]     hostBuffer           Pointer to the raw host-side memory, also typed \p ValueType.
 * \param[in]     startingOffset       Offset (in values) at the device-side buffer to copy into.
 * \param[in]     numValues            Number of values to copy.
 * \param[in]     deviceStream         GPU stream to perform asynchronous copy in.
 * \param[in]     transferKind         Copy type: synchronous or asynchronous.
 * \param[out]    timingEvent          A pointer to the H2D copy timing event to be filled in.
 *                                     Ignored in SYCL.
 */
template<typename ValueType>
void copyToDeviceBuffer(DeviceBuffer<ValueType>* buffer,
                        const ValueType*         hostBuffer,
                        size_t                   startingOffset,
                        size_t                   numValues,
                        const DeviceStream&      deviceStream,
                        GpuApiCallBehavior       transferKind,
                        CommandEvent* gmx_unused timingEvent)
{
    if (numValues == 0)
    {
        return; // such calls are actually made with empty domains
    }
    GMX_ASSERT(buffer, "needs a buffer pointer");
    GMX_ASSERT(hostBuffer, "needs a host buffer pointer");

    GMX_ASSERT(checkDeviceBuffer(*buffer, startingOffset + numValues),
               "buffer too small or not initialized");

    cl::sycl::buffer<ValueType>& syclBuffer = *buffer->buffer_;

    cl::sycl::event ev = deviceStream.stream().submit([&](cl::sycl::handler& cgh) {
        /* Here and elsewhere in this file, accessor constructor is user instead of a more common
         * buffer::get_access, since the compiler (icpx 2021.1-beta09) occasionally gets confused
         * by all the overloads */
        auto d_bufferAccessor = cl::sycl::accessor<ValueType, 1, cl::sycl::access::mode::discard_write>{
            syclBuffer, cgh, cl::sycl::range(numValues), cl::sycl::id(startingOffset)
        };
        cgh.copy(hostBuffer, d_bufferAccessor);
    });
    if (transferKind == GpuApiCallBehavior::Sync)
    {
        ev.wait_and_throw();
    }
}

/*! \brief
 * Performs the device-to-host data copy, synchronous or asynchronously on request.
 *
 * Unlike in CUDA and OpenCL, synchronous call does not guarantee that all previously
 * submitted operations are complete, only the ones that are required for \p buffer consistency.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] hostBuffer           Pointer to the raw host-side memory, also typed \p ValueType
 * \param[in]     buffer               Pointer to the device-side buffer.
 * \param[in]     startingOffset       Offset (in values) at the device-side buffer to copy from.
 * \param[in]     numValues            Number of values to copy.
 * \param[in]     deviceStream         GPU stream to perform asynchronous copy in.
 * \param[in]     transferKind         Copy type: synchronous or asynchronous.
 * \param[out]    timingEvent          A pointer to the H2D copy timing event to be filled in.
 *                                     Ignored in SYCL.
 */
template<typename ValueType>
void copyFromDeviceBuffer(ValueType*               hostBuffer,
                          DeviceBuffer<ValueType>* buffer,
                          size_t                   startingOffset,
                          size_t                   numValues,
                          const DeviceStream&      deviceStream,
                          GpuApiCallBehavior       transferKind,
                          CommandEvent* gmx_unused timingEvent)
{
    if (numValues == 0)
    {
        return; // such calls are actually made with empty domains
    }
    GMX_ASSERT(buffer, "needs a buffer pointer");
    GMX_ASSERT(hostBuffer, "needs a host buffer pointer");

    GMX_ASSERT(checkDeviceBuffer(*buffer, startingOffset + numValues),
               "buffer too small or not initialized");

    cl::sycl::buffer<ValueType>& syclBuffer = *buffer->buffer_;

    cl::sycl::event ev = deviceStream.stream().submit([&](cl::sycl::handler& cgh) {
        const auto d_bufferAccessor = cl::sycl::accessor<ValueType, 1, cl::sycl::access::mode::read>{
            syclBuffer, cgh, cl::sycl::range(numValues), cl::sycl::id(startingOffset)
        };
        cgh.copy(d_bufferAccessor, hostBuffer);
    });
    if (transferKind == GpuApiCallBehavior::Sync)
    {
        ev.wait_and_throw();
    }
}

/*! \brief
 * Clears the device buffer asynchronously.
 *
 * \tparam        ValueType       Raw value type of the \p buffer.
 * \param[in,out] buffer          Pointer to the device-side buffer.
 * \param[in]     startingOffset  Offset (in values) at the device-side buffer to start clearing at.
 * \param[in]     numValues       Number of values to clear.
 * \param[in]     deviceStream    GPU stream.
 */
template<typename ValueType>
void clearDeviceBufferAsync(DeviceBuffer<ValueType>* buffer,
                            size_t                   startingOffset,
                            size_t                   numValues,
                            const DeviceStream&      deviceStream)
{
    if (numValues == 0)
    {
        return;
    }
    GMX_ASSERT(buffer, "needs a buffer pointer");

    GMX_ASSERT(checkDeviceBuffer(*buffer, startingOffset + numValues),
               "buffer too small or not initialized");

    const ValueType              pattern{};
    cl::sycl::buffer<ValueType>& syclBuffer = *(buffer->buffer_);

    cl::sycl::event ev = deviceStream.stream().submit([&](cl::sycl::handler& cgh) {
        auto d_bufferAccessor = cl::sycl::accessor<ValueType, 1, cl::sycl::access::mode::discard_write>{
            syclBuffer, cgh, cl::sycl::range(numValues), cl::sycl::id(startingOffset)
        };
        cgh.fill(d_bufferAccessor, pattern);
    });
}

/*! \brief Create a texture object for an array of type ValueType.
 *
 * Creates the device buffer and copies read-only data for an array of type ValueType.
 * Like OpenCL, does not really do anything with textures, simply creates a buffer
 * and initializes it.
 *
 * \tparam      ValueType      Raw data type.
 *
 * \param[out]  deviceBuffer   Device buffer to store data in.
 * \param[in]   hostBuffer     Host buffer to get date from.
 * \param[in]   numValues      Number of elements in the buffer.
 * \param[in]   deviceContext  GPU device context.
 */
template<typename ValueType>
void initParamLookupTable(DeviceBuffer<ValueType>* deviceBuffer,
                          DeviceTexture* /* deviceTexture */,
                          const ValueType*     hostBuffer,
                          int                  numValues,
                          const DeviceContext& deviceContext)
{
    GMX_ASSERT(hostBuffer, "Host buffer should be specified.");
    GMX_ASSERT(deviceBuffer, "Device buffer should be specified.");

    /* Constructing buffer with cl::sycl::buffer(T* data, size_t size) will take ownership
     * of this memory region making it unusable, which might lead to side-effects.
     * On the other hand, cl::sycl::buffer(InputIterator<T> begin, InputIterator<T> end) will
     * initialize the buffer without affecting ownership of the memory, although
     * it will consume extra memory on host. */
    const cl::sycl::property_list bufferProperties{ cl::sycl::property::buffer::context_bound(
            deviceContext.context()) };
    deviceBuffer->buffer_.reset(new ClSyclBufferWrapper<ValueType>(
            hostBuffer, hostBuffer + numValues, bufferProperties));
}

/*! \brief Release the OpenCL device buffer.
 *
 * \tparam        ValueType     Raw data type.
 *
 * \param[in,out] deviceBuffer  Device buffer to store data in.
 */
template<typename ValueType>
void destroyParamLookupTable(DeviceBuffer<ValueType>* deviceBuffer, DeviceTexture& /* deviceTexture */)
{
    deviceBuffer->buffer_.reset(nullptr);
}

/*! \brief Wrapper around cl::sycl::accessor.
 *
 * Compared to the cl::sycl::accessor, this class has following benefits:
 *
 * - Can be constructed from the \c DeviceBuffer (or pointer to it) directly.
 * - Can be made not to create any actual accessors by setting \p isDummy to \c true.
 *
 * Passing accessors to SYCL kernels requires storing them as functor's members. Since SYCL's
 * accessor has no trivial constructor and does not like getting invalid buffers, having
 * cl::sycl::accessor's as functor members would require always initializing them, even
 * if they are not used (can happen if the set of kernel arguments depend on the parameters).
 * Setting up an accessor to a real buffer would introduce unnecessary data dependencies.
 * Creating a dummy buffer just to get it's accessor is both a runtime overhead and complicates
 * the code. For dummy buffer, one can safely pass \c nullptr instead of a buffer.
 *
 * \tparam T Type of data stored in the buffer.
 * \tparam mode Buffer access mode (from \c cl::sycl::access::mode)
 * \tparam isDummy If set to \p true, a dummy struct would be generated, not associated with any
 *                 actual SYCL accessor and not intended to be actually used in kernel code.
 */
template<typename T, enum cl::sycl::access::mode mode, bool isDummy = false>
struct DeviceAccessor
{
    using ReturnType = typename DeviceAccessor<T, mode, true>::ReturnType;
    DeviceAccessor(cl::sycl::buffer<T, 1>* buf, cl::sycl::handler& cgh) :
        accessor_(cl::sycl::accessor<T, 1, mode>(*buf, cgh))
    {
    }
    DeviceAccessor(DeviceBuffer<T>& buf, cl::sycl::handler& cgh) :
        accessor_(cl::sycl::accessor<T, 1, mode>(*buf.buffer_.get(), cgh))
    {
    }
    template<typename IndexingType>
    inline ReturnType operator[](IndexingType index) const
    {
        return accessor_[index];
    }
    cl::sycl::accessor<T, 1, mode> accessor_;
};

/*! \brief Template instantiation of DeviceAccessor for cases when we don't need an accessor.
 *
 * Helpful if we want to get an accessor in some instantiations of kernel functor, but not
 * in others.
 *
 * \warning While this class provides `operator[]`, it is not intended to be used.
 *
 */
template<typename T, enum cl::sycl::access::mode mode>
struct DeviceAccessor<T, mode, true>
{
    using ReturnType = std::conditional_t<mode == cl::sycl::access::mode::read, T, T&>;
    DeviceAccessor(cl::sycl::buffer<T, 1>* /*buf*/, cl::sycl::handler& /*cgh*/) : value_() {}
    DeviceAccessor(DeviceBuffer<T>& /*buf*/, cl::sycl::handler& /*cgh*/) : value_() {}
    template<typename IndexingType>
    inline ReturnType operator[](IndexingType /*index*/) const
    {
        return value_;
    }
    T value_;
};


#endif // GMX_GPU_UTILS_DEVICEBUFFER_SYCL_H
