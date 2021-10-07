/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018,2019,2020,2021, by the GROMACS development team, led by
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

/*! \file
 * \brief Tools for converting simulation input data to and from TPR files.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 * \ingroup gmxapi_compat
 */

#ifndef GMXAPICOMPAT_TPR_H
#define GMXAPICOMPAT_TPR_H

#include <memory>
#include <vector>

#include "gmxapi/gmxapicompat.h"
#include "gmxapi/compat/mdparams.h"

namespace gmxapicompat
{

/*!
 * \brief Manager for TPR file resources.
 *
 * To avoid copies, this resource-owning object is shared by consumers of its
 * resources, even when different resources are consumed.
 *
 * Multiple read-only handles may be issued if there are no write-handles.
 * One write handle may be issued if there are no other open handles.
 *
 * A const TprFile may only issue read file-handles, allowing handles to be
 * issued more quickly by avoiding atomic resource locking.
 *
 * \note Shared ownership of file manager could be avoided if owned by a Context.
 * It is appropriate for a Context to own and mediate access to the manager because
 * the Context should provide the filesystem abstraction to more intelligently
 * map named file paths to resources. For now, handles and other consumers share ownership
 * of the TprContents manager object via shared_ptr.
 */
class TprContents;

class TprReadHandle
{
public:
    explicit TprReadHandle(std::shared_ptr<TprContents> tprFile);
    explicit TprReadHandle(TprContents&& tprFile);
    TprReadHandle(const TprReadHandle&) = default;
    TprReadHandle& operator=(const TprReadHandle&) = default;
    TprReadHandle(TprReadHandle&&) noexcept        = default;
    TprReadHandle& operator=(TprReadHandle&&) noexcept = default;
    ~TprReadHandle();

    /*!
     * \brief Allow API functions to access data resources.
     *
     * Used internally. The entire TPR contents are never extracted to the
     * client, but API implementation details need to be
     * able to access some or all entire contents in later operations.
     *
     * \return Reference-counted handle to data container.
     */
    [[nodiscard]] std::shared_ptr<TprContents> get() const;

private:
    std::shared_ptr<TprContents> tprContents_;
};

/*!
 * \brief Helper function for early implementation.
 *
 * Allows extraction of TPR file information from special params objects.
 *
 * \todo This is a very temporary shim! Find a better way to construct simulation input.
 */
TprReadHandle getSourceFileHandle(const GmxMdParams& params);

class StructureSource
{
public:
    std::shared_ptr<TprContents> tprFile_;
};

class TopologySource
{
public:
    std::shared_ptr<TprContents> tprFile_;
};

class SimulationState
{
public:
    std::shared_ptr<TprContents> tprFile_;
};

/*!
 * \brief Buffer descriptor for GROMACS coordinates data.
 *
 * This structure should be sufficient to map a GROMACS managed coordinates buffer to common
 * buffer protocols for API array data exchange.
 *
 * \warning The coordinates may be internally stored as 32-bit floating point numbers, but
 * GROMACS developers have not yet agreed to include 32-bit floating point data in the gmxapi
 * data typing specification.
 */
struct CoordinatesBuffer
{
    void*               ptr;
    gmxapi::GmxapiType  itemType;
    size_t              itemSize;
    size_t              ndim;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
};

template<typename T, class Enable = void>
struct is_gmxapi_data_buffer : std::false_type
{
};

template<typename T>
struct is_gmxapi_data_buffer<
        T,
        std::enable_if_t<
                std::is_pointer_v<decltype(
                        T::ptr)> && std::is_same_v<decltype(T::itemType), gmxapi::GmxapiType> && std::is_convertible_v<decltype(T::itemSize), size_t> && std::is_convertible_v<decltype(T::ndim), size_t> && std::is_convertible_v<decltype(T::shape), std::vector<size_t>> && std::is_convertible_v<decltype(T::strides), std::vector<size_t>>>> :
    std::true_type
{
};

static_assert(is_gmxapi_data_buffer<CoordinatesBuffer>::value,
              "Interface cannot support buffer protocols.");

/*!
 * \brief Floating point precision mismatch.
 *
 * Operation cannot be performed at the requested precision for the provided input.
 *
 * \ingroup gmxapi_exceptions
 */
class PrecisionError : public gmxapi::BasicException<PrecisionError>
{
public:
    using BasicException<PrecisionError>::BasicException;
};

/*!
 * \brief Get buffer description for coordinates from a source of structure data.
 *
 * \param structure
 * \param tag type tag for dispatching
 * \return Buffer description.
 *
 * Caller is responsible for keeping the source alive while the buffer is in use.
 *
 * \throws PrecisionError if template parameter does not match the available data.
 */
/*! \{ */
CoordinatesBuffer coordinates(const StructureSource& structure, const float& tag);
CoordinatesBuffer coordinates(const StructureSource& structure, const double& tag);
/*! \} */

class TprBuilder
{
public:
    explicit TprBuilder(std::unique_ptr<TprContents> tprFile);
    explicit TprBuilder(TprContents&& tprFile);
    // Multiple write handles to the same resource is not supported, and we don't have a good way
    // to copy the resource.
    TprBuilder(const TprBuilder&) = delete;
    TprBuilder& operator=(const TprBuilder&) = delete;
    // Move semantics should be straighforward.
    TprBuilder(TprBuilder&&) noexcept = default;
    TprBuilder& operator=(TprBuilder&&) noexcept = default;
    ~TprBuilder();

    /*!
     * \brief Get the floating point precision for the TPR contents being edited.
     *
     * \return Number of bytes for floating point numbers in fields described as "real".
     */
    [[nodiscard]] size_t get_precision() const;

    /*!
     * \brief Replace particle coordinates.
     *
     * \param coordinates
     * \return Reference to the same builder.
     * \throws PrecisionError if provided CoordinatesBuffer does not match current contents
     * precision.
     * \throws ProtocolError if CoordinatesBuffer contains data that is inconsistent with
     * documented usage.
     */
    TprBuilder& set(const CoordinatesBuffer& coordinates);

    /*!
     * \brief Write contents to the specified filename.
     *
     * \param filename
     */
    void write(const std::string& filename);

private:
    std::unique_ptr<TprContents> tprContents_;
};

/*!
 * \brief Copy TPR file.
 *
 * \param input TPR source to copy from
 * \param outFile output TPR file name
 * \return true if successful. else false.
 */
bool copy_tprfile(const gmxapicompat::TprReadHandle& input, const std::string& outFile);

/*!
 * \brief Copy and possibly update TPR file by name.
 *
 * \param inFile Input file name
 * \param outFile Output file name
 * \param endTime Replace `nsteps` in infile with `endTime/dt`
 * \return true if successful, else false
 */
bool rewrite_tprfile(const std::string& inFile, const std::string& outFile, double endTime);

} // end namespace gmxapicompat

#endif // GMXAPICOMPAT_TPR_H
