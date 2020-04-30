/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020, by the GROMACS development team, led by
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
 * \brief Provides the checkpoint data structure for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdtypes
 */

#ifndef GMX_MODULARSIMULATOR_CHECKPOINTDATA_H
#define GMX_MODULARSIMULATOR_CHECKPOINTDATA_H

#include "gromacs/compat/optional.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/keyvaluetreebuilder.h"

namespace gmx
{
/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief The operations on CheckpointData
 *
 * This enum defines the two modes of operation on CheckpointData objects,
 * reading and writing. This allows to template all access functions, which
 * in turn enables clients to write a single function for read and write
 * access, eliminating the risk of having read and write functions getting
 * out of sync.
 */
enum class CheckpointDataOperation
{
    Read,
    Write
};

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Get an ArrayRef whose const-ness is defined by the checkpointing operation
 *
 * @tparam operation  Whether we are reading or writing
 * @tparam T          The type of values stored in the ArrayRef
 * @param container   The container the ArrayRef is referencing to
 * @return            The ArrayRef
 *
 * \see ArrayRef
 */
template<CheckpointDataOperation operation, typename T>
ArrayRef<std::conditional_t<operation == CheckpointDataOperation::Write || std::is_const<T>::value, const typename T::value_type, typename T::value_type>>
makeCheckpointArrayRef(T& container)
{
    return container;
}

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Struct allowing to check if data is serializable
 *
 * This list of types is copied from ValueSerializer::initSerializers()
 * Having this here allows us to catch errors at compile time
 * instead of having cryptical runtime errors
 */
template<typename T>
struct IsSerializableType
{
    static bool const value = std::is_same<T, std::string>::value || std::is_same<T, bool>::value
                              || std::is_same<T, int>::value || std::is_same<T, int64_t>::value
                              || std::is_same<T, float>::value || std::is_same<T, double>::value;
};

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Struct allowing to check if enum has a serializable underlying type
 */
//! {
template<typename T, bool = std::is_enum<T>::value>
struct IsSerializableEnum
{
    static bool const value = IsSerializableType<std::underlying_type_t<T>>::value;
};
template<typename T>
struct IsSerializableEnum<T, false>
{
    static bool const value = false;
};
//! }

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Data type hiding checkpoint implementation details
 *
 * This data type allows to separate the implementation details of the
 * checkpoint writing / reading from the implementation of the checkpoint
 * clients. Checkpoint clients interface via the methods of the CheckpointData
 * object, and do not need knowledge of data types used to store the data.
 */
class CheckpointData
{
public:
    /*! \brief Read or write a single value from / to checkpoint
     *
     * Allowed scalar types include std::string, bool, int, int64_t,
     * float, double, or any enum with one of the previously mentioned
     * scalar types as underlying type. Type compatibility is checked
     * at compile time.
     *
     * @tparam operation  Whether we are reading or writing
     * @tparam T          The type of the value
     * @param key         The key to [read|write] the value [from|to]
     * @param value       The value to [read|write]
     */
    //! {
    // Read
    template<CheckpointDataOperation operation, typename T>
    typename std::enable_if_t<operation == CheckpointDataOperation::Read && IsSerializableType<T>::value, void>
    scalar(const std::string& key, T* value) const;
    template<CheckpointDataOperation operation, typename T>
    typename std::enable_if_t<operation == CheckpointDataOperation::Read && IsSerializableEnum<T>::value, void>
    scalar(const std::string& key, T* value) const;
    // Write
    template<CheckpointDataOperation operation, typename T>
    typename std::enable_if_t<operation == CheckpointDataOperation::Write && IsSerializableType<T>::value, void>
    scalar(const std::string& key, const T* value);
    template<CheckpointDataOperation operation, typename T>
    typename std::enable_if_t<operation == CheckpointDataOperation::Write && IsSerializableEnum<T>::value, void>
    scalar(const std::string& key, const T* value);
    //! }

    /*! \brief Read or write an ArrayRef from / to checkpoint
     *
     * Allowed types stored in the ArrayRef include std::string, bool, int,
     * int64_t, float, double, and gmx::RVec. Type compatibility is checked
     * at compile time.
     *
     * @tparam operation  Whether we are reading or writing
     * @tparam T          The type of values stored in the ArrayRef
     * @param key         The key to [read|write] the ArrayRef [from|to]
     * @param values      The ArrayRef to [read|write]
     */
    //! {
    // Read ArrayRef of scalar
    template<CheckpointDataOperation operation, typename T>
    typename std::enable_if_t<operation == CheckpointDataOperation::Read && IsSerializableType<T>::value, void>
    arrayRef(const std::string& key, ArrayRef<T> values) const;
    // Write ArrayRef of scalar
    template<CheckpointDataOperation operation, typename T>
    typename std::enable_if_t<operation == CheckpointDataOperation::Write && IsSerializableType<T>::value, void>
    arrayRef(const std::string& key, ArrayRef<const T> values);
    // Read ArrayRef of RVec
    template<CheckpointDataOperation operation>
    typename std::enable_if_t<operation == CheckpointDataOperation::Read, void>
    arrayRef(const std::string& key, ArrayRef<RVec> values) const;
    // Write ArrayRef of RVec
    template<CheckpointDataOperation operation>
    typename std::enable_if_t<operation == CheckpointDataOperation::Write, void>
    arrayRef(const std::string& key, ArrayRef<const RVec> values);
    //! }

    /*! \brief Read or write a tensor from / to checkpoint
     *
     * @tparam operation  Whether we are reading or writing
     * @param key         The key to [read|write] the tensor [from|to]
     * @param values      The tensor to [read|write]
     */
    //! {
    // Read
    template<CheckpointDataOperation operation>
    typename std::enable_if<operation == CheckpointDataOperation::Read, void>::type
    tensor(const std::string& key, ::tensor values) const;
    // Write
    template<CheckpointDataOperation operation>
    typename std::enable_if<operation == CheckpointDataOperation::Write, void>::type
    tensor(const std::string& key, const ::tensor values);
    //! }

    /*! \brief Return a subset of the current CheckpointData
     *
     * @tparam operation  Whether we are reading or writing
     * @param key         The key to [read|write] the sub data [from|to]
     * @return            A CheckpointData object representing a subset of the current object
     */
    //!{
    // Read
    template<CheckpointDataOperation operation>
    typename std::enable_if<operation == CheckpointDataOperation::Read, CheckpointData>::type
    subCheckpointData(const std::string& key) const;
    // Write
    template<CheckpointDataOperation operation>
    typename std::enable_if<operation == CheckpointDataOperation::Write, CheckpointData>::type
    subCheckpointData(const std::string& key);
    //!}

    //! Construct an input checkpoint data object
    explicit CheckpointData(const KeyValueTreeObject& inputTree) :
        outputTreeBuilder_(compat::nullopt),
        inputTree_(&inputTree)
    {
    }
    //! Construct an output checkpoint data object
    explicit CheckpointData(KeyValueTreeObjectBuilder&& outputTreeBuilder) :
        outputTreeBuilder_(outputTreeBuilder),
        inputTree_(nullptr)
    {
    }

private:
    //! Builder for the tree to be written to checkpoint
    compat::optional<KeyValueTreeObjectBuilder> outputTreeBuilder_;
    //! KV tree read from checkpoint
    const KeyValueTreeObject* inputTree_;
};

// Implementation of scalar reading
template<CheckpointDataOperation operation, typename T>
typename std::enable_if_t<operation == CheckpointDataOperation::Read && IsSerializableType<T>::value, void>
CheckpointData::scalar(const std::string& key, T* value) const
{
    GMX_RELEASE_ASSERT(inputTree_, "No input checkpoint data available.");
    *value = (*inputTree_)[key].cast<T>();
}
// Implementation of scalar reading (enum)
template<CheckpointDataOperation operation, typename T>
typename std::enable_if_t<operation == CheckpointDataOperation::Read && IsSerializableEnum<T>::value, void>
CheckpointData::scalar(const std::string& key, T* value) const
{
    GMX_RELEASE_ASSERT(inputTree_, "No input checkpoint data available.");
    std::underlying_type_t<T> castValue;
    castValue = (*inputTree_)[key].cast<std::underlying_type_t<T>>();
    *value    = static_cast<T>(castValue);
}
// Implementation of scalar writing
template<CheckpointDataOperation operation, typename T>
typename std::enable_if_t<operation == CheckpointDataOperation::Write && IsSerializableType<T>::value, void>
CheckpointData::scalar(const std::string& key, const T* value)
{
    GMX_RELEASE_ASSERT(outputTreeBuilder_, "No output checkpoint data available.");
    outputTreeBuilder_->addValue(key, *value);
}
// Implementation of scalar writing (enum)
template<CheckpointDataOperation operation, typename T>
typename std::enable_if_t<operation == CheckpointDataOperation::Write && IsSerializableEnum<T>::value, void>
CheckpointData::scalar(const std::string& key, const T* value)
{
    GMX_RELEASE_ASSERT(outputTreeBuilder_, "No output checkpoint data available.");
    auto castValue = static_cast<std::underlying_type_t<T>>(*value);
    outputTreeBuilder_->addValue(key, castValue);
}

// Implementation of scalar ArrayRef reading
template<CheckpointDataOperation operation, typename T>
typename std::enable_if_t<operation == CheckpointDataOperation::Read && IsSerializableType<T>::value, void>
CheckpointData::arrayRef(const std::string& key, ArrayRef<T> values) const
{
    GMX_RELEASE_ASSERT(inputTree_, "No input checkpoint data available.");
    GMX_RELEASE_ASSERT(values.size() >= (*inputTree_)[key].asArray().values().size(),
                       "Read vector does not fit in passed ArrayRef.");
    auto outputIt  = values.begin();
    auto inputIt   = (*inputTree_)[key].asArray().values().begin();
    auto outputEnd = values.end();
    auto inputEnd  = (*inputTree_)[key].asArray().values().end();
    for (; outputIt != outputEnd && inputIt != inputEnd; outputIt++, inputIt++)
    {
        *outputIt = inputIt->cast<T>();
    }
}
// Implementation of scalar ArrayRef writing
template<CheckpointDataOperation operation, typename T>
std::enable_if_t<operation == CheckpointDataOperation::Write && IsSerializableType<T>::value, void>
CheckpointData::arrayRef(const std::string& key, ArrayRef<const T> values)
{
    GMX_RELEASE_ASSERT(outputTreeBuilder_, "No output checkpoint data available.");
    auto builder = outputTreeBuilder_->addUniformArray<T>(key);
    for (const auto& value : values)
    {
        builder.addValue(value);
    }
}
// Implementation of RVec ArrayRef reading
template<>
inline void CheckpointData::arrayRef<CheckpointDataOperation::Read>(const std::string& key,
                                                                    ArrayRef<RVec>     values) const
{
    GMX_RELEASE_ASSERT(inputTree_, "No input checkpoint data available.");
    GMX_RELEASE_ASSERT(values.size() >= (*inputTree_)[key].asArray().values().size(),
                       "Read vector does not fit in passed ArrayRef.");
    auto* outputIt  = values.begin();
    auto  inputIt   = (*inputTree_)[key].asArray().values().begin();
    auto* outputEnd = values.end();
    auto  inputEnd  = (*inputTree_)[key].asArray().values().end();
    for (; outputIt != outputEnd && inputIt != inputEnd; outputIt++, inputIt++)
    {
        auto storedRVec = inputIt->asObject()["RVec"].asArray().values();
        *outputIt       = { storedRVec[XX].cast<real>(), storedRVec[YY].cast<real>(),
                      storedRVec[ZZ].cast<real>() };
    }
}
// Implementation of RVec ArrayRef writing
template<>
inline void CheckpointData::arrayRef<CheckpointDataOperation::Write>(const std::string&   key,
                                                                     ArrayRef<const RVec> values)
{
    GMX_RELEASE_ASSERT(outputTreeBuilder_, "No output checkpoint data available.");
    auto builder = outputTreeBuilder_->addObjectArray(key);
    for (const auto& value : values)
    {
        auto subbuilder = builder.addObject();
        subbuilder.addUniformArray("RVec", { value[XX], value[YY], value[ZZ] });
    }
}

// Implementation of tensor reading
template<CheckpointDataOperation operation>
typename std::enable_if<operation == CheckpointDataOperation::Read, void>::type
CheckpointData::tensor(const std::string& key, ::tensor values) const
{
    GMX_RELEASE_ASSERT(inputTree_, "No input checkpoint data available.");
    auto array     = (*inputTree_)[key].asArray().values();
    values[XX][XX] = array[0].cast<real>();
    values[XX][YY] = array[1].cast<real>();
    values[XX][ZZ] = array[2].cast<real>();
    values[YY][XX] = array[3].cast<real>();
    values[YY][YY] = array[4].cast<real>();
    values[YY][ZZ] = array[5].cast<real>();
    values[ZZ][XX] = array[6].cast<real>();
    values[ZZ][YY] = array[7].cast<real>();
    values[ZZ][ZZ] = array[8].cast<real>();
}
// Implementation of tensor writing
template<CheckpointDataOperation operation>
typename std::enable_if<operation == CheckpointDataOperation::Write, void>::type
CheckpointData::tensor(const std::string& key, const ::tensor values)
{
    GMX_RELEASE_ASSERT(outputTreeBuilder_, "No output checkpoint data available.");
    auto builder = outputTreeBuilder_->addUniformArray<real>(key);
    builder.addValue(values[XX][XX]);
    builder.addValue(values[XX][YY]);
    builder.addValue(values[XX][ZZ]);
    builder.addValue(values[YY][XX]);
    builder.addValue(values[YY][YY]);
    builder.addValue(values[YY][ZZ]);
    builder.addValue(values[ZZ][XX]);
    builder.addValue(values[ZZ][YY]);
    builder.addValue(values[ZZ][ZZ]);
}

// Implementation of sub data reading
template<CheckpointDataOperation operation>
typename std::enable_if<operation == CheckpointDataOperation::Read, CheckpointData>::type
CheckpointData::subCheckpointData(const std::string& key) const
{
    GMX_RELEASE_ASSERT(inputTree_, "No input checkpoint data available.");
    return CheckpointData((*inputTree_)[key].asObject());
}
// Implementation of sub data writing
template<CheckpointDataOperation operation>
typename std::enable_if<operation == CheckpointDataOperation::Write, CheckpointData>::type
CheckpointData::subCheckpointData(const std::string& key)
{
    GMX_RELEASE_ASSERT(outputTreeBuilder_, "No output tree builder available.");
    return CheckpointData(outputTreeBuilder_->addObject(key));
}

} // namespace gmx

#endif // GMX_MODULARSIMULATOR_CHECKPOINTDATA_H
