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

#include "gmxpre.h"

#include "gromacs/mdtypes/checkpointdata.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/utility/fatalerror.h"

namespace gmx
{
namespace
{

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Struct allowing to check if type is vector of serializable data
 */
//! \{
template<class T>
struct IsVectorOfSerializableType
{
    static bool const value = false;
};
template<class T>
struct IsVectorOfSerializableType<std::vector<T>>
{
    static bool const value = IsSerializableType<T>::value;
};
//! \}

/*! \brief Unified looping over test data
 *
 * This class allows to write a loop over test data as
 *   for (const auto& value : TestValues::testValueGenerator<type>())
 * where type can be any of std::string, int, int64_t, bool, float, double,
 * std::vector<[std::string, int, int64_6, float, double]>, or tensor.
 */
class TestValues
{
public:
    template<typename T>
    class TestValueGenerator
    {
    public:
        class Iterator
        {
        public:
            explicit Iterator(const T* ptr) : ptr_(ptr) {}
            Iterator operator++();
            bool     operator!=(const Iterator& other) const { return ptr_ != other.ptr_; }
            const T& operator*() const { return *ptr_; }

        private:
            const T* ptr_;
        };

        Iterator begin() const;
        Iterator end() const;
    };

    template<typename T>
    static TestValueGenerator<T> testValueGenerator()
    {
        static const TestValueGenerator<T> testValueGenerator;
        return testValueGenerator;
    }

private:
    template<typename T>
    static const std::vector<T>& getTestVector();

    template<typename T>
    static std::enable_if_t<IsSerializableType<T>::value && !std::is_same<T, bool>::value, const T*>
    getBeginPointer();
    template<typename T>
    static std::enable_if_t<IsVectorOfSerializableType<T>::value, const T*> getBeginPointer();
    template<typename T>
    static std::enable_if_t<std::is_same<T, bool>::value, const T*> getBeginPointer();
    template<typename T>
    static std::enable_if_t<std::is_same<T, tensor>::value, const T*> getBeginPointer();

    template<typename T>
    static std::enable_if_t<IsSerializableType<T>::value && !std::is_same<T, bool>::value, const T*>
    getEndPointer();
    template<typename T>
    static std::enable_if_t<IsVectorOfSerializableType<T>::value, const T*> getEndPointer();
    template<typename T>
    static std::enable_if_t<std::is_same<T, bool>::value, const T*> getEndPointer();
    template<typename T>
    static std::enable_if_t<std::is_same<T, tensor>::value, const T*> getEndPointer();

    template<typename T>
    static std::enable_if_t<IsSerializableType<T>::value && !std::is_same<T, bool>::value, void>
    increment(const T** ptr);
    template<typename T>
    static std::enable_if_t<IsVectorOfSerializableType<T>::value, void> increment(const T** ptr);
    template<typename T>
    static std::enable_if_t<std::is_same<T, bool>::value, void> increment(const T** ptr);
    template<typename T>
    static std::enable_if_t<std::is_same<T, tensor>::value, void> increment(const T** ptr);

    constexpr static bool   testTrue    = true;
    constexpr static bool   testFalse   = false;
    constexpr static tensor testTensor1 = { { 1.6234, 2.4632, 3.1112 },
                                            { 4.66234, 5.9678, 6.088 },
                                            { 7.00001, 8.43535, 9.11233 } };
#if GMX_DOUBLE
    constexpr static tensor testTensor2 = { { 1, GMX_DOUBLE_EPS, 3 },
                                            { GMX_DOUBLE_MIN, 5, 6 },
                                            { 7, 8, GMX_DOUBLE_MAX } };
#else
    constexpr static tensor testTensor2 = { { 1, GMX_FLOAT_EPS, 3 },
                                            { GMX_FLOAT_MIN, 5, 6 },
                                            { 7, 8, GMX_FLOAT_MAX } };
#endif
};

// Remove for c++17
constexpr bool   TestValues::testTrue;
constexpr bool   TestValues::testFalse;
constexpr tensor TestValues::testTensor1;
constexpr tensor TestValues::testTensor2;

// Begin implementations of TestValues methods
template<>
const std::vector<std::string>& TestValues::getTestVector()
{
    static const std::vector<std::string> testStrings({ "Test string\nwith newlines\n", "" });
    return testStrings;
}
template<>
const std::vector<int>& TestValues::getTestVector()
{
    static const std::vector<int> testInts({ { 3, INT_MAX, INT_MIN } });
    return testInts;
}
template<>
const std::vector<int64_t>& TestValues::getTestVector()
{
    static const std::vector<int64_t> testInt64s({ -7, LLONG_MAX, LLONG_MIN });
    return testInt64s;
}
template<>
const std::vector<float>& TestValues::getTestVector()
{
    static const std::vector<float> testFloats({ 33.9, GMX_FLOAT_MAX, GMX_FLOAT_MIN, GMX_FLOAT_EPS });
    return testFloats;
}
template<>
const std::vector<double>& TestValues::getTestVector()
{
    static const std::vector<double> testDoubles({ -123.45, GMX_DOUBLE_MAX, GMX_DOUBLE_MIN, GMX_DOUBLE_EPS });
    return testDoubles;
}

template<typename T>
std::enable_if_t<IsSerializableType<T>::value && !std::is_same<T, bool>::value, const T*> TestValues::getBeginPointer()
{
    return getTestVector<T>().data();
}
template<typename T>
std::enable_if_t<IsVectorOfSerializableType<T>::value, const T*> TestValues::getBeginPointer()
{
    return &getTestVector<typename T::value_type>();
}
template<typename T>
std::enable_if_t<std::is_same<T, bool>::value, const T*> TestValues::getBeginPointer()
{
    return &testTrue;
}
template<typename T>
std::enable_if_t<std::is_same<T, tensor>::value, const T*> TestValues::getBeginPointer()
{
    return &testTensor1;
}

template<typename T>
std::enable_if_t<IsSerializableType<T>::value && !std::is_same<T, bool>::value, const T*> TestValues::getEndPointer()
{
    return getTestVector<T>().data() + getTestVector<T>().size();
}
template<typename T>
std::enable_if_t<IsVectorOfSerializableType<T>::value, const T*> TestValues::getEndPointer()
{
    return &getTestVector<typename T::value_type>() + 1;
}
template<typename T>
std::enable_if_t<std::is_same<T, bool>::value, const T*> TestValues::getEndPointer()
{
    return nullptr;
}
template<typename T>
std::enable_if_t<std::is_same<T, tensor>::value, const T*> TestValues::getEndPointer()
{
    return nullptr;
}

template<typename T>
std::enable_if_t<IsSerializableType<T>::value && !std::is_same<T, bool>::value, void>
TestValues::increment(const T** ptr)
{
    ++(*ptr);
}
template<typename T>
std::enable_if_t<IsVectorOfSerializableType<T>::value, void> TestValues::increment(const T** ptr)
{
    ++(*ptr);
}
template<typename T>
std::enable_if_t<std::is_same<T, bool>::value, void> TestValues::increment(const T** ptr)
{
    *ptr = (*ptr == &testTrue) ? &testFalse : nullptr;
}
template<typename T>
std::enable_if_t<std::is_same<T, tensor>::value, void> TestValues::increment(const T** ptr)
{
    *ptr = (*ptr == &testTensor1) ? &testTensor2 : nullptr;
}

template<typename T>
typename TestValues::TestValueGenerator<T>::Iterator TestValues::TestValueGenerator<T>::begin() const
{
    return TestValues::TestValueGenerator<T>::Iterator(getBeginPointer<T>());
}

template<typename T>
typename TestValues::TestValueGenerator<T>::Iterator TestValues::TestValueGenerator<T>::end() const
{
    return TestValues::TestValueGenerator<T>::Iterator(getEndPointer<T>());
}

template<typename T>
typename TestValues::TestValueGenerator<T>::Iterator TestValues::TestValueGenerator<T>::Iterator::operator++()
{
    TestValues::increment(&ptr_);
    return *this;
}
// End implementations of TestValues methods

//! Write scalar input to CheckpointData
template<typename T>
typename std::enable_if_t<IsSerializableType<T>::value, void> writeInput(const std::string& key,
                                                                         const T&        inputValue,
                                                                         CheckpointData* checkpointData)
{
    checkpointData->scalar<CheckpointDataOperation::Write>(key, &inputValue);
}
//! Read scalar from CheckpointData and test if equal to input
template<typename T>
typename std::enable_if_t<IsSerializableType<T>::value, void> testOutput(const std::string& key,
                                                                         const T&        inputValue,
                                                                         CheckpointData* checkpointData)
{
    T outputValue;
    checkpointData->scalar<CheckpointDataOperation::Read>(key, &outputValue);
    EXPECT_EQ(inputValue, outputValue);
}
//! Write vector input to CheckpointData
template<typename T>
void writeInput(const std::string& key, const std::vector<T> inputVector, CheckpointData* checkpointData)
{
    checkpointData->arrayRef<CheckpointDataOperation::Write>(key, makeConstArrayRef(inputVector));
}
//! Read vector from CheckpointData and test if equal to input
template<typename T>
void testOutput(const std::string& key, const std::vector<T> inputVector, CheckpointData* checkpointData)
{
    std::vector<T> outputVector;
    outputVector.resize(inputVector.size());
    checkpointData->arrayRef<CheckpointDataOperation::Read>(key, makeArrayRef(outputVector));
    ASSERT_THAT(outputVector, ::testing::ContainerEq(inputVector));
}
//! Write tensor input to CheckpointData
void writeInput(const std::string& key, const tensor inputTensor, CheckpointData* checkpointData)
{
    checkpointData->tensor<CheckpointDataOperation::Write>(key, inputTensor);
}
//! Read tensor from CheckpointData and test if equal to input
void testOutput(const std::string& key, const tensor inputTensor, CheckpointData* checkpointData)
{
    tensor outputTensor = { { 0 } };
    checkpointData->tensor<CheckpointDataOperation::Read>(key, outputTensor);
    std::array<std::array<real, 3>, 3> inputTensorA = {
        { { inputTensor[XX][XX], inputTensor[XX][YY], inputTensor[XX][ZZ] },
          { inputTensor[YY][XX], inputTensor[YY][YY], inputTensor[YY][ZZ] },
          { inputTensor[ZZ][XX], inputTensor[ZZ][YY], inputTensor[ZZ][ZZ] } }
    };
    std::array<std::array<real, 3>, 3> outputTensorA = {
        { { outputTensor[XX][XX], outputTensor[XX][YY], outputTensor[XX][ZZ] },
          { outputTensor[YY][XX], outputTensor[YY][YY], outputTensor[YY][ZZ] },
          { outputTensor[ZZ][XX], outputTensor[ZZ][YY], outputTensor[ZZ][ZZ] } }
    };
    ASSERT_THAT(outputTensorA, ::testing::ContainerEq(inputTensorA));
}

/*!\brief CheckpointData test fixture
 *
 * Test whether input is equal to output, either with a single data type
 * or with a combination of three data types.
 */
class CheckpointDataTest : public ::testing::Test
{
public:
    template<typename T>
    void singleTest()
    {
        for (const auto& inputValue : TestValues::testValueGenerator<T>())
        {
            KeyValueTreeBuilder treeBuilder;
            CheckpointData      inputCheckpointData(treeBuilder.rootObject());
            writeInput("test", inputValue, &inputCheckpointData);

            const auto kvTree = treeBuilder.build();

            CheckpointData outputCheckpointData(kvTree);
            testOutput("test", inputValue, &outputCheckpointData);
        }
    }
    template<typename T1, typename T2, typename T3>
    void multiTest()
    {
        for (const auto& inputValue1 : TestValues::testValueGenerator<T1>())
        {
            for (const auto& inputValue2 : TestValues::testValueGenerator<T2>())
            {
                for (const auto& inputValue3 : TestValues::testValueGenerator<T3>())
                {
                    KeyValueTreeBuilder treeBuilder;
                    CheckpointData      inputCheckpointData(treeBuilder.rootObject());
                    writeInput("multi1", inputValue1, &inputCheckpointData);
                    writeInput("multi2", inputValue2, &inputCheckpointData);
                    writeInput("multi3", inputValue3, &inputCheckpointData);

                    const auto kvTree = treeBuilder.build();

                    CheckpointData outputCheckpointData(kvTree);
                    // read out in different order
                    testOutput("multi2", inputValue2, &outputCheckpointData);
                    testOutput("multi3", inputValue3, &outputCheckpointData);
                    testOutput("multi1", inputValue1, &outputCheckpointData);
                }
            }
        }
    }
};

TEST_F(CheckpointDataTest, SingleDataTest)
{
    // All separate data types
    singleTest<std::string>();
    singleTest<int>();
    singleTest<int64_t>();
    singleTest<bool>();
    singleTest<float>();
    singleTest<double>();
    singleTest<std::vector<std::string>>();
    singleTest<std::vector<int>>();
    singleTest<std::vector<int64_t>>();
    singleTest<std::vector<float>>();
    singleTest<std::vector<double>>();
    singleTest<tensor>();
}

TEST_F(CheckpointDataTest, PartialMultiDataTest)
{
    // Some randomly generated combinations of multiple data types
    multiTest<std::vector<int>, std::vector<double>, std::vector<float>>();
    multiTest<std::vector<int>, double, std::vector<int>>();
    multiTest<std::vector<double>, std::string, std::vector<float>>();
    multiTest<int, int64_t, std::vector<int>>();
    multiTest<int64_t, std::vector<int>, tensor>();
    multiTest<std::vector<double>, std::string, std::vector<std::string>>();
    multiTest<float, float, tensor>();
    multiTest<std::vector<int64_t>, bool, std::vector<float>>();
    multiTest<int64_t, int64_t, int64_t>();
    multiTest<std::vector<int>, std::vector<int>, std::vector<int>>();
    multiTest<double, std::vector<std::string>, std::vector<std::string>>();
    multiTest<std::vector<std::string>, double, bool>();
    multiTest<int, float, std::vector<int>>();
    multiTest<std::vector<double>, std::vector<float>, std::vector<float>>();
    multiTest<std::vector<float>, tensor, std::vector<int64_t>>();
    multiTest<int, int64_t, std::vector<int64_t>>();
    multiTest<tensor, std::vector<std::string>, double>();
    multiTest<std::vector<double>, std::vector<int>, std::vector<int64_t>>();
    multiTest<std::vector<int64_t>, std::vector<float>, int64_t>();
    multiTest<int64_t, float, int>();
}

TEST_F(CheckpointDataTest, AllMultiDataTests)
{
    // All possible tests (assuming order doesn't matter, 362 tests)
    multiTest<std::string, std::string, std::string>();
    multiTest<std::string, std::string, int>();
    multiTest<std::string, std::string, int64_t>();
    multiTest<std::string, std::string, bool>();
    multiTest<std::string, std::string, float>();
    multiTest<std::string, std::string, double>();
    multiTest<std::string, std::string, std::vector<std::string>>();
    multiTest<std::string, std::string, std::vector<int>>();
    multiTest<std::string, std::string, std::vector<int64_t>>();
    multiTest<std::string, std::string, std::vector<float>>();
    multiTest<std::string, std::string, std::vector<double>>();
    multiTest<std::string, std::string, tensor>();
    multiTest<std::string, int, int>();
    multiTest<std::string, int, int64_t>();
    multiTest<std::string, int, bool>();
    multiTest<std::string, int, float>();
    multiTest<std::string, int, double>();
    multiTest<std::string, int, std::vector<std::string>>();
    multiTest<std::string, int, std::vector<int>>();
    multiTest<std::string, int, std::vector<int64_t>>();
    multiTest<std::string, int, std::vector<float>>();
    multiTest<std::string, int, std::vector<double>>();
    multiTest<std::string, int, tensor>();
    multiTest<std::string, int64_t, int64_t>();
    multiTest<std::string, int64_t, bool>();
    multiTest<std::string, int64_t, float>();
    multiTest<std::string, int64_t, double>();
    multiTest<std::string, int64_t, std::vector<std::string>>();
    multiTest<std::string, int64_t, std::vector<int>>();
    multiTest<std::string, int64_t, std::vector<int64_t>>();
    multiTest<std::string, int64_t, std::vector<float>>();
    multiTest<std::string, int64_t, std::vector<double>>();
    multiTest<std::string, int64_t, tensor>();
    multiTest<std::string, bool, bool>();
    multiTest<std::string, bool, float>();
    multiTest<std::string, bool, double>();
    multiTest<std::string, bool, std::vector<std::string>>();
    multiTest<std::string, bool, std::vector<int>>();
    multiTest<std::string, bool, std::vector<int64_t>>();
    multiTest<std::string, bool, std::vector<float>>();
    multiTest<std::string, bool, std::vector<double>>();
    multiTest<std::string, bool, tensor>();
    multiTest<std::string, float, float>();
    multiTest<std::string, float, double>();
    multiTest<std::string, float, std::vector<std::string>>();
    multiTest<std::string, float, std::vector<int>>();
    multiTest<std::string, float, std::vector<int64_t>>();
    multiTest<std::string, float, std::vector<float>>();
    multiTest<std::string, float, std::vector<double>>();
    multiTest<std::string, float, tensor>();
    multiTest<std::string, double, double>();
    multiTest<std::string, double, std::vector<std::string>>();
    multiTest<std::string, double, std::vector<int>>();
    multiTest<std::string, double, std::vector<int64_t>>();
    multiTest<std::string, double, std::vector<float>>();
    multiTest<std::string, double, std::vector<double>>();
    multiTest<std::string, double, tensor>();
    multiTest<std::string, std::vector<std::string>, std::vector<std::string>>();
    multiTest<std::string, std::vector<std::string>, std::vector<int>>();
    multiTest<std::string, std::vector<std::string>, std::vector<int64_t>>();
    multiTest<std::string, std::vector<std::string>, std::vector<float>>();
    multiTest<std::string, std::vector<std::string>, std::vector<double>>();
    multiTest<std::string, std::vector<std::string>, tensor>();
    multiTest<std::string, std::vector<int>, std::vector<int>>();
    multiTest<std::string, std::vector<int>, std::vector<int64_t>>();
    multiTest<std::string, std::vector<int>, std::vector<float>>();
    multiTest<std::string, std::vector<int>, std::vector<double>>();
    multiTest<std::string, std::vector<int>, tensor>();
    multiTest<std::string, std::vector<int64_t>, std::vector<int64_t>>();
    multiTest<std::string, std::vector<int64_t>, std::vector<float>>();
    multiTest<std::string, std::vector<int64_t>, std::vector<double>>();
    multiTest<std::string, std::vector<int64_t>, tensor>();
    multiTest<std::string, std::vector<float>, std::vector<float>>();
    multiTest<std::string, std::vector<float>, std::vector<double>>();
    multiTest<std::string, std::vector<float>, tensor>();
    multiTest<std::string, std::vector<double>, std::vector<double>>();
    multiTest<std::string, std::vector<double>, tensor>();
    multiTest<std::string, tensor, tensor>();
    multiTest<int, int, int>();
    multiTest<int, int, int64_t>();
    multiTest<int, int, bool>();
    multiTest<int, int, float>();
    multiTest<int, int, double>();
    multiTest<int, int, std::vector<std::string>>();
    multiTest<int, int, std::vector<int>>();
    multiTest<int, int, std::vector<int64_t>>();
    multiTest<int, int, std::vector<float>>();
    multiTest<int, int, std::vector<double>>();
    multiTest<int, int, tensor>();
    multiTest<int, int64_t, int64_t>();
    multiTest<int, int64_t, bool>();
    multiTest<int, int64_t, float>();
    multiTest<int, int64_t, double>();
    multiTest<int, int64_t, std::vector<std::string>>();
    multiTest<int, int64_t, std::vector<int>>();
    multiTest<int, int64_t, std::vector<int64_t>>();
    multiTest<int, int64_t, std::vector<float>>();
    multiTest<int, int64_t, std::vector<double>>();
    multiTest<int, int64_t, tensor>();
    multiTest<int, bool, bool>();
    multiTest<int, bool, float>();
    multiTest<int, bool, double>();
    multiTest<int, bool, std::vector<std::string>>();
    multiTest<int, bool, std::vector<int>>();
    multiTest<int, bool, std::vector<int64_t>>();
    multiTest<int, bool, std::vector<float>>();
    multiTest<int, bool, std::vector<double>>();
    multiTest<int, bool, tensor>();
    multiTest<int, float, float>();
    multiTest<int, float, double>();
    multiTest<int, float, std::vector<std::string>>();
    multiTest<int, float, std::vector<int>>();
    multiTest<int, float, std::vector<int64_t>>();
    multiTest<int, float, std::vector<float>>();
    multiTest<int, float, std::vector<double>>();
    multiTest<int, float, tensor>();
    multiTest<int, double, double>();
    multiTest<int, double, std::vector<std::string>>();
    multiTest<int, double, std::vector<int>>();
    multiTest<int, double, std::vector<int64_t>>();
    multiTest<int, double, std::vector<float>>();
    multiTest<int, double, std::vector<double>>();
    multiTest<int, double, tensor>();
    multiTest<int, std::vector<std::string>, std::vector<std::string>>();
    multiTest<int, std::vector<std::string>, std::vector<int>>();
    multiTest<int, std::vector<std::string>, std::vector<int64_t>>();
    multiTest<int, std::vector<std::string>, std::vector<float>>();
    multiTest<int, std::vector<std::string>, std::vector<double>>();
    multiTest<int, std::vector<std::string>, tensor>();
    multiTest<int, std::vector<int>, std::vector<int>>();
    multiTest<int, std::vector<int>, std::vector<int64_t>>();
    multiTest<int, std::vector<int>, std::vector<float>>();
    multiTest<int, std::vector<int>, std::vector<double>>();
    multiTest<int, std::vector<int>, tensor>();
    multiTest<int, std::vector<int64_t>, std::vector<int64_t>>();
    multiTest<int, std::vector<int64_t>, std::vector<float>>();
    multiTest<int, std::vector<int64_t>, std::vector<double>>();
    multiTest<int, std::vector<int64_t>, tensor>();
    multiTest<int, std::vector<float>, std::vector<float>>();
    multiTest<int, std::vector<float>, std::vector<double>>();
    multiTest<int, std::vector<float>, tensor>();
    multiTest<int, std::vector<double>, std::vector<double>>();
    multiTest<int, std::vector<double>, tensor>();
    multiTest<int, tensor, tensor>();
    multiTest<int64_t, int64_t, int64_t>();
    multiTest<int64_t, int64_t, bool>();
    multiTest<int64_t, int64_t, float>();
    multiTest<int64_t, int64_t, double>();
    multiTest<int64_t, int64_t, std::vector<std::string>>();
    multiTest<int64_t, int64_t, std::vector<int>>();
    multiTest<int64_t, int64_t, std::vector<int64_t>>();
    multiTest<int64_t, int64_t, std::vector<float>>();
    multiTest<int64_t, int64_t, std::vector<double>>();
    multiTest<int64_t, int64_t, tensor>();
    multiTest<int64_t, bool, bool>();
    multiTest<int64_t, bool, float>();
    multiTest<int64_t, bool, double>();
    multiTest<int64_t, bool, std::vector<std::string>>();
    multiTest<int64_t, bool, std::vector<int>>();
    multiTest<int64_t, bool, std::vector<int64_t>>();
    multiTest<int64_t, bool, std::vector<float>>();
    multiTest<int64_t, bool, std::vector<double>>();
    multiTest<int64_t, bool, tensor>();
    multiTest<int64_t, float, float>();
    multiTest<int64_t, float, double>();
    multiTest<int64_t, float, std::vector<std::string>>();
    multiTest<int64_t, float, std::vector<int>>();
    multiTest<int64_t, float, std::vector<int64_t>>();
    multiTest<int64_t, float, std::vector<float>>();
    multiTest<int64_t, float, std::vector<double>>();
    multiTest<int64_t, float, tensor>();
    multiTest<int64_t, double, double>();
    multiTest<int64_t, double, std::vector<std::string>>();
    multiTest<int64_t, double, std::vector<int>>();
    multiTest<int64_t, double, std::vector<int64_t>>();
    multiTest<int64_t, double, std::vector<float>>();
    multiTest<int64_t, double, std::vector<double>>();
    multiTest<int64_t, double, tensor>();
    multiTest<int64_t, std::vector<std::string>, std::vector<std::string>>();
    multiTest<int64_t, std::vector<std::string>, std::vector<int>>();
    multiTest<int64_t, std::vector<std::string>, std::vector<int64_t>>();
    multiTest<int64_t, std::vector<std::string>, std::vector<float>>();
    multiTest<int64_t, std::vector<std::string>, std::vector<double>>();
    multiTest<int64_t, std::vector<std::string>, tensor>();
    multiTest<int64_t, std::vector<int>, std::vector<int>>();
    multiTest<int64_t, std::vector<int>, std::vector<int64_t>>();
    multiTest<int64_t, std::vector<int>, std::vector<float>>();
    multiTest<int64_t, std::vector<int>, std::vector<double>>();
    multiTest<int64_t, std::vector<int>, tensor>();
    multiTest<int64_t, std::vector<int64_t>, std::vector<int64_t>>();
    multiTest<int64_t, std::vector<int64_t>, std::vector<float>>();
    multiTest<int64_t, std::vector<int64_t>, std::vector<double>>();
    multiTest<int64_t, std::vector<int64_t>, tensor>();
    multiTest<int64_t, std::vector<float>, std::vector<float>>();
    multiTest<int64_t, std::vector<float>, std::vector<double>>();
    multiTest<int64_t, std::vector<float>, tensor>();
    multiTest<int64_t, std::vector<double>, std::vector<double>>();
    multiTest<int64_t, std::vector<double>, tensor>();
    multiTest<int64_t, tensor, tensor>();
    multiTest<bool, bool, bool>();
    multiTest<bool, bool, float>();
    multiTest<bool, bool, double>();
    multiTest<bool, bool, std::vector<std::string>>();
    multiTest<bool, bool, std::vector<int>>();
    multiTest<bool, bool, std::vector<int64_t>>();
    multiTest<bool, bool, std::vector<float>>();
    multiTest<bool, bool, std::vector<double>>();
    multiTest<bool, bool, tensor>();
    multiTest<bool, float, float>();
    multiTest<bool, float, double>();
    multiTest<bool, float, std::vector<std::string>>();
    multiTest<bool, float, std::vector<int>>();
    multiTest<bool, float, std::vector<int64_t>>();
    multiTest<bool, float, std::vector<float>>();
    multiTest<bool, float, std::vector<double>>();
    multiTest<bool, float, tensor>();
    multiTest<bool, double, double>();
    multiTest<bool, double, std::vector<std::string>>();
    multiTest<bool, double, std::vector<int>>();
    multiTest<bool, double, std::vector<int64_t>>();
    multiTest<bool, double, std::vector<float>>();
    multiTest<bool, double, std::vector<double>>();
    multiTest<bool, double, tensor>();
    multiTest<bool, std::vector<std::string>, std::vector<std::string>>();
    multiTest<bool, std::vector<std::string>, std::vector<int>>();
    multiTest<bool, std::vector<std::string>, std::vector<int64_t>>();
    multiTest<bool, std::vector<std::string>, std::vector<float>>();
    multiTest<bool, std::vector<std::string>, std::vector<double>>();
    multiTest<bool, std::vector<std::string>, tensor>();
    multiTest<bool, std::vector<int>, std::vector<int>>();
    multiTest<bool, std::vector<int>, std::vector<int64_t>>();
    multiTest<bool, std::vector<int>, std::vector<float>>();
    multiTest<bool, std::vector<int>, std::vector<double>>();
    multiTest<bool, std::vector<int>, tensor>();
    multiTest<bool, std::vector<int64_t>, std::vector<int64_t>>();
    multiTest<bool, std::vector<int64_t>, std::vector<float>>();
    multiTest<bool, std::vector<int64_t>, std::vector<double>>();
    multiTest<bool, std::vector<int64_t>, tensor>();
    multiTest<bool, std::vector<float>, std::vector<float>>();
    multiTest<bool, std::vector<float>, std::vector<double>>();
    multiTest<bool, std::vector<float>, tensor>();
    multiTest<bool, std::vector<double>, std::vector<double>>();
    multiTest<bool, std::vector<double>, tensor>();
    multiTest<bool, tensor, tensor>();
    multiTest<float, float, float>();
    multiTest<float, float, double>();
    multiTest<float, float, std::vector<std::string>>();
    multiTest<float, float, std::vector<int>>();
    multiTest<float, float, std::vector<int64_t>>();
    multiTest<float, float, std::vector<float>>();
    multiTest<float, float, std::vector<double>>();
    multiTest<float, float, tensor>();
    multiTest<float, double, double>();
    multiTest<float, double, std::vector<std::string>>();
    multiTest<float, double, std::vector<int>>();
    multiTest<float, double, std::vector<int64_t>>();
    multiTest<float, double, std::vector<float>>();
    multiTest<float, double, std::vector<double>>();
    multiTest<float, double, tensor>();
    multiTest<float, std::vector<std::string>, std::vector<std::string>>();
    multiTest<float, std::vector<std::string>, std::vector<int>>();
    multiTest<float, std::vector<std::string>, std::vector<int64_t>>();
    multiTest<float, std::vector<std::string>, std::vector<float>>();
    multiTest<float, std::vector<std::string>, std::vector<double>>();
    multiTest<float, std::vector<std::string>, tensor>();
    multiTest<float, std::vector<int>, std::vector<int>>();
    multiTest<float, std::vector<int>, std::vector<int64_t>>();
    multiTest<float, std::vector<int>, std::vector<float>>();
    multiTest<float, std::vector<int>, std::vector<double>>();
    multiTest<float, std::vector<int>, tensor>();
    multiTest<float, std::vector<int64_t>, std::vector<int64_t>>();
    multiTest<float, std::vector<int64_t>, std::vector<float>>();
    multiTest<float, std::vector<int64_t>, std::vector<double>>();
    multiTest<float, std::vector<int64_t>, tensor>();
    multiTest<float, std::vector<float>, std::vector<float>>();
    multiTest<float, std::vector<float>, std::vector<double>>();
    multiTest<float, std::vector<float>, tensor>();
    multiTest<float, std::vector<double>, std::vector<double>>();
    multiTest<float, std::vector<double>, tensor>();
    multiTest<float, tensor, tensor>();
    multiTest<double, double, double>();
    multiTest<double, double, std::vector<std::string>>();
    multiTest<double, double, std::vector<int>>();
    multiTest<double, double, std::vector<int64_t>>();
    multiTest<double, double, std::vector<float>>();
    multiTest<double, double, std::vector<double>>();
    multiTest<double, double, tensor>();
    multiTest<double, std::vector<std::string>, std::vector<std::string>>();
    multiTest<double, std::vector<std::string>, std::vector<int>>();
    multiTest<double, std::vector<std::string>, std::vector<int64_t>>();
    multiTest<double, std::vector<std::string>, std::vector<float>>();
    multiTest<double, std::vector<std::string>, std::vector<double>>();
    multiTest<double, std::vector<std::string>, tensor>();
    multiTest<double, std::vector<int>, std::vector<int>>();
    multiTest<double, std::vector<int>, std::vector<int64_t>>();
    multiTest<double, std::vector<int>, std::vector<float>>();
    multiTest<double, std::vector<int>, std::vector<double>>();
    multiTest<double, std::vector<int>, tensor>();
    multiTest<double, std::vector<int64_t>, std::vector<int64_t>>();
    multiTest<double, std::vector<int64_t>, std::vector<float>>();
    multiTest<double, std::vector<int64_t>, std::vector<double>>();
    multiTest<double, std::vector<int64_t>, tensor>();
    multiTest<double, std::vector<float>, std::vector<float>>();
    multiTest<double, std::vector<float>, std::vector<double>>();
    multiTest<double, std::vector<float>, tensor>();
    multiTest<double, std::vector<double>, std::vector<double>>();
    multiTest<double, std::vector<double>, tensor>();
    multiTest<double, tensor, tensor>();
    multiTest<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>>();
    multiTest<std::vector<std::string>, std::vector<std::string>, std::vector<int>>();
    multiTest<std::vector<std::string>, std::vector<std::string>, std::vector<int64_t>>();
    multiTest<std::vector<std::string>, std::vector<std::string>, std::vector<float>>();
    multiTest<std::vector<std::string>, std::vector<std::string>, std::vector<double>>();
    multiTest<std::vector<std::string>, std::vector<std::string>, tensor>();
    multiTest<std::vector<std::string>, std::vector<int>, std::vector<int>>();
    multiTest<std::vector<std::string>, std::vector<int>, std::vector<int64_t>>();
    multiTest<std::vector<std::string>, std::vector<int>, std::vector<float>>();
    multiTest<std::vector<std::string>, std::vector<int>, std::vector<double>>();
    multiTest<std::vector<std::string>, std::vector<int>, tensor>();
    multiTest<std::vector<std::string>, std::vector<int64_t>, std::vector<int64_t>>();
    multiTest<std::vector<std::string>, std::vector<int64_t>, std::vector<float>>();
    multiTest<std::vector<std::string>, std::vector<int64_t>, std::vector<double>>();
    multiTest<std::vector<std::string>, std::vector<int64_t>, tensor>();
    multiTest<std::vector<std::string>, std::vector<float>, std::vector<float>>();
    multiTest<std::vector<std::string>, std::vector<float>, std::vector<double>>();
    multiTest<std::vector<std::string>, std::vector<float>, tensor>();
    multiTest<std::vector<std::string>, std::vector<double>, std::vector<double>>();
    multiTest<std::vector<std::string>, std::vector<double>, tensor>();
    multiTest<std::vector<std::string>, tensor, tensor>();
    multiTest<std::vector<int>, std::vector<int>, std::vector<int>>();
    multiTest<std::vector<int>, std::vector<int>, std::vector<int64_t>>();
    multiTest<std::vector<int>, std::vector<int>, std::vector<float>>();
    multiTest<std::vector<int>, std::vector<int>, std::vector<double>>();
    multiTest<std::vector<int>, std::vector<int>, tensor>();
    multiTest<std::vector<int>, std::vector<int64_t>, std::vector<int64_t>>();
    multiTest<std::vector<int>, std::vector<int64_t>, std::vector<float>>();
    multiTest<std::vector<int>, std::vector<int64_t>, std::vector<double>>();
    multiTest<std::vector<int>, std::vector<int64_t>, tensor>();
    multiTest<std::vector<int>, std::vector<float>, std::vector<float>>();
    multiTest<std::vector<int>, std::vector<float>, std::vector<double>>();
    multiTest<std::vector<int>, std::vector<float>, tensor>();
    multiTest<std::vector<int>, std::vector<double>, std::vector<double>>();
    multiTest<std::vector<int>, std::vector<double>, tensor>();
    multiTest<std::vector<int>, tensor, tensor>();
    multiTest<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>();
    multiTest<std::vector<int64_t>, std::vector<int64_t>, std::vector<float>>();
    multiTest<std::vector<int64_t>, std::vector<int64_t>, std::vector<double>>();
    multiTest<std::vector<int64_t>, std::vector<int64_t>, tensor>();
    multiTest<std::vector<int64_t>, std::vector<float>, std::vector<float>>();
    multiTest<std::vector<int64_t>, std::vector<float>, std::vector<double>>();
    multiTest<std::vector<int64_t>, std::vector<float>, tensor>();
    multiTest<std::vector<int64_t>, std::vector<double>, std::vector<double>>();
    multiTest<std::vector<int64_t>, std::vector<double>, tensor>();
    multiTest<std::vector<int64_t>, tensor, tensor>();
    multiTest<std::vector<float>, std::vector<float>, std::vector<float>>();
    multiTest<std::vector<float>, std::vector<float>, std::vector<double>>();
    multiTest<std::vector<float>, std::vector<float>, tensor>();
    multiTest<std::vector<float>, std::vector<double>, std::vector<double>>();
    multiTest<std::vector<float>, std::vector<double>, tensor>();
    multiTest<std::vector<float>, tensor, tensor>();
    multiTest<std::vector<double>, std::vector<double>, std::vector<double>>();
    multiTest<std::vector<double>, std::vector<double>, tensor>();
    multiTest<std::vector<double>, tensor, tensor>();
    multiTest<tensor, tensor, tensor>();
}

} // namespace
} // namespace gmx
