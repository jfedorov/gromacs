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
 * \brief
 * Tests for the t_state class.
 *
 * \author berk Hess <hess@kth.se>
 * \ingroup module_mdtypes
 */
#include "gmxpre.h"

#include "gromacs/mdtypes/state.h"

#include <array>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "testutils/testasserts.h"
#include "testutils/testmatchers.h"

namespace gmx
{

const std::array<RVec, 2> c_forces = { { { 0.5, 0.1, 1.2 }, { -2.1, 0.2, 0.3 } } };

TEST(StateRVecVectors, AddsRVecVector)
{
    t_state   state;
    const int numAtoms = 2;

    state.changeNumAtoms(numAtoms);

    const std::string testString = "testString";

    const gmx::ArrayRef<const gmx::RVec> arrayRef = state.addRVecVector(testString);

    EXPECT_EQ(state.rvecVectors().size(), 1);

    EXPECT_EQ(arrayRef.size(), numAtoms);

    for (const gmx::RVec& elem : arrayRef)
    {
        // Note that EXPECT_EQ does not seem to work for RVec
        for (int d = 0; d < DIM; d++)
        {
            EXPECT_EQ(elem[d], 0.0_real);
        }
    }
}

TEST(StateRVecVectors, AddSameNameThrows)
{
    t_state   state;
    const int numAtoms = 0;

    state.changeNumAtoms(numAtoms);

    const std::string testString = "testString";

    state.addRVecVector(testString);

    EXPECT_EQ(state.rvecVectors().size(), 1);

    EXPECT_THROW_GMX(state.addRVecVector(testString), InvalidInputError);
}

TEST(StateRVecVectors, ArrayRefReferenceRemainsValid)
{
    t_state   state;
    const int numAtoms = 3;

    state.changeNumAtoms(numAtoms);

    const std::string         testString1 = "testString1";
    gmx::ArrayRef<gmx::RVec>& arrayRefRef = state.addRVecVector(testString1);

    EXPECT_EQ(state.rvecVectors().size(), 1);

    ASSERT_EQ(ssize(arrayRefRef), numAtoms);

    const int       testIndex = 1;
    const gmx::RVec testValue = { 3, 1, 2 };
    arrayRefRef[testIndex]    = testValue;

    const std::string testString2 = "testString2";
    state.addRVecVector(testString2);

    EXPECT_EQ(state.rvecVectors().size(), 2);

    ASSERT_EQ(ssize(arrayRefRef), numAtoms);

    // Note that EXPECT_EQ does not seem to work for RVec
    for (int d = 0; d < DIM; d++)
    {
        EXPECT_EQ(arrayRefRef[testIndex][d], testValue[d]);
    }
}

TEST(StateRVecVectors, RetrievesRVecVector)
{
    t_state   state;
    const int numAtoms = 3;

    state.changeNumAtoms(numAtoms);

    const std::string testString1 = "testString1";
    state.addRVecVector(testString1);
    const std::string testString2 = "testString2";
    auto&             arrayRefRef = state.addRVecVector(testString2);
    const std::string testString3 = "testString3";
    state.addRVecVector(testString3);

    EXPECT_EQ(state.rvecVectors().size(), 3);

    const int       testIndex = 1;
    const gmx::RVec testValue = { 3, 1, 2 };
    arrayRefRef[testIndex]    = testValue;

    auto req = state.rvecVector(testString2);
    ASSERT_EQ(req.has_value(), true);
    gmx::ArrayRef<const gmx::RVec> vec = req.value();

    // Note that EXPECT_EQ does not seem to work for RVec
    for (int d = 0; d < DIM; d++)
    {
        EXPECT_EQ(vec[testIndex][d], testValue[d]);
    }
}

// Checks that two state object with the same vectors added in the same order
// iterate over all vectors in the same order (as would happen with MPI in mdrun)
TEST(StateRVecVectors, HaveIndenticalOrder)
{
    constexpr int                          numStates = 2;
    std::array<t_state, numStates>         states;
    const int                              numAtoms = 3;
    constexpr int                          numVecs  = 3;
    const std::array<std::string, numVecs> strings = { "testString1", "testString3", "testString2" };

    for (t_state& state : states)
    {
        state.changeNumAtoms(numAtoms);
    }

    // Add the vectors to both states and fill the vectors with different values
    int value = 0;
    for (const std::string& string : strings)
    {
        for (t_state& state : states)
        {
            auto ref = state.addRVecVector(string);
            for (auto& elem : ref)
            {
                elem = { real(value), 0.0_real, 0.0_real };
            }
            value += 1;
        }
    }

    // Collect array of ArrayRefs which are expected to be in the same order for both states
    std::array<std::vector<ArrayRef<RVec>>, numStates> arrayOfRefs;
    int                                                stateIndex = 0;
    for (t_state& state : states)
    {
        auto& refs = arrayOfRefs[stateIndex];
        for (auto& ref : state.rvecVectors())
        {
            refs.push_back(ref);
        }
        ASSERT_EQ(refs.size(), numVecs);
        stateIndex++;
    }
    // Swap the contents of the vectors with the same array index
    for (int v = 0; v < numVecs; v++)
    {
        ASSERT_EQ(arrayOfRefs[1][v].size(), arrayOfRefs[0][v].size());
        for (index i = 0; i < ssize(arrayOfRefs[0][v]); i++)
        {
            std::swap(arrayOfRefs[0][v][i], arrayOfRefs[1][v][i]);
        }
    }

    // Check that the values matched the expected swapped values
    value = 0;
    for (const std::string& string : strings)
    {
        // Reversed loop to match the swap
        for (auto stateIt = states.rbegin(); stateIt < states.rend(); stateIt++)
        {
            auto ref = stateIt->rvecVector(string).value();
            for (auto& elem : ref)
            {
                EXPECT_EQ(elem[0], real(value));
            }
            value += 1;
        }
    }
}

} // namespace gmx
