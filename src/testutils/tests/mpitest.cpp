/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2019,2021, by the GROMACS development team, led by
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
 * Tests for infrastructure for running tests under MPI.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include "testutils/mpitest.h"

#include "config.h"

#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/gmxmpi.h"

namespace gmx
{
namespace test
{
namespace
{

class MpiSelfTest : public MpiTest
{
public:
    MpiSelfTest() : reached_(numRanks_, 0) {}
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int /*numRanks*/) { return true; }
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
    //! Whether each rank participated
    std::vector<int> reached_;
};

class Runs : public MpiSelfTest
{
public:
    //! Body of the test
    void TestBody() override;
};

void Runs::TestBody()
{
    GMX_MPI_TEST(numRanks_);
#if GMX_THREAD_MPI
    reached_[gmx_node_rank()] = 1;
    MPI_Barrier(MPI_COMM_WORLD);
#else
    int value = 1;
    MPI_Gather(&value, 1, MPI_INT, reached_.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    if (gmx_node_rank() == 0)
    {
        EXPECT_THAT(reached_, testing::Each(1));
    }
}

} // namespace

void registerMpiTests(int numRanks)
{
    MpiTest::tryToRegisterTest<MpiSelfTest, Runs>(numRanks, "MpiSelfTest", "Runs");
}

} // namespace test
} // namespace gmx
