/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017,2019,2020,2021, by the GROMACS development team, led by
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

#include <array>

#include <gtest/gtest.h>

#include "gromacs/utility/basenetwork.h"

#include "testutils/mpitest.h"

#include "threadaffinitytest.h"

namespace gmx
{
namespace test
{
namespace
{

//! MPI test case class
class PinsWholeNode : public MpiTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool PinsWholeNode::canRun(int /*numRanks*/)
{
    return true;
}

void PinsWholeNode::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    helper.setLogicalProcessorCount(numRanks_);
    helper.expectPinningMessage(false, 1);
    helper.expectAffinitySet(gmx_node_rank());
    helper.setAffinity(1);
}

//! MPI test case class
class PinsWithOffsetAndStride : public MpiTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool PinsWithOffsetAndStride::canRun(int /*numRanks*/)
{
    return true;
}

void PinsWithOffsetAndStride::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    helper.setAffinityOption(ThreadAffinity::On);
    helper.setOffsetAndStride(1, 2);
    helper.setLogicalProcessorCount(2 * numRanks_);
    helper.expectWarningMatchingRegex("Applying core pinning offset 1");
    helper.expectPinningMessage(true, 2);
    helper.expectAffinitySet(1 + 2 * gmx_node_rank());
    helper.setAffinity(1);
}

//! MPI test case class
class PinsTwoNodes : public MpiTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool PinsTwoNodes::canRun(int numRanks)
{
    return numRanks % 2 == 0;
}

void PinsTwoNodes::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    helper.setPhysicalNodeId(gmx_node_rank() / 2);
    helper.setLogicalProcessorCount(2);
    helper.expectPinningMessage(false, 1);
    helper.expectAffinitySet(gmx_node_rank() % 2);
    helper.setAffinity(1);
}

//! MPI test case class
class DoesNothingWhenDisabled : public MpiTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool DoesNothingWhenDisabled::canRun(int /*numRanks*/)
{
    return true;
}

void DoesNothingWhenDisabled::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    helper.setAffinityOption(ThreadAffinity::Off);
    helper.setLogicalProcessorCount(numRanks_);
    helper.setAffinity(1);
}

//! MPI test case class
class HandlesTooManyThreadsWithAuto : public MpiTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool HandlesTooManyThreadsWithAuto::canRun(int /*numRanks*/)
{
    return true;
}

void HandlesTooManyThreadsWithAuto::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    const int                threadsPerRank = 2;
    helper.setLogicalProcessorCount(threadsPerRank * numRanks_ - 1);
    helper.expectWarningMatchingRegex("Oversubscribing the CPU");
    helper.setAffinity(threadsPerRank);
}

//! MPI test case class
class HandlesTooManyThreadsWithForce : public MpiTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool HandlesTooManyThreadsWithForce::canRun(int /*numRanks*/)
{
    return true;
}

void HandlesTooManyThreadsWithForce::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    const int                threadsPerRank = 2;
    helper.setAffinityOption(ThreadAffinity::On);
    helper.setLogicalProcessorCount(threadsPerRank * numRanks_ - 1);
    helper.expectWarningMatchingRegex("Oversubscribing the CPU");
    helper.setAffinity(threadsPerRank);
}

class ThreadAffinityHeterogeneousNodesTest : public MpiTest
{
public:
    static int  currentNode() { return gmx_node_rank() / 2; }
    static int  indexInNode() { return gmx_node_rank() % 2; }
    static bool isMaster() { return gmx_node_rank() == 0; }

    static void setupNodes(ThreadAffinityTestHelper* helper, int coresOnNodeZero, int coresOnOtherNodes)
    {
        const int node = currentNode();
        helper->setPhysicalNodeId(node);
        helper->setLogicalProcessorCount(node == 0 ? coresOnNodeZero : coresOnOtherNodes);
    }
    static void expectNodeAffinitySet(ThreadAffinityTestHelper* helper, int node, int core)
    {
        if (currentNode() == node)
        {
            helper->expectAffinitySet(core);
        }
    }
};

//! MPI test case class
class PinsOnMasterOnly : public ThreadAffinityHeterogeneousNodesTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool PinsOnMasterOnly::canRun(int numRanks)
{
    return (numRanks > 2) && (numRanks % 2 == 0);
}

void PinsOnMasterOnly::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    helper.setAffinityOption(ThreadAffinity::On);
    setupNodes(&helper, 2, 1);
    helper.expectWarningMatchingRegexIf("Oversubscribing the CPU", isMaster() || currentNode() == 1);
    if (currentNode() == 0)
    {
        helper.expectPinningMessage(false, 1);
    }
    expectNodeAffinitySet(&helper, 0, indexInNode());
    helper.setAffinity(1);
}

//! MPI test case class
class PinsOnNonMasterOnly : public ThreadAffinityHeterogeneousNodesTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool PinsOnNonMasterOnly::canRun(int numRanks)
{
    return (numRanks > 2) && (numRanks % 2 == 0);
}

void PinsOnNonMasterOnly::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    helper.setAffinityOption(ThreadAffinity::On);
    setupNodes(&helper, 1, 2);
    helper.expectWarningMatchingRegexIf("Oversubscribing the CPU", currentNode() == 0);
    if (currentNode() >= 1)
    {
        helper.expectPinningMessage(false, 1);
        expectNodeAffinitySet(&helper, currentNode(), indexInNode());
    }
    helper.setAffinity(1);
}

//! MPI test case class
class HandlesUnknownHardwareOnNonMaster : public ThreadAffinityHeterogeneousNodesTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool HandlesUnknownHardwareOnNonMaster::canRun(int numRanks)
{
    return (numRanks > 2) && (numRanks % 2 == 0);
}

void HandlesUnknownHardwareOnNonMaster::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    helper.setAffinityOption(ThreadAffinity::On);
    setupNodes(&helper, 2, 0);
    helper.expectWarningMatchingRegexIf("No information on available cores",
                                        isMaster() || currentNode() == 1);
    if (currentNode() == 0)
    {
        helper.expectPinningMessage(false, 1);
    }
    expectNodeAffinitySet(&helper, 0, indexInNode());
    helper.setAffinity(1);
}

//! MPI test case class
class PinsAutomaticallyOnMasterOnly : public ThreadAffinityHeterogeneousNodesTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool PinsAutomaticallyOnMasterOnly::canRun(int numRanks)
{
    return (numRanks > 2) && (numRanks % 2 == 0);
}

void PinsAutomaticallyOnMasterOnly::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    setupNodes(&helper, 2, 1);
    helper.expectWarningMatchingRegexIf("Oversubscribing the CPU", isMaster() || currentNode() == 1);
    if (currentNode() == 0)
    {
        helper.expectPinningMessage(false, 1);
    }
    expectNodeAffinitySet(&helper, 0, indexInNode());
    helper.setAffinity(1);
}

//! MPI test case class
class PinsAutomaticallyOnNonMasterOnly : public ThreadAffinityHeterogeneousNodesTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool PinsAutomaticallyOnNonMasterOnly::canRun(int numRanks)
{
    return (numRanks > 2) && (numRanks % 2 == 0);
}

void PinsAutomaticallyOnNonMasterOnly::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    setupNodes(&helper, 1, 2);
    helper.expectWarningMatchingRegexIf("Oversubscribing the CPU", currentNode() == 0);
    if (currentNode() >= 1)
    {
        helper.expectPinningMessage(false, 1);
        expectNodeAffinitySet(&helper, currentNode(), indexInNode());
    }
    helper.setAffinity(1);
}

//! MPI test case class
class HandlesInvalidOffsetOnNonMasterOnly : public ThreadAffinityHeterogeneousNodesTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool HandlesInvalidOffsetOnNonMasterOnly::canRun(int numRanks)
{
    return (numRanks > 2) && (numRanks % 2 == 0);
}

void HandlesInvalidOffsetOnNonMasterOnly::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    helper.setAffinityOption(ThreadAffinity::On);
    helper.setOffsetAndStride(2, 0);
    setupNodes(&helper, 4, 2);
    helper.expectWarningMatchingRegex("Applying core pinning offset 2");
    helper.expectWarningMatchingRegexIf("Requested offset too large", isMaster() || currentNode() >= 1);
    if (currentNode() == 0)
    {
        helper.expectPinningMessage(false, 1);
    }
    expectNodeAffinitySet(&helper, 0, indexInNode() + 2);
    helper.setAffinity(1);
}

//! MPI test case class
class HandlesInvalidStrideOnNonMasterOnly : public ThreadAffinityHeterogeneousNodesTest
{
public:
    //! Body of the test
    void TestBody() override;
    //! Whether this test case can run with the current number of MPI ranks
    static bool canRun(int numRanks);
    //! The MPI rank count
    int numRanks_ = getNumberOfTestMpiRanks();
};

bool HandlesInvalidStrideOnNonMasterOnly::canRun(int numRanks)
{
    return (numRanks > 2) && (numRanks % 2 == 0);
}

void HandlesInvalidStrideOnNonMasterOnly::TestBody()
{
    GMX_MPI_TEST(numRanks_);
    ThreadAffinityTestHelper helper;
    helper.setAffinityOption(ThreadAffinity::On);
    helper.setOffsetAndStride(0, 2);
    setupNodes(&helper, 4, 2);
    helper.expectWarningMatchingRegexIf("Requested stride too large", isMaster() || currentNode() == 1);
    if (currentNode() == 0)
    {
        helper.expectPinningMessage(true, 2);
    }
    expectNodeAffinitySet(&helper, 0, 2 * indexInNode());
    helper.setAffinity(1);
}

} // namespace

void registerMpiTests(int numRanks)
{
    // ThreadAffinityMultiRankTest cases
    MpiTest::tryToRegisterTest<MpiTest, PinsWholeNode>(
            numRanks, "ThreadAffinityMultiRankTest", "PinsWholeNode");
    MpiTest::tryToRegisterTest<MpiTest, PinsWithOffsetAndStride>(
            numRanks, "ThreadAffinityMultiRankTest", "PinsWithOffsetAndStride");
    MpiTest::tryToRegisterTest<MpiTest, PinsTwoNodes>(
            numRanks, "ThreadAffinityMultiRankTest", "PinsTwoNodes");
    MpiTest::tryToRegisterTest<MpiTest, DoesNothingWhenDisabled>(
            numRanks, "ThreadAffinityMultiRankTest", "DoesNothingWhenDisabled");
    MpiTest::tryToRegisterTest<MpiTest, HandlesTooManyThreadsWithAuto>(
            numRanks, "ThreadAffinityMultiRankTest", "HandlesTooManyThreadsWithAuto");
    MpiTest::tryToRegisterTest<MpiTest, HandlesTooManyThreadsWithForce>(
            numRanks, "ThreadAffinityMultiRankTest", "HandlesTooManyThreadsWithForce");

    // ThreadAffinityHeterogeneousNodesTest cases
    MpiTest::tryToRegisterTest<ThreadAffinityHeterogeneousNodesTest, PinsOnMasterOnly>(
            numRanks, "ThreadAffinityHeterogeneousNodesTest", "PinsOnMasterOnly");
    MpiTest::tryToRegisterTest<ThreadAffinityHeterogeneousNodesTest, PinsOnNonMasterOnly>(
            numRanks, "ThreadAffinityHeterogeneousNodesTest", "PinsOnNonMasterOnly");
    MpiTest::tryToRegisterTest<ThreadAffinityHeterogeneousNodesTest, HandlesUnknownHardwareOnNonMaster>(
            numRanks, "ThreadAffinityHeterogeneousNodesTest", "HandlesUnknownHardwareOnNonMaster");
    MpiTest::tryToRegisterTest<ThreadAffinityHeterogeneousNodesTest, PinsAutomaticallyOnMasterOnly>(
            numRanks, "ThreadAffinityHeterogeneousNodesTest", "PinsAutomaticallyOnMasterOnly");
    MpiTest::tryToRegisterTest<ThreadAffinityHeterogeneousNodesTest, PinsAutomaticallyOnNonMasterOnly>(
            numRanks, "ThreadAffinityHeterogeneousNodesTest", "PinsAutomaticallyOnNonMasterOnly");
    MpiTest::tryToRegisterTest<ThreadAffinityHeterogeneousNodesTest, HandlesInvalidOffsetOnNonMasterOnly>(
            numRanks, "ThreadAffinityHeterogeneousNodesTest", "HandlesInvalidOffsetOnNonMasterOnly");
    MpiTest::tryToRegisterTest<ThreadAffinityHeterogeneousNodesTest, HandlesInvalidStrideOnNonMasterOnly>(
            numRanks, "ThreadAffinityHeterogeneousNodesTest", "HandlesInvalidStrideOnNonMasterOnly");
}

} // namespace test
} // namespace gmx
