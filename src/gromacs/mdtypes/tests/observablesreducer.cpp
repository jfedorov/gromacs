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
 * Tests for ObservablesReducer.
 *
 * \ingroup module_mdtypes
 */
#include "gmxpre.h"

#include "gromacs/mdtypes/observablesreducer.h"

#include <numeric>
#include <optional>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/stringutil.h"

namespace gmx::test
{
namespace
{

// Unit tests of ObservablesReducer. Note that these do not *require*
// using the builder.

TEST(ObservablesReducerTest, CanDefaultConstruct)
{
    ObservablesReducer observablesReducer;
    EXPECT_TRUE(observablesReducer.communicationBuffer().empty())
            << "no buffer available when no subscribers requested reduction";
}

TEST(ObservablesReducerTest, MoveOperationsWork)
{
    ObservablesReducer observablesReducer(ObservablesReducer{});
    observablesReducer = ObservablesReducer{};
}

TEST(ObservablesReducerTest, CanConstructFromVector)
{
    std::vector<double> communicationBuffer(3, -1.0);

    ObservablesReducer observablesReducer(std::move(communicationBuffer), {});
    EXPECT_TRUE(observablesReducer.communicationBuffer().empty())
            << "no buffer available when no subscribers requested reduction";
}

TEST(ObservablesReducerTest, CanConstructFromVectorAndUse)
{
    std::vector<double> communicationBuffer(3, -1.0);
    ArrayRef<double>    bufferView(communicationBuffer);

    ObservablesReducer observablesReducer(std::move(communicationBuffer), {});
    EXPECT_TRUE(observablesReducer.communicationBuffer().empty())
            << "no buffer available when no subscribers requested reduction";
    observablesReducer.reductionComplete(0);

    EXPECT_TRUE(observablesReducer.communicationBuffer().empty())
            << "no buffer available after reductionComplete()";
    EXPECT_THAT(bufferView, testing::AllOf(testing::SizeIs(3), testing::Each(0.0)));
}

TEST(ObservablesReducerTest, CanBuildAndUse)
{
    std::vector<double> communicationBuffer(3, -1.0);
    ArrayRef<double>    bufferView(communicationBuffer);

    ObservablesReducer observablesReducer(std::move(communicationBuffer), {});
    EXPECT_TRUE(observablesReducer.communicationBuffer().empty())
            << "no buffer available when no subscribers requested reduction";
    observablesReducer.reductionComplete(0);

    EXPECT_TRUE(observablesReducer.communicationBuffer().empty())
            << "no buffer available after reductionComplete()";
    EXPECT_THAT(bufferView, testing::AllOf(testing::SizeIs(3), testing::Each(0.0)));
}

// Unit tests of ObservablesReducerBuilder.

TEST(ObservablesReducerTest, CanBuildAndUseWithNoSubscribers)
{
    ObservablesReducerBuilder builder;

    ObservablesReducer observablesReducer = builder.build();
    EXPECT_TRUE(observablesReducer.communicationBuffer().empty())
            << "no buffer available when no subscribers requested reduction";
}

TEST(ObservablesReducerTest, CanBuildAndUseWithOneSubscriber)
{
    ObservablesReducerBuilder builder;

    // This test implements the caller, the builder and the
    // ObservablesReducer all in the one scope, which likely does not
    // resemble any actual use case. Those are tested in the
    // integration tests below.

    std::optional<int>                                    stepUponWhichReductionOccured;
    ObservablesReducerBuilder::CallbackToRequireReduction callbackToRequireReduction;
    ArrayRef<double>                                      bufferView;
    ObservablesReducerBuilder::CallbackFromBuilder        callbackFromBuilder =
            [&](ObservablesReducerBuilder::CallbackToRequireReduction&& c, ArrayRef<double> b) {
                callbackToRequireReduction = std::move(c);
                bufferView                 = b;
            };

    ObservablesReducerBuilder::CallbackAfterReduction callbackAfterReduction =
            [&stepUponWhichReductionOccured](Step step) { stepUponWhichReductionOccured = step; };
    const int requiredBufferSize = 2;
    builder.addSubscriber(
            requiredBufferSize, std::move(callbackFromBuilder), std::move(callbackAfterReduction));

    ObservablesReducer observablesReducer = builder.build();
    EXPECT_TRUE(observablesReducer.communicationBuffer().empty())
            << "no buffer available when no subscribers requested reduction";
    ASSERT_EQ(requiredBufferSize, bufferView.size());
    ASSERT_NE(callbackToRequireReduction, nullptr)
            << "must have valid callback supplied by the builder";
    EXPECT_FALSE(stepUponWhichReductionOccured.has_value())
            << "no callbacks until reductionComplete() is called";

    // Fill some dummy data, so we can check the clearing later
    bufferView[0] = 3.0;
    bufferView[1] = 4.0;

    {
        SCOPED_TRACE("Test that ReductionRequirement::Eventually doesn't trigger behavior");

        callbackToRequireReduction(ReductionRequirement::Eventually);
        EXPECT_TRUE(observablesReducer.communicationBuffer().empty())
                << "no buffer available when the only subscribers requested reduction eventually";
        EXPECT_FALSE(stepUponWhichReductionOccured.has_value())
                << "no callbacks until reductionComplete() is called";
    }
    {
        SCOPED_TRACE("Test that ReductionRequirement::Soon does trigger behavior");

        callbackToRequireReduction(ReductionRequirement::Soon);
        EXPECT_EQ(observablesReducer.communicationBuffer().size(), requiredBufferSize)
                << "buffer available when a subscriber requested reduction soon";
        EXPECT_FALSE(stepUponWhichReductionOccured.has_value())
                << "no callbacks until reductionComplete() is called";

        // In the intended use case, some external component must do the
        // actual reduction across ranks using the buffer at this point.

        int step = 2;
        observablesReducer.reductionComplete(step);
        ASSERT_TRUE(stepUponWhichReductionOccured.has_value()) << "reduction callbacks took place";
        EXPECT_EQ(stepUponWhichReductionOccured.value(), step)
                << "reduction step is passed through correctly";
        EXPECT_THAT(bufferView, testing::AllOf(testing::SizeIs(requiredBufferSize), testing::Each(0.0)));
    }
}

// Integration tests of ObservablesReducer, builder, and fake
// subscriber(s). These will model multiple ranks each with multiple
// subscribers.

//! Helper class that models an MD module that needs to make a subscription to \c ObservablesReducer
class Subscriber
{
public:
    //! Constructor
    Subscriber(int sizeRequired, double valueOffset, int numRanks) :
        sizeRequired_(sizeRequired), valueOffset_(valueOffset), numRanks_(numRanks)
    {
    }

    //! Make the subscription via the \c observablesReducerBuilder
    void makeSubscription(ObservablesReducerBuilder* observablesReducerBuilder)
    {
        observablesReducerBuilder->addSubscriber(
                sizeRequired_,
                [this](ObservablesReducerBuilder::CallbackToRequireReduction callback,
                       ArrayRef<double>                                      bufferView) {
                    this->callbackWhenBufferAvailable(std::move(callback), bufferView);
                },
                [this](Step step) { this->callbackAfterReduction(step); });
    }

    //! Callback to recieve the view of the communication buffer
    void callbackWhenBufferAvailable(ObservablesReducerBuilder::CallbackToRequireReduction callbackToRequireReduction,
                                     ArrayRef<double> bufferView)
    {
        SCOPED_TRACE("In callback from builder");

        callbackToRequireReduction_ = std::move(callbackToRequireReduction);
        communicationBuffer_        = bufferView;
        EXPECT_THAT(communicationBuffer_, testing::AllOf(testing::SizeIs(sizeRequired_), testing::Each(0.0)))
                << "size of buffer did not match request";
    }

    //! Pretend to do some simulation work characteristic of \c step
    void doSimulationWork(Step step) const
    {
        // Some imaginary real MD simulation for this step would go here.
        // ...
        // Then we put its intermediate output into the communication buffer.
        std::iota(communicationBuffer_.begin(), communicationBuffer_.end(), valueOffset_ + double(step));
        // Then we require reduction.
        callbackToRequireReduction_(ReductionRequirement::Soon);
    }

    //! After the reduction, check the values for this subscriber are as expected
    void callbackAfterReduction(Step step)
    {
        SCOPED_TRACE("In callback after reduction");

        // Expected values are different for each subscriber, and
        // vary with step and number of ranks.
        std::vector<double> expectedResult(communicationBuffer_.size());
        std::iota(expectedResult.begin(), expectedResult.end(), valueOffset_ + double(step));
        std::for_each(expectedResult.begin(), expectedResult.end(), [this](auto& v) {
            v *= this->numRanks_;
        });
        EXPECT_THAT(communicationBuffer_, testing::Pointwise(testing::Eq(), expectedResult))
                << "wrong values were reduced";
        EXPECT_THAT(communicationBuffer_, testing::Not(testing::Each(0)))
                << "zero may not be the result of an reduction during testing";
    }

    //! The number of doubles required to reduce
    int sizeRequired_;
    //! The callback used to require reduction
    ObservablesReducerBuilder::CallbackToRequireReduction callbackToRequireReduction_;
    //! The buffer used for communication, supplied by an \c ObservablesReducer
    ArrayRef<double> communicationBuffer_;
    //! Offset that differentiates the values reduced by each subscriber
    double valueOffset_;
    //! Number of ranks, used in constructing test expectations
    int numRanks_;
};

//! Test fixture class
class ObservablesReducerIntegrationTest : public testing::TestWithParam<std::tuple<int, int>>
{
public:
    //! Helper struct to model data on a single MPI rank
    struct RankData
    {
        //! Builder of \c observablesReducer
        ObservablesReducerBuilder builder;
        //! Subscribers to \c observablesReducer
        std::vector<Subscriber> subscribers;
        //! Manages reduction of observables on behalf of this "rank".
        ObservablesReducer observablesReducer;
    };

    //! Constructor
    ObservablesReducerIntegrationTest() : numSubscribers_(std::get<0>(GetParam()))
    {
        int numRanks(std::get<1>(GetParam()));
        // Ensure that each subscriber sends an interesting amount of data
        int subscriberBufferMinSize = 3;
        // Ensure the data reduced by each subscriber is
        // distinct, to help diagnose bugs. Also contributes to
        // ensuring that the reduced total is never zero.
        double subscriberOffset = 1000;

        rankData_.resize(numRanks);
        for (auto& rankData : rankData_)
        {
            for (int i = 0; i < numSubscribers_; ++i)
            {
                // Ensure each subscriber sends a different (but small) amount of data
                rankData.subscribers.emplace_back(
                        Subscriber(subscriberBufferMinSize + i, subscriberOffset, numRanks));
            }
            // Now that the addresses of the subscribers are
            // stable, set up the build-time callback.
            for (auto& subscriber : rankData.subscribers)
            {
                subscriber.makeSubscription(&rankData.builder);
            }
        }
    }

    //! Performs the equivalent of MPI_Allreduce over \c rankData_
    void fakeMpiAllReduce()
    {
        std::vector<double> reducedValues(
                rankData_[0].observablesReducer.communicationBuffer().size(), 0.0);
        // Reduce the values across "ranks"
        for (auto& rankData : rankData_)
        {
            for (size_t i = 0; i != reducedValues.size(); ++i)
            {
                reducedValues[i] += rankData.observablesReducer.communicationBuffer()[i];
            }
        }
        // Copy the reduced values to all "ranks"
        for (auto& rankData : rankData_)
        {
            auto buffer = rankData.observablesReducer.communicationBuffer();
            std::copy(reducedValues.begin(), reducedValues.end(), buffer.begin());
        }
    }

    //! The number of subscribers
    int numSubscribers_;
    //! Models data distributed over MPI ranks
    std::vector<RankData> rankData_;
};

TEST_P(ObservablesReducerIntegrationTest, CanBuildAndUseSimply)
{
    for (auto& rankData : rankData_)
    {
        rankData.observablesReducer = rankData.builder.build();
        EXPECT_TRUE(rankData.observablesReducer.communicationBuffer().empty())
                << "no buffer available when no subscribers requested reduction";
    }

    Step step = 0;
    for (auto& rankData : rankData_)
    {
        for (auto& subscriber : rankData.subscribers)
        {
            subscriber.doSimulationWork(step);
        }
        EXPECT_EQ(numSubscribers_ == 0, rankData.observablesReducer.communicationBuffer().empty())
                << "buffer should be available only when there are subscribers";
    }

    fakeMpiAllReduce();

    for (auto& rankData : rankData_)
    {
        rankData.observablesReducer.reductionComplete(step);
        EXPECT_TRUE(rankData.observablesReducer.communicationBuffer().empty())
                << "no buffer available after reductionComplete()";
    }
}

TEST_P(ObservablesReducerIntegrationTest, CanBuildAndUseOverMultipleSteps)
{
    for (auto& rankData : rankData_)
    {
        rankData.observablesReducer = rankData.builder.build();
        EXPECT_TRUE(rankData.observablesReducer.communicationBuffer().empty())
                << "no buffer available when no subscribers requested reduction";
    }

    for (Step step = 0; step < 20; step += 10)
    {
        for (auto& rankData : rankData_)
        {
            for (auto& subscriber : rankData.subscribers)
            {
                subscriber.doSimulationWork(step);
            }
            EXPECT_EQ(numSubscribers_ == 0, rankData.observablesReducer.communicationBuffer().empty())
                    << "buffer should be available only when there are subscribers";
        }

        fakeMpiAllReduce();

        for (auto& rankData : rankData_)
        {
            rankData.observablesReducer.reductionComplete(step);
            EXPECT_TRUE(rankData.observablesReducer.communicationBuffer().empty())
                    << "no buffer available after reductionComplete()";
        }
    }
}

TEST_P(ObservablesReducerIntegrationTest, CanBuildAndUseWithoutAllNeedingReduction)
{
    if (numSubscribers_ == 0)
    {
        // Test is meaningless with no subscribers
        return;
    }

    for (auto& rankData : rankData_)
    {
        rankData.observablesReducer = rankData.builder.build();
        EXPECT_TRUE(rankData.observablesReducer.communicationBuffer().empty())
                << "no buffer available when no subscribers requested reduction";
    }

    // Only one subscriber does work leading to reduction
    size_t subscriberNeedingReduction = 0;
    Step   step                       = 0;
    for (auto& rankData : rankData_)
    {
        auto& subscriber = rankData.subscribers[subscriberNeedingReduction];
        subscriber.doSimulationWork(step);
        EXPECT_FALSE(rankData.observablesReducer.communicationBuffer().empty())
                << "buffer should be available when there is a subscriber";
    }

    fakeMpiAllReduce();

    // Check that other subscribers didn't reduce anything
    for (auto& rankData : rankData_)
    {
        for (size_t r = 0; r != rankData.subscribers.size(); ++r)
        {
            if (r == subscriberNeedingReduction)
            {
                continue;
            }
            EXPECT_THAT(rankData.subscribers[r].communicationBuffer_, testing::Each(0.0))
                    << "buffer for non-subscribers should be zero";
        }
    }

    for (auto& rankData : rankData_)
    {
        rankData.observablesReducer.reductionComplete(step);
        EXPECT_TRUE(rankData.observablesReducer.communicationBuffer().empty())
                << "no buffer available after reductionComplete()";
    }
}

//! Help GoogleTest name our test cases
std::string namesOfTests(const testing::TestParamInfo<ObservablesReducerIntegrationTest::ParamType>& info)
{
    // NB alphanumeric characters only
    return formatString("numSubscribers%dnumRanks%d", std::get<0>(info.param), std::get<1>(info.param));
}
INSTANTIATE_TEST_CASE_P(WithVariousSubscriberCounts,
                        ObservablesReducerIntegrationTest,
                        testing::Combine(testing::Values(0, 1, 2, 3), // subscriber counts
                                         testing::Values(1, 2, 3)),   // rank counts
                        namesOfTests);

} // namespace
} // namespace gmx::test
