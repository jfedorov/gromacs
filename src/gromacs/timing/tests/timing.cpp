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
 * This implements tests for timing function wrappers and decorators
 *
 * \author Sebastian Keller <keller@cscs.ch>
 */

#include "gmxpre.h"

#include "config.h"

#include <chrono>
#include <thread>

#include "gromacs/timing/cyclecounter.h"
#include "gromacs/timing/wallcycle.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

//! Test function
void sleep(int msecs)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(msecs));
}

//! Test Fixture for timing tests
class TimingTest : public ::testing::Test
{
protected:
    void SetUp() override { wcycle = wallcycle_init(nullptr, 0, nullptr); }

    int             delay_ms = 1;
    gmx_wallcycle_t wcycle;
};


//! Test whether the decorator correctly wraps
TEST_F(TimingTest, DecorateWallCycle)
{
    TimerDecorator td(wcycle);

    int    probe = 0, ref = 1;
    int    n1, n2;
    double c1, c2;

    //! credit cycles from enclosing call to the ref field of wcycle
    wallcycle_start(wcycle, ref);
    //! cycles from the decorator are credited to the probe field
    td.wallcycle(probe, sleep, delay_ms);
    wallcycle_stop(wcycle, ref);
    //! extract both
    wallcycle_get(wcycle, probe, &n1, &c1);
    wallcycle_get(wcycle, ref, &n2, &c2);

    EXPECT_EQ(n1, n2);
    EXPECT_DOUBLE_EQ_TOL(c1, c2, relativeToleranceAsFloatingPoint(c1, 5e-3));
}

/*! Test whether the decorator correctly wraps sub counters
 * This test needs to be disabled if GMX_CYCLE_SUBCOUNTERS are not
 * enabled because in that case, the subcounts (wccs) data structure is not allocated
 */
#if GMX_CYCLE_SUBCOUNTERS
TEST_F(TimingTest, DecorateWallCycleSub)
#else
TEST_F(TimingTest, DISABLED_DecorateWallCycleSub)
#endif
{
    TimerDecorator td(wcycle);

    //! local variable to query wallcycle counters
    int    probe = 0;
    int    ref   = 1;
    int    n1, n2;
    double c1, c2;

    wallcycle_sub_start(wcycle, ref);
    td.wallcycle_sub(probe, sleep, delay_ms);
    wallcycle_sub_stop(wcycle, ref);
    wallcycle_sub_get(wcycle, probe, &n1, &c1);
    wallcycle_sub_get(wcycle, ref, &n2, &c2);

    EXPECT_EQ(n1, n2);
    EXPECT_DOUBLE_EQ_TOL(c1, c2, relativeToleranceAsFloatingPoint(c1, 5e-3));
}

/*! Test whether the decorator correctly wraps both counters simultaneously
 * This test needs to be disabled if GMX_CYCLE_SUBCOUNTERS are not
 * enabled because in that case, the subcounts (wccs) data structure is not allocated
 */
#if GMX_CYCLE_SUBCOUNTERS
TEST_F(TimingTest, DecorateWallCycleBoth)
#else
TEST_F(TimingTest, DISABLED_DecorateWallCycleBoth)
#endif
{
    TimerDecorator td(wcycle);

    int    probe = 0;
    int    ref   = 1;
    int    n1, n2;
    double c1, c2;

    wallcycle_start_nocount(wcycle, ref);
    wallcycle_sub_start(wcycle, ref);
    td.wallcycle_both(probe, probe, sleep, delay_ms);
    wallcycle_sub_stop(wcycle, ref);
    wallcycle_stop(wcycle, ref);

    wallcycle_get(wcycle, probe, &n1, &c1);
    wallcycle_get(wcycle, ref, &n2, &c2);
    EXPECT_EQ(n1, n2);
    EXPECT_DOUBLE_EQ_TOL(c1, c2, relativeToleranceAsFloatingPoint(c1, 5e-3));

    wallcycle_sub_get(wcycle, probe, &n1, &c1);
    wallcycle_sub_get(wcycle, ref, &n2, &c2);
    EXPECT_EQ(n1, n2);
    EXPECT_DOUBLE_EQ_TOL(c1, c2, relativeToleranceAsFloatingPoint(c1, 5e-3));
}

/*! \brief
 * Test that cycle counters cost less than 100 micro seconds per call pair
 * With the exception of Intel CPP, all compilers stay below 100 ns for this test
 */
TEST_F(TimingTest, CheckOverhead)
{
    TimerDecorator td(wcycle);

    int    n;
    double c;
    //! seconds per cycle
    double spc = gmx_cycles_calibrate(0.1);

    int reps = 1000000;

    wallcycle_start(wcycle, 0);
    for (int i = 0; i < reps; ++i)
    {
        td.wallcycle(1, []() {});
    }
    c = wallcycle_stop(wcycle, 0);

#if defined(__INTEL_COMPILER)
    //! 100 micro seconds for intel
    double expectedDelay = 100 * 1e-6 * reps;
#else
    //! < 100 ns per invocation pair
    double expectedDelay = 100 * 1e-9 * reps;
#endif


    //! c*spc: time elapsed in seconds
    EXPECT_LT(c * spc, expectedDelay);

    for (int i = 0; i < reps; ++i)
    {
        wallcycle_start(wcycle, 2);
        wallcycle_stop(wcycle, 2);
    }

    wallcycle_get(wcycle, 2, &n, &c);
    EXPECT_LT(c * spc, expectedDelay);
}

} // namespace
} // namespace test
} // namespace gmx
