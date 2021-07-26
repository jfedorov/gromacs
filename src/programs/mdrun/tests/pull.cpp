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
 * Tests for mdrun pull functionality.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/options/filenameoption.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/mpitest.h"
#include "testutils/refdata.h"
#include "testutils/simulationdatabase.h"
#include "testutils/testasserts.h"

#include "energycomparison.h"
#include "moduletest.h"

namespace gmx
{
namespace test
{
namespace
{

/*! \brief Database of energy tolerances on the various systems. */
std::unordered_map<std::string, FloatingPointTolerance> energyToleranceForSystem_g = {
    { { "spc216", relativeToleranceAsFloatingPoint(1, 1e-4) } }
};

/*! \brief Database of pressure tolerances on the various systems. */
std::unordered_map<std::string, FloatingPointTolerance> pressureToleranceForSystem_g = {
    { { "spc216", relativeToleranceAsFloatingPoint(1, 2e-4) } }
};

const std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> c_mdpPullParams = {
    { { "umbrella-3D",
        { { "pull-ngroups", "2" },
          { "pull-ncoords", "1" },
          { "pull-nstxout", "0" },
          { "pull-nstfout", "0" },
          { "pull-group1-name", "r_1" },
          { "pull-group2-name", "r_2" },
          { "pull-coord1-groups", "1 2" },
          { "pull-coord1-type", "umbrella" },
          { "pull-coord1-geometry", "distance" },
          { "pull-coord1-dim", "Y Y Y" },
          { "pull-coord1-init", "0.6" },
          { "pull-coord1-k", "100" } } },
      { "umbrella-2D",
        { { "pull-ngroups", "2" },
          { "pull-ncoords", "1" },
          { "pull-nstxout", "0" },
          { "pull-nstfout", "0" },
          { "pull-group1-name", "r_1" },
          { "pull-group2-name", "r_2" },
          { "pull-coord1-groups", "1 2" },
          { "pull-coord1-type", "umbrella" },
          { "pull-coord1-geometry", "distance" },
          { "pull-coord1-dim", "Y Y N" },
          { "pull-coord1-init", "0.4" },
          { "pull-coord1-k", "100" } } },
      { "constraint-flatbottom",
        { { "pull-ngroups", "3" },
          { "pull-ncoords", "2" },
          { "pull-nstxout", "0" },
          { "pull-nstfout", "0" },
          { "pull-group1-name", "r_1" },
          { "pull-group2-name", "r_2" },
          { "pull-group3-name", "r_3" },
          { "pull-coord1-groups", "1 2" },
          { "pull-coord1-type", "constraint" },
          { "pull-coord1-geometry", "distance" },
          { "pull-coord1-dim", "Y Y Y" },
          { "pull-coord1-init", "0.5" },
          { "pull-coord2-groups", "1 3" },
          { "pull-coord2-type", "flat-bottom" },
          { "pull-coord2-geometry", "distance" },
          { "pull-coord2-dim", "Y Y Y" },
          { "pull-coord2-init", "0.4" },
          { "pull-coord2-k", "100" } } } }
};

//! Helper type
using MdpField = MdpFieldValues::value_type;

/*! \brief Test fixture base for simple mdrun systems
 *
 * This test ensures mdrun can run a simulation, reaching
 * reproducible energies.
 *
 * The choices for tolerance are arbitrary but sufficient. */
class PullIntegrationTest :
    public MdrunTestFixture,
    public ::testing::WithParamInterface<std::tuple<std::string, std::string>>
{
};

//! Adds integrator and nonbonded parameter setup
void addBasicMdpValues(MdpFieldValues* mdpFieldValues)
{
    (*mdpFieldValues)["nsteps"]        = "20";
    (*mdpFieldValues)["nstcomm"]       = "10";
    (*mdpFieldValues)["nstlist"]       = "10";
    (*mdpFieldValues)["nstcalcenergy"] = "5";
    (*mdpFieldValues)["nstenergy"]     = "5";
    (*mdpFieldValues)["coulombtype"]   = "Reaction-field";
    (*mdpFieldValues)["vdwtype"]       = "Cut-off";
}

TEST_P(PullIntegrationTest, WithinTolerances)
{
    auto params         = GetParam();
    auto simulationName = std::get<0>(params);
    auto pullSetup      = std::get<1>(params);
    SCOPED_TRACE(formatString("Comparing simple mdrun for '%s'", simulationName.c_str()));

    // TODO At some point we should also test PME-only ranks.
    int numRanksAvailable = getNumberOfTestMpiRanks();
    if (!isNumberOfPpRanksSupported(simulationName, numRanksAvailable))
    {
        fprintf(stdout,
                "Test system '%s' cannot run with %d ranks.\n"
                "The supported numbers are: %s\n",
                simulationName.c_str(),
                numRanksAvailable,
                reportNumbersOfPpRanksSupported(simulationName).c_str());
        return;
    }
    auto mdpFieldValues = prepareMdpFieldValues(simulationName.c_str(), "md", "no", "no");
    addBasicMdpValues(&mdpFieldValues);

    // Add the pull parameters
    mdpFieldValues["pull"]    = "yes";
    const auto& mdpPullParams = c_mdpPullParams.at(pullSetup);
    for (const auto& param : mdpPullParams)
    {
        mdpFieldValues[param.first] = param.second;
    }

    // Prepare the .tpr file
    {
        CommandLine caller;
        runner_.useTopGroAndNdxFromDatabase(simulationName);
        runner_.useNdxFromDatabase(simulationName + "_pull");
        runner_.useStringAsMdpFile(prepareMdpFileContents(mdpFieldValues));
        EXPECT_EQ(0, runner_.callGrompp(caller));
    }
    // Do mdrun
    {
        CommandLine mdrunCaller;
        ASSERT_EQ(0, runner_.callMdrun(mdrunCaller));
        EnergyTermsToCompare energyTermsToCompare{ {
                { interaction_function[F_COM_PULL].longname, energyToleranceForSystem_g.at(simulationName) },
                { interaction_function[F_EPOT].longname, energyToleranceForSystem_g.at(simulationName) },
                { interaction_function[F_EKIN].longname, energyToleranceForSystem_g.at(simulationName) },
                { interaction_function[F_PRES].longname, pressureToleranceForSystem_g.at(simulationName) },
        } };
        TestReferenceData    refData;
        auto                 checker = refData.rootChecker()
                               .checkCompound("Simulation", simulationName)
                               .checkCompound("PullSetup", pullSetup);
        checkEnergiesAgainstReferenceData(runner_.edrFileName_, energyTermsToCompare, &checker);
    }
}

INSTANTIATE_TEST_SUITE_P(PullTest,
                         PullIntegrationTest,
                         ::testing::Combine(::testing::Values("spc216"),
                                            ::testing::Values("umbrella-3D",
                                                              "umbrella-2D",
                                                              "constraint-flatbottom")));

} // namespace
} // namespace test
} // namespace gmx
