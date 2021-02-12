/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
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
 * Tests to compare two simulators which are expected to be identical
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/mpitest.h"
#include "testutils/setenv.h"
#include "testutils/simulationdatabase.h"

#include "moduletest.h"
#include "simulatorcomparison.h"

namespace gmx
{
namespace test
{
namespace
{
//! Denotes a local database of mdp parameters
enum class MdpParameterDatabase
{
    Default,
    Andersen,
    SimulatedAnnealing,
    Pull,
    Pull2,
    Awh,
    Freeze,
    Count
};

/*! \brief Test fixture base for two equivalent simulators
 *
 * This test ensures that two simulator code paths (called via different mdp
 * options and/or environment variables) yield identical coordinate, velocity,
 * box, force and energy trajectories, up to some (arbitrary) precision.
 *
 * These tests are useful to check that re-implementations of existing simulators
 * are correct, and that different code paths expected to yield identical results
 * are equivalent.
 */
using SimulatorComparisonTestParams =
        std::tuple<std::tuple<std::string, std::string, std::string, std::string>, std::string, MdpParameterDatabase>;
class SimulatorComparisonTest :
    public MdrunTestFixture,
    public ::testing::WithParamInterface<SimulatorComparisonTestParams>
{
};

MdpFieldValues additionalMdpParametersDatabase(MdpParameterDatabase databaseEntry)
{
    switch (databaseEntry)
    {
        case MdpParameterDatabase::Default: return {};
        case MdpParameterDatabase::Andersen:
            // Fixes error "nstcomm must be 1, not 4 for Andersen, as velocities of
            //              atoms in coupled groups are randomized every time step"
            return { { "nstcomm", "1" }, { "nstcalcenergy", "1" } };
        case MdpParameterDatabase::SimulatedAnnealing:
            return {
                { "tc-grps", "Methanol SOL" },
                { "tau-t", "0.1 0.1" },
                { "ref-t", "298 298" },
                { "annealing", "single periodic" },
                { "annealing-npoints", "3 4" },
                { "annealing-time", "0 0.004 0.008 0 0.004 0.008 0.012" },
                { "annealing-temp", "298 280 270 298 320 320 298" },
            };
        case MdpParameterDatabase::Pull:
            return { { "coulombtype", "reaction-field" },
                     { "pull", "yes" },
                     // Prev step reference is scheduled by element
                     { "pull-pbc-ref-prev-step-com", "yes" },
                     { "pull-ngroups", "2" },
                     { "pull-group1-name", "FirstWaterMolecule" },
                     { "pull-group2-name", "SecondWaterMolecule" },
                     { "pull-ncoords", "1" },
                     { "pull-coord1-type", "umbrella" },
                     { "pull-coord1-geometry", "distance" },
                     { "pull-coord1-groups", "1 2" },
                     { "pull-coord1-init", "1" },
                     { "pull-coord1-k", "10000" } };
        case MdpParameterDatabase::Pull2:
            return { { "pull", "yes" },
                     { "pull-ngroups", "5" },
                     { "pull-ncoords", "2" },
                     { "pull-group1-name", "C_&_r_1" },
                     { "pull-group2-name", "N_&_r_2" },
                     { "pull-group3-name", "CA" },
                     { "pull-group4-name", "C_&_r_2" },
                     { "pull-group5-name", "N_&_r_3" },
                     { "pull-coord1-geometry", "dihedral" },
                     { "pull-coord1-groups", "1 2 2 3 3 4" },
                     { "pull-coord1-k", "4000" },
                     { "pull-coord1-kB", "1000" },
                     { "pull-coord2-geometry", "dihedral" },
                     { "pull-coord2-groups", "2 3 3 4 4 5" },
                     { "pull-coord2-k", "4000" },
                     { "pull-coord2-kB", "1000" } };
        case MdpParameterDatabase::Awh:
        {
            auto pull2Params = additionalMdpParametersDatabase(MdpParameterDatabase::Pull2);
            pull2Params.insert({ { "pull-coord1-type", "external-potential" },
                                 { "pull-coord1-potential-provider", "awh" },
                                 { "pull-coord2-type", "external-potential" },
                                 { "pull-coord2-potential-provider", "awh" },
                                 { "awh", "yes" },
                                 { "awh-potential", "convolved" },
                                 { "awh-nstout", "4" },
                                 { "awh-nstsample", "4" },
                                 { "awh-nsamples-update", "1" },
                                 { "awh-share-multisim", "no" },
                                 { "awh-nbias", "2" },
                                 { "awh1-ndim", "1" },
                                 { "awh1-dim1-coord-index", "2" },
                                 { "awh1-dim1-start", "150" },
                                 { "awh1-dim1-end", "180" },
                                 { "awh1-dim1-force-constant", "4000" },
                                 { "awh1-dim1-diffusion", "0.1" },
                                 { "awh2-ndim", "1" },
                                 { "awh2-dim1-coord-index", "1" },
                                 { "awh2-dim1-start", "178" },
                                 { "awh2-dim1-end", "-178" },
                                 { "awh2-dim1-force-constant", "4000" },
                                 { "awh2-dim1-diffusion", "0.1" } });
            return pull2Params;
        }
        case MdpParameterDatabase::Freeze:
            // One fully frozen, one partially frozen group
            // Constraints because these may be problematic with partially frozen groups
            // COMM removal is wrong with partially frozen atoms, so turn off
            return { { "freezegrps", "Backbone SideChain" },
                     { "freezedim", "Y Y Y N N Y" },
                     { "constraints", "all-bonds" },
                     { "comm-mode", "none" } };
        default: GMX_THROW(InvalidInputError("Unknown additional parameters."));
    }
}

MdpFieldValues prepareMdpFieldValues(const std::string&   simulationName,
                                     const std::string&   integrator,
                                     const std::string&   tcoupling,
                                     const std::string&   pcoupling,
                                     MdpParameterDatabase additionalMdpParameters)
{
    auto mdpFieldValues = test::prepareMdpFieldValues(simulationName, integrator, tcoupling, pcoupling);
    for (auto const& [key, value] : additionalMdpParametersDatabase(additionalMdpParameters))
    {
        mdpFieldValues[key] = value;
    }
    if (tcoupling == "nose-hoover" && pcoupling == "parrinello-rahman")
    {
        // Default value yields warning (tau-p needs to be twice as large as tau-t)
        mdpFieldValues["tau-p"] = "2";
    }
    return mdpFieldValues;
}

TEST_P(SimulatorComparisonTest, WithinTolerances)
{
    const auto& params              = GetParam();
    const auto& mdpParams           = std::get<0>(params);
    const auto& simulationName      = std::get<0>(mdpParams);
    const auto& integrator          = std::get<1>(mdpParams);
    const auto& tcoupling           = std::get<2>(mdpParams);
    const auto& pcoupling           = std::get<3>(mdpParams);
    const auto& environmentVariable = std::get<1>(params);
    auto        additionalParams    = std::get<2>(params);

    const bool hasConstraints    = (simulationName != "argon12");
    const bool isAndersen        = (tcoupling == "andersen-massive" || tcoupling == "andersen");
    const auto hasConservedField = !(tcoupling == "no" && pcoupling == "no") && !isAndersen;

    int maxNumWarnings = 0;

    // TODO At some point we should also test PME-only ranks.
    const int numRanksAvailable = getNumberOfTestMpiRanks();
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

    if (integrator == "md-vv" && pcoupling == "Parrinello-Rahman")
    {
        // do_md calls this MTTK, requires Nose-Hoover, and
        // does not work with constraints or anisotropically
        return;
    }

    if (pcoupling == "mttk" && (tcoupling != "nose-hoover" || hasConstraints))
    {
        // Legacy MTTK works only with Nose-Hoover and without constraints
        return;
    }
    if (tcoupling == "nose-hoover" && pcoupling == "berendsen")
    {
        if (integrator == "md-vv")
        {
            // Combination not allowed by legacy do_md.
            return;
        }
        else
        {
            // "Using Berendsen pressure coupling invalidates the true ensemble for the thermostat"
            maxNumWarnings++;
        }
    }
    if (isAndersen && pcoupling == "berendsen")
    {
        // "Using Berendsen pressure coupling invalidates the true ensemble for the thermostat"
        maxNumWarnings++;
    }
    if (tcoupling == "andersen" && hasConstraints)
    {
        // Constraints are not allowed with non-massive Andersen
        return;
    }

    const std::string envVariableModSimOn  = "GMX_USE_MODULAR_SIMULATOR";
    const std::string envVariableModSimOff = "GMX_DISABLE_MODULAR_SIMULATOR";

    GMX_RELEASE_ASSERT(
            environmentVariable == envVariableModSimOn || environmentVariable == envVariableModSimOff,
            ("Expected tested environment variable to be " + envVariableModSimOn + " or " + envVariableModSimOff)
                    .c_str());

    SCOPED_TRACE(formatString(
            "Comparing two simulations of '%s' "
            "with integrator '%s', '%s' temperature coupling, and '%s' pressure coupling "
            "switching environment variable '%s'",
            simulationName.c_str(),
            integrator.c_str(),
            tcoupling.c_str(),
            pcoupling.c_str(),
            environmentVariable.c_str()));

    if (tcoupling == "andersen" && additionalParams == MdpParameterDatabase::Default)
    {
        additionalParams = MdpParameterDatabase::Andersen;
    }
    const auto mdpFieldValues =
            prepareMdpFieldValues(simulationName, integrator, tcoupling, pcoupling, additionalParams);

    EnergyTermsToCompare energyTermsToCompare{ {
            { interaction_function[F_EPOT].longname, relativeToleranceAsPrecisionDependentUlp(60.0, 200, 160) },
            { interaction_function[F_EKIN].longname, relativeToleranceAsPrecisionDependentUlp(60.0, 200, 160) },
            { interaction_function[F_PRES].longname,
              relativeToleranceAsPrecisionDependentFloatingPoint(10.0, 0.01, 0.001) },
    } };
    if (hasConservedField)
    {
        energyTermsToCompare.emplace(interaction_function[F_ECONSERVED].longname,
                                     relativeToleranceAsPrecisionDependentUlp(50.0, 100, 80));
    }

    if (simulationName == "argon12")
    {
        // Without constraints, we can be more strict
        energyTermsToCompare = { {
                { interaction_function[F_EPOT].longname,
                  relativeToleranceAsPrecisionDependentUlp(10.0, 24, 80) },
                { interaction_function[F_EKIN].longname,
                  relativeToleranceAsPrecisionDependentUlp(10.0, 24, 80) },
                { interaction_function[F_PRES].longname,
                  relativeToleranceAsPrecisionDependentFloatingPoint(10.0, 0.001, 0.0001) },
        } };
        if (hasConservedField)
        {
            energyTermsToCompare.emplace(interaction_function[F_ECONSERVED].longname,
                                         relativeToleranceAsPrecisionDependentUlp(10.0, 24, 80));
        }
    }

    // Specify how trajectory frame matching must work.
    const TrajectoryFrameMatchSettings trajectoryMatchSettings{ true,
                                                                true,
                                                                true,
                                                                ComparisonConditions::MustCompare,
                                                                ComparisonConditions::MustCompare,
                                                                ComparisonConditions::MustCompare,
                                                                MaxNumFrames::compareAllFrames() };
    TrajectoryTolerances trajectoryTolerances = TrajectoryComparison::s_defaultTrajectoryTolerances;
    if (simulationName != "argon12")
    {
        trajectoryTolerances.velocities = trajectoryTolerances.coordinates;
    }

    // Build the functor that will compare reference and test
    // trajectory frames in the chosen way.
    const TrajectoryComparison trajectoryComparison{ trajectoryMatchSettings, trajectoryTolerances };

    // Set file names
    const auto simulator1TrajectoryFileName = fileManager_.getTemporaryFilePath("sim1.trr");
    const auto simulator1EdrFileName        = fileManager_.getTemporaryFilePath("sim1.edr");
    const auto simulator2TrajectoryFileName = fileManager_.getTemporaryFilePath("sim2.trr");
    const auto simulator2EdrFileName        = fileManager_.getTemporaryFilePath("sim2.edr");

    // Run grompp
    runner_.tprFileName_ = fileManager_.getTemporaryFilePath("sim.tpr");
    runner_.useTopGroAndNdxFromDatabase(simulationName);
    runner_.useStringAsMdpFile(prepareMdpFileContents(mdpFieldValues));
    runGrompp(&runner_, { SimulationOptionTuple("-maxwarn", std::to_string(maxNumWarnings)) });

    // Backup current state of both environment variables and unset them
    const char* environmentVariableBackupOn  = getenv(envVariableModSimOn.c_str());
    const char* environmentVariableBackupOff = getenv(envVariableModSimOff.c_str());
    gmxUnsetenv(envVariableModSimOn.c_str());
    gmxUnsetenv(envVariableModSimOff.c_str());

    // Do first mdrun
    runner_.fullPrecisionTrajectoryFileName_ = simulator1TrajectoryFileName;
    runner_.edrFileName_                     = simulator1EdrFileName;
    runMdrun(&runner_);

    // Set tested environment variable
    const int overWriteEnvironmentVariable = 1;
    gmxSetenv(environmentVariable.c_str(), "ON", overWriteEnvironmentVariable);

    // Do second mdrun
    runner_.fullPrecisionTrajectoryFileName_ = simulator2TrajectoryFileName;
    runner_.edrFileName_                     = simulator2EdrFileName;
    runMdrun(&runner_);

    // Unset tested environment variable
    gmxUnsetenv(environmentVariable.c_str());
    // Reset both environment variables to leave further tests undisturbed
    if (environmentVariableBackupOn != nullptr)
    {
        gmxSetenv(environmentVariable.c_str(), environmentVariableBackupOn, overWriteEnvironmentVariable);
    }
    if (environmentVariableBackupOff != nullptr)
    {
        gmxSetenv(environmentVariable.c_str(), environmentVariableBackupOff, overWriteEnvironmentVariable);
    }

    // Compare simulation results
    compareEnergies(simulator1EdrFileName, simulator2EdrFileName, energyTermsToCompare);
    compareTrajectories(simulator1TrajectoryFileName, simulator2TrajectoryFileName, trajectoryComparison);
}

// TODO: The time for OpenCL kernel compilation means these tests time
//       out. Once that compilation is cached for the whole process, these
//       tests can run in such configurations.
// These tests are very sensitive, so we only run them in double precision.
// As we change call ordering, they might actually become too strict to be useful.
#if !GMX_GPU_OPENCL && GMX_DOUBLE
INSTANTIATE_TEST_CASE_P(SimulatorsAreEquivalentDefaultModular,
                        SimulatorComparisonTest,
                        ::testing::Combine(::testing::Combine(::testing::Values("argon12", "tip3p5"),
                                                              ::testing::Values("md-vv"),
                                                              ::testing::Values("no",
                                                                                "v-rescale",
                                                                                "berendsen",
                                                                                "nose-hoover",
                                                                                "andersen-massive",
                                                                                "andersen"),
                                                              ::testing::Values("no",
                                                                                "mttk",
                                                                                "berendsen",
                                                                                "c-rescale")),
                                           ::testing::Values("GMX_DISABLE_MODULAR_SIMULATOR"),
                                           ::testing::Values(MdpParameterDatabase::Default)));
INSTANTIATE_TEST_CASE_P(
        SimulatorsAreEquivalentDefaultLegacy,
        SimulatorComparisonTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values("argon12", "tip3p5"),
                        ::testing::Values("md"),
                        ::testing::Values("no", "v-rescale", "berendsen", "nose-hoover"),
                        ::testing::Values("no", "Parrinello-Rahman", "berendsen", "c-rescale")),
                ::testing::Values("GMX_USE_MODULAR_SIMULATOR"),
                ::testing::Values(MdpParameterDatabase::Default)));
INSTANTIATE_TEST_CASE_P(
        SimulatorsAreEquivalentDefaultModularVirtualSites,
        SimulatorComparisonTest,
        ::testing::Combine(::testing::Combine(::testing::Values("vsite_test"),
                                              ::testing::Values("md-vv"),
                                              ::testing::Values("no", "v-rescale", "nose-hoover"),
                                              ::testing::Values("no", "c-rescale", "mttk")),
                           ::testing::Values("GMX_DISABLE_MODULAR_SIMULATOR"),
                           ::testing::Values(MdpParameterDatabase::Default)));
INSTANTIATE_TEST_CASE_P(
        SimulatorsAreEquivalentDefaultLegacyVirtualSites,
        SimulatorComparisonTest,
        ::testing::Combine(
                ::testing::Combine(::testing::Values("vsite_test"),
                                   ::testing::Values("md"),
                                   ::testing::Values("no", "v-rescale", "nose-hoover"),
                                   ::testing::Values("no", "parrinello-rahman", "c-rescale")),
                ::testing::Values("GMX_USE_MODULAR_SIMULATOR"),
                ::testing::Values(MdpParameterDatabase::Default)));
INSTANTIATE_TEST_CASE_P(
        SimulatorsAreEquivalentDefaultModularSimulatedAnnealing,
        SimulatorComparisonTest,
        ::testing::Combine(::testing::Combine(::testing::Values("spc-and-methanol"),
                                              ::testing::Values("md-vv"),
                                              ::testing::Values("v-rescale", "nose-hoover"),
                                              ::testing::Values("no", "c-rescale", "mttk")),
                           ::testing::Values("GMX_DISABLE_MODULAR_SIMULATOR"),
                           ::testing::Values(MdpParameterDatabase::SimulatedAnnealing)));
INSTANTIATE_TEST_CASE_P(
        SimulatorsAreEquivalentDefaultLegacySimulatedAnnealing,
        SimulatorComparisonTest,
        ::testing::Combine(
                ::testing::Combine(::testing::Values("spc-and-methanol"),
                                   ::testing::Values("md"),
                                   ::testing::Values("no", "v-rescale", "nose-hoover"),
                                   ::testing::Values("no", "parrinello-rahman", "c-rescale")),
                ::testing::Values("GMX_USE_MODULAR_SIMULATOR"),
                ::testing::Values(MdpParameterDatabase::SimulatedAnnealing)));
INSTANTIATE_TEST_CASE_P(SimulatorsAreEquivalentDefaultModularPull,
                        SimulatorComparisonTest,
                        ::testing::Combine(::testing::Combine(::testing::Values("spc2"),
                                                              ::testing::Values("md-vv"),
                                                              ::testing::Values("no"),
                                                              ::testing::Values("no")),
                                           ::testing::Values("GMX_DISABLE_MODULAR_SIMULATOR"),
                                           ::testing::Values(MdpParameterDatabase::Pull)));
INSTANTIATE_TEST_CASE_P(SimulatorsAreEquivalentDefaultLegacyPull,
                        SimulatorComparisonTest,
                        ::testing::Combine(::testing::Combine(::testing::Values("spc2"),
                                                              ::testing::Values("md"),
                                                              ::testing::Values("no"),
                                                              ::testing::Values("no")),
                                           ::testing::Values("GMX_USE_MODULAR_SIMULATOR"),
                                           ::testing::Values(MdpParameterDatabase::Pull)));
INSTANTIATE_TEST_CASE_P(SimulatorsAreEquivalentDefaultModularAwh,
                        SimulatorComparisonTest,
                        ::testing::Combine(::testing::Combine(::testing::Values("alanine_vacuo"),
                                                              ::testing::Values("md-vv"),
                                                              ::testing::Values("v-rescale"),
                                                              ::testing::Values("no")),
                                           ::testing::Values("GMX_DISABLE_MODULAR_SIMULATOR"),
                                           ::testing::Values(MdpParameterDatabase::Awh)));
INSTANTIATE_TEST_CASE_P(SimulatorsAreEquivalentDefaultLegacyAwh,
                        SimulatorComparisonTest,
                        ::testing::Combine(::testing::Combine(::testing::Values("alanine_vacuo"),
                                                              ::testing::Values("md"),
                                                              ::testing::Values("v-rescale"),
                                                              ::testing::Values("no")),
                                           ::testing::Values("GMX_USE_MODULAR_SIMULATOR"),
                                           ::testing::Values(MdpParameterDatabase::Awh)));
INSTANTIATE_TEST_CASE_P(SimulatorsAreEquivalentDefaultModularFreeze,
                        SimulatorComparisonTest,
                        ::testing::Combine(::testing::Combine(::testing::Values("alanine_vacuo"),
                                                              ::testing::Values("md-vv"),
                                                              ::testing::Values("v-rescale"),
                                                              ::testing::Values("no")),
                                           ::testing::Values("GMX_DISABLE_MODULAR_SIMULATOR"),
                                           ::testing::Values(MdpParameterDatabase::Freeze)));
INSTANTIATE_TEST_CASE_P(SimulatorsAreEquivalentDefaultLegacyFreeze,
                        SimulatorComparisonTest,
                        ::testing::Combine(::testing::Combine(::testing::Values("alanine_vacuo"),
                                                              ::testing::Values("md"),
                                                              ::testing::Values("v-rescale"),
                                                              ::testing::Values("no")),
                                           ::testing::Values("GMX_USE_MODULAR_SIMULATOR"),
                                           ::testing::Values(MdpParameterDatabase::Freeze)));
#else
INSTANTIATE_TEST_CASE_P(DISABLED_SimulatorsAreEquivalentDefaultModular,
                        SimulatorComparisonTest,
                        ::testing::Combine(::testing::Combine(::testing::Values("argon12", "tip3p5"),
                                                              ::testing::Values("md-vv"),
                                                              ::testing::Values("no",
                                                                                "v-rescale",
                                                                                "berendsen",
                                                                                "nose-hoover",
                                                                                "andersen-massive",
                                                                                "andersen"),
                                                              ::testing::Values("no",
                                                                                "mttk",
                                                                                "berendsen",
                                                                                "c-rescale")),
                                           ::testing::Values("GMX_DISABLE_MODULAR_SIMULATOR"),
                                           ::testing::Values(MdpParameterDatabase::Default)));
INSTANTIATE_TEST_CASE_P(
        DISABLED_SimulatorsAreEquivalentDefaultLegacy,
        SimulatorComparisonTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values("argon12", "tip3p5"),
                        ::testing::Values("md"),
                        ::testing::Values("no", "v-rescale", "berendsen", "nose-hoover"),
                        ::testing::Values("no", "Parrinello-Rahman", "berendsen", "c-rescale")),
                ::testing::Values("GMX_USE_MODULAR_SIMULATOR"),
                ::testing::Values(MdpParameterDatabase::Default)));
INSTANTIATE_TEST_CASE_P(
        DISABLED_SimulatorsAreEquivalentDefaultModularVirtualSites,
        SimulatorComparisonTest,
        ::testing::Combine(::testing::Combine(::testing::Values("vsite_test"),
                                              ::testing::Values("md-vv"),
                                              ::testing::Values("no", "v-rescale", "nose-hoover"),
                                              ::testing::Values("no", "c-rescale", "mttk")),
                           ::testing::Values("GMX_DISABLE_MODULAR_SIMULATOR"),
                           ::testing::Values(MdpParameterDatabase::Default)));
INSTANTIATE_TEST_CASE_P(
        DISABLED_SimulatorsAreEquivalentDefaultLegacyVirtualSites,
        SimulatorComparisonTest,
        ::testing::Combine(
                ::testing::Combine(::testing::Values("vsite_test"),
                                   ::testing::Values("md"),
                                   ::testing::Values("no", "v-rescale", "nose-hoover"),
                                   ::testing::Values("no", "parrinello-rahman", "c-rescale")),
                ::testing::Values("GMX_USE_MODULAR_SIMULATOR"),
                ::testing::Values(MdpParameterDatabase::Default)));
INSTANTIATE_TEST_CASE_P(
        DISABLED_SimulatorsAreEquivalentDefaultModularSimulatedAnnealing,
        SimulatorComparisonTest,
        ::testing::Combine(::testing::Combine(::testing::Values("spc-and-methanol"),
                                              ::testing::Values("md-vv"),
                                              ::testing::Values("v-rescale", "nose-hoover"),
                                              ::testing::Values("no", "c-rescale", "mttk")),
                           ::testing::Values("GMX_DISABLE_MODULAR_SIMULATOR"),
                           ::testing::Values(MdpParameterDatabase::SimulatedAnnealing)));
INSTANTIATE_TEST_CASE_P(
        DISABLED_SimulatorsAreEquivalentDefaultLegacySimulatedAnnealing,
        SimulatorComparisonTest,
        ::testing::Combine(
                ::testing::Combine(::testing::Values("spc-and-methanol"),
                                   ::testing::Values("md"),
                                   ::testing::Values("no", "v-rescale", "nose-hoover"),
                                   ::testing::Values("no", "parrinello-rahman", "c-rescale")),
                ::testing::Values("GMX_USE_MODULAR_SIMULATOR"),
                ::testing::Values(MdpParameterDatabase::SimulatedAnnealing)));
INSTANTIATE_TEST_CASE_P(DISABLED_SimulatorsAreEquivalentDefaultModularFreeze,
                        SimulatorComparisonTest,
                        ::testing::Combine(::testing::Combine(::testing::Values("alanine_vacuo"),
                                                              ::testing::Values("md-vv"),
                                                              ::testing::Values("v-rescale"),
                                                              ::testing::Values("no")),
                                           ::testing::Values("GMX_DISABLE_MODULAR_SIMULATOR"),
                                           ::testing::Values(MdpParameterDatabase::Freeze)));
INSTANTIATE_TEST_CASE_P(DISABLED_SimulatorsAreEquivalentDefaultLegacyFreeze,
                        SimulatorComparisonTest,
                        ::testing::Combine(::testing::Combine(::testing::Values("alanine_vacuo"),
                                                              ::testing::Values("md"),
                                                              ::testing::Values("v-rescale"),
                                                              ::testing::Values("no")),
                                           ::testing::Values("GMX_USE_MODULAR_SIMULATOR"),
                                           ::testing::Values(MdpParameterDatabase::Freeze)));
#endif

} // namespace
} // namespace test
} // namespace gmx
