/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017,2018,2019,2020,2021, by the GROMACS development team, led by
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
 * This implements basic PME sanity tests for end-to-end mdrun simulations.
 * It runs the input system with PME for several steps (on CPU and GPU, if available),
 * and checks the reciprocal and conserved energies.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include <map>
#include <string>
#include <vector>

#include <gtest/gtest-spi.h>

#include "gromacs/ewald/pme.h"
#include "gromacs/gmxpreprocess/grompp.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/trajectory/energyframe.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/message_string_collector.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/mpitest.h"
#include "testutils/refdata.h"

#include "energyreader.h"
#include "moduletest.h"

namespace gmx
{
namespace test
{
namespace
{

//! Enum describing the flavors of PME tests that are run
enum class PmeTestFlavor : int
{
    Basic,
    WithWalls,
    Count
};

//! Helper to print a string to describe the PME test flavor
const char* enumValueToString(const PmeTestFlavor enumValue)
{
    static constexpr gmx::EnumerationArray<PmeTestFlavor, const char*> s_names = {
        "basic",
        "with walls",
    };
    return s_names[enumValue];
}

// Paramters for parametrized test fixture: the flavor of PME test to
// run, and options for an mdrun command line.
using PmeTestParameters = std::tuple<PmeTestFlavor, std::string>;

/*! \brief Help GoogleTest name our tests
 *
 * If changes are needed here, consider making matching changes in
 * makeRefDataFileName(). */
std::string nameOfTest(const testing::TestParamInfo<PmeTestParameters>& info)
{
    std::string testName = formatString(
            "%s_mdrun_%s", enumValueToString(std::get<0>(info.param)), std::get<1>(info.param).c_str());

    // Note that the returned names must be unique and may use only
    // alphanumeric ASCII characters. It's not supposed to contain
    // underscores (see the GoogleTest FAQ
    // why-should-test-suite-names-and-test-names-not-contain-underscore),
    // but doing so works for now, is likely to remain so, and makes
    // such test names much more readable.
    testName = replaceAll(testName, "-", "");
    testName = replaceAll(testName, " ", "_");
    return testName;
}

/*! \brief Construct a refdata filename for this test
 *
 * We want the same reference data to apply to every mdrun command
 * line that we test. That means we need to store it in a file whose
 * name relates to the name of the test excluding the part related to
 * the mdrun command line. By default, the reference data filename is
 * set via a call to gmx::TestFileManager::getTestSpecificFileName()
 * that queries GoogleTest and gets a string that includes the return
 * value for nameOfTest(). This code works similarly, but removes the
 * aforementioned part. This logic must match the implementation of
 * nameOfTest() so that it works as intended. */
std::string makeRefDataFileName()
{
    // Get the info about the test
    const ::testing::TestInfo* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();

    // Get the test name and edit it to remove the mdrun command-line
    // part.
    std::string testName(testInfo->name());
    auto        separatorPos = testName.find("_mdrun");
    testName                 = testName.substr(0, separatorPos);

    // Build the complete filename like getTestSpecificFilename() does
    // it.
    std::string testSuiteName(testInfo->test_suite_name());
    std::string refDataFileName = testSuiteName + "_" + testName + ".xml";
    std::replace(refDataFileName.begin(), refDataFileName.end(), '/', '_');

    return refDataFileName;
}

/*! \brief Test fixture for end-to-end execution of PME */
class PmeTest : public MdrunTestFixture, public ::testing::WithParamInterface<PmeTestParameters>
{
public:
    //! Names of tpr files built by grompp in SetUpTestSuite to run in tests
    inline static EnumerationArray<PmeTestFlavor, std::string> s_tprFileNames;
    //! Manager needed in SetUpTestSuite to handle making tpr files
    inline static std::unique_ptr<TestFileManager> s_testFileManager;
    //! Name of the system from the simulation database to use in SetupTestSuite
    inline static const std::string s_inputFile = "spc-and-methanol";
    //! Runs grompp to prepare the tpr files that are reused in the tests
    static void SetUpTestSuite();
    /*! \brief If the mdrun command line can't run on this build or
     * hardware, we should mark it as skipped and describe why. */
    static MessageStringCollector getSkipMessagesIfNecessary(const CommandLine& commandLine);
    //! Check the energies against the reference data.
    void checkEnergies(bool usePmeTuning) const;
};

// static
void PmeTest::SetUpTestSuite()
{
    MdrunTestFixture::SetUpTestSuite();
    // Ensure we only make one per process
    if (GMX_THREAD_MPI && 0 == gmx_node_rank())
    {
        s_testFileManager = std::make_unique<TestFileManager>();
    }

    const static std::unordered_map<PmeTestFlavor, std::string> sc_pmeTestFlavorExtraMdpLines = {
        { PmeTestFlavor::Basic,
          { "nsteps = 20\n"
            "pbc    = xyz\n" } },
        { PmeTestFlavor::WithWalls,
          { "nsteps          = 0\n"
            "pbc             = xy\n"
            "nwall           = 2\n"
            "ewald-geometry  = 3dc\n"
            "wall_atomtype   = CMet H\n"
            "wall_density    = 9 9.0\n"
            "wall-ewald-zfac = 5\n" } },
    };

    // Make the tpr files for the different flavors
    for (PmeTestFlavor pmeTestFlavor : EnumerationWrapper<PmeTestFlavor>{})
    {
        SimulationRunner runner(s_testFileManager.get());
        runner.useTopGroAndNdxFromDatabase(s_inputFile);

        std::string mdpInputFileContents(
                "coulombtype     = PME\n"
                "nstcalcenergy   = 1\n"
                "nstenergy       = 1\n"
                "pme-order       = 4\n");
        mdpInputFileContents += sc_pmeTestFlavorExtraMdpLines.at(pmeTestFlavor);
        runner.useStringAsMdpFile(mdpInputFileContents);

        std::string tprFileNameSuffix = formatString("%s.tpr", enumValueToString(pmeTestFlavor));
        std::replace(tprFileNameSuffix.begin(), tprFileNameSuffix.end(), ' ', '_');
        runner.tprFileName_ = s_testFileManager->getTemporaryFilePath(tprFileNameSuffix);
        runner.callGrompp();
        s_tprFileNames[pmeTestFlavor] = runner.tprFileName_;
    }
}

/*! \brief If the mdrun command line can't run on this build or
 * hardware, we should mark it as skipped and describe why. */
MessageStringCollector PmeTest::getSkipMessagesIfNecessary(const CommandLine& commandLine)
{
    // Note that we can't call GTEST_SKIP() from within this method,
    // because it only returns from the current function. So we
    // collect all the reasons why the test cannot run, return them
    // and skip in a higher stack frame.

    MessageStringCollector messages;
    messages.startContext("Test is being skipped because:");

    const int numRanks = getNumberOfTestMpiRanks();

    std::optional<std::string_view> npmeOptionArgument = commandLine.argumentOf("-npme");
    const bool                      commandLineTargetsPmeOnlyRanks =
            !npmeOptionArgument.has_value() || std::stoi(std::string(npmeOptionArgument.value())) > 0;

    messages.appendIf(commandLineTargetsPmeOnlyRanks && numRanks == 1,
                      "it targets using PME rank(s) but the simulation is using only one rank");

    std::optional<std::string_view> pmeOptionArgument = commandLine.argumentOf("-pme");
    const bool                      commandLineTargetsPmeOnGpu =
            !pmeOptionArgument.has_value() || pmeOptionArgument.value() == "gpu";
    if (commandLineTargetsPmeOnGpu)
    {
        std::string prefix = "Test targets running PME on GPUs, but is being skipped because ";
        messages.appendIf(getCompatibleDevices(s_hwinfo->deviceInfoList).empty(),
                          "it targets GPU execution, but is being skipped because no compatible "
                          "devices were detected");
        messages.appendIf(!commandLineTargetsPmeOnlyRanks && numRanks > 1,
                          "it targets PME decomposition, but that is not supported");

        std::string errorMessage;
        messages.appendIf(!pme_gpu_supports_build(&errorMessage), errorMessage);
        messages.appendIf(!pme_gpu_supports_hardware(*s_hwinfo, &errorMessage), errorMessage);
        // A check on whether the .tpr is supported for PME on GPUs is
        // not needed, because it is supported by design.
    }
    return messages;
}

void PmeTest::checkEnergies(const bool usePmeTuning) const
{
    // Only the master rank should do this I/O intensive operation
    if (gmx_node_rank() != 0)
    {
        return;
    }

    TestReferenceData    refData(makeRefDataFileName());
    TestReferenceChecker rootChecker(refData.rootChecker());

    auto energyReader = openEnergyFileToReadTerms(
            runner_.edrFileName_, { "Coul. recip.", "Total Energy", "Kinetic En." });
    auto conservedChecker  = rootChecker.checkCompound("Energy", "Conserved");
    auto reciprocalChecker = rootChecker.checkCompound("Energy", "Reciprocal");
    // PME tuning causes differing grids and differing
    // reciprocal energy, so we don't check against the same
    // reciprocal energy computed on the CPU
    if (usePmeTuning)
    {
        reciprocalChecker.disableUnusedEntriesCheck();
    }
    bool firstIteration = true;
    while (energyReader->readNextFrame())
    {
        const EnergyFrame& frame            = energyReader->frame();
        const std::string  stepName         = frame.frameName();
        const real         conservedEnergy  = frame.at("Total Energy");
        const real         reciprocalEnergy = frame.at("Coul. recip.");
        if (firstIteration)
        {
            // use first step values as references for tolerance
            const real startingKineticEnergy = frame.at("Kinetic En.");
            const auto conservedTolerance = relativeToleranceAsFloatingPoint(startingKineticEnergy, 2e-5);
            const auto reciprocalTolerance = relativeToleranceAsFloatingPoint(reciprocalEnergy, 3e-5);
            reciprocalChecker.setDefaultTolerance(reciprocalTolerance);
            conservedChecker.setDefaultTolerance(conservedTolerance);
            firstIteration = false;
        }
        conservedChecker.checkReal(conservedEnergy, stepName.c_str());
        // When not using PME tuning, the reciprocal energy is
        // reproducible enough to check.
        if (!usePmeTuning)
        {
            reciprocalChecker.checkReal(reciprocalEnergy, stepName.c_str());
        }
    }
}

TEST_P(PmeTest, Runs)
{
    auto [pmeTestFlavor, mdrunCommandLine] = GetParam();
    CommandLine commandLine(splitString(mdrunCommandLine));

    MessageStringCollector skipMessages = PmeTest::getSkipMessagesIfNecessary(commandLine);
    if (!skipMessages.isEmpty())
    {
        GTEST_SKIP() << skipMessages.toString();
    }

    // When using PME tuning on a short mdrun, nstlist needs to be set very short
    // so that the tuning might do something while the test is running.
    const bool usePmeTuning = commandLine.contains("-tunepme");
    if (usePmeTuning)
    {
        commandLine.addOption("-nstlist", 1);
    }

    // Run mdrun on the tpr file that was built in SetUpTestSuite()
    runner_.tprFileName_ = s_tprFileNames[pmeTestFlavor];
    ASSERT_EQ(0, runner_.callMdrun(commandLine));

    // Check the contents of the edr file
    checkEnergies(usePmeTuning);
}

// To keep test execution time down, we check auto and pme tuning only
// in the Basic case.
//
// Note that some of these cases can only run when there is one MPI
// rank and some require more than one MPI rank. CTest has been
// instructed to run the test binary twice, with respectively one and
// two ranks, so that all tests that can run do run. The test binaries
// consider the hardware and rank count and skip those tests that they
// cannot run.
const auto c_reproducesEnergies = ::testing::ValuesIn(std::vector<PmeTestParameters>{ {
        // Here are all tests without a PME-only rank. These can
        // always run with a single rank, but can only run with two
        // ranks when not targeting GPUS.
        { PmeTestFlavor::Basic, "-notunepme -npme 0 -pme cpu" },
        { PmeTestFlavor::Basic, "-notunepme -npme 0 -pme auto" },
        { PmeTestFlavor::Basic, "-notunepme -npme 0 -pme gpu -pmefft cpu" },
        { PmeTestFlavor::Basic, "-notunepme -npme 0 -pme gpu -pmefft gpu" },
        { PmeTestFlavor::Basic, "-notunepme -npme 0 -pme gpu -pmefft auto" },
        { PmeTestFlavor::WithWalls, "-notunepme -npme 0 -pme cpu" },
        { PmeTestFlavor::WithWalls, "-notunepme -npme 0 -pme gpu -pmefft cpu" },
        { PmeTestFlavor::WithWalls, "-notunepme -npme 0 -pme gpu -pmefft gpu" },
        // Here are all tests with a PME-only rank, which requires
        // more than one total rank
        { PmeTestFlavor::Basic, "-notunepme -npme 1 -pme cpu" },
        { PmeTestFlavor::Basic, "-notunepme -npme 1 -pme auto" },
        { PmeTestFlavor::Basic, "-notunepme -npme 1 -pme gpu -pmefft cpu" },
        { PmeTestFlavor::Basic, "-notunepme -npme 1 -pme gpu -pmefft gpu" },
        { PmeTestFlavor::Basic, "-notunepme -npme 1 -pme gpu -pmefft auto" },
        { PmeTestFlavor::WithWalls, "-notunepme -npme 1 -pme cpu" },
        { PmeTestFlavor::WithWalls, "-notunepme -npme 1 -pme gpu -pmefft cpu" },
        { PmeTestFlavor::WithWalls, "-notunepme -npme 1 -pme gpu -pmefft gpu" },
        // All tests with PME tuning here
        { PmeTestFlavor::Basic, "-tunepme -npme 0 -pme cpu" },
        { PmeTestFlavor::Basic, "-tunepme -npme 0 -pme gpu -pmefft cpu" },
        { PmeTestFlavor::Basic, "-tunepme -npme 0 -pme gpu -pmefft gpu" },
} });

INSTANTIATE_TEST_SUITE_P(ReproducesEnergies, PmeTest, c_reproducesEnergies, nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
