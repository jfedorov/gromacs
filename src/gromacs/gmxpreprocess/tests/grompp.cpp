/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021 by the GROMACS development team.
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
 * Tests for grompp directives parsing
 *
 * \author Eliane Briand <eliane@br.iand.fr>
 */

#include "gmxpre.h"

#include "gromacs/fileio/tpxio.h"
#include "gromacs/gmxpreprocess/grompp.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textreader.h"
#include "gromacs/utility/textwriter.h"
#include "gromacs/topology/topology.h"

#include "testutils/cmdlinetest.h"
#include "testutils/conftest.h"
#include "testutils/refdata.h"
#include "testutils/testfilemanager.h"
#include "testutils/textblockmatchers.h"

namespace gmx
{
namespace
{

using gmx::test::CommandLine;
using gmx::test::TestFileManager;

class GromppTest : public ::testing::Test
{
public:
    GromppTest() = default;

    /*! \brief Run grompp from command line based on a set of inputs
     *
     * \param mdpContent The content of an MDP file as string
     * \param tprFileName
     * \param structureFile The .gro
     * \param topologyFile The topology file name
     *
     * \return the exit code from the grompp command
     * */
    int runGrompp(const std::string mdpContent,
                  const std::string tprFileName       = "directives.tpr",
                  const std::string structureFileName = "directives.gro",
                  const std::string topologyFileName  = "directives.top")
    {
        CommandLine cmdline;
        cmdline.addOption("grompp");

        const std::string mdpInputFileName = fileManager_.getTemporaryFilePath("input_grompp.mdp");
        gmx::TextWriter::writeFileFromString(mdpInputFileName, mdpContent);
        cmdline.addOption("-f", mdpInputFileName);


        cmdline.addOption("-c", TestFileManager::getInputFilePath(structureFileName));
        cmdline.addOption("-p", TestFileManager::getInputFilePath(topologyFileName));

        std::string outTprFilename = fileManager_.getTemporaryFilePath(tprFileName);
        cmdline.addOption("-o", outTprFilename);
        return gmx_grompp(cmdline.argc(), cmdline.argv());
    }

protected:
    gmx::test::TestFileManager fileManager_;
};

TEST_F(GromppTest, DirectiveEdgeCaseAtomTypeNames)
{
    std::string mdpContentString_ =
            "title                   = Directive edge case test \n"
            "integrator              = md \n"
            "nsteps                  = 1 \n"
            "dt                      = 0.002 \n"
            "vdwtype                 = cutoff \n"
            "coulombtype             = cutoff \n"
            "tcoupl                  = no \n"
            "pcoupl                  = no \n"
            "pbc                     = xyz \n"
            "gen_vel                 = yes \n";
    std::string tprFileName = "directives.tpr";
    int         exitCode    = runGrompp(mdpContentString_, tprFileName);

    ASSERT_EQ(0, exitCode);


    std::string outTprFilename = fileManager_.getTemporaryFilePath(tprFileName);
    gmx_mtop_t  top_after;
    t_inputrec  ir_after;
    t_state     state;
    read_tpx_state(outTprFilename.c_str(), &ir_after, &state, &top_after);

    // Check atomic numbers (or lack thereof coded as -1)
    ASSERT_EQ(top_after.atomtypes.nr, 4);
    EXPECT_EQ(top_after.atomtypes.atomnumber[0], -1);
    EXPECT_EQ(top_after.atomtypes.atomnumber[1], 6);
    EXPECT_EQ(top_after.atomtypes.atomnumber[2], 7);
    EXPECT_EQ(top_after.atomtypes.atomnumber[3], -1);
}

#if HAVE_MUPARSER

TEST_F(GromppTest, ValidTransformationCoord)
{
    const char* inputMdpFile[] = {
        "pull = yes",
        "pull-ncoords = 1",
        "pull-group1-name = System",
        "pull-coord1-geometry = transformation",
        "pull-coord1-expression = 10",
    };
    ASSERT_EQ(0, runGrompp(joinStrings(inputMdpFile, "\n")));
}

TEST_F(GromppTest, InvalidTransformationCoord)
{
    const char* inputMdpFile[] = {
        "pull = yes",
        "pull-ncoords = 1",
        "pull-group1-name = System",
        "pull-coord1-geometry = transformation",
        "pull-coord1-expression = x1", // invalid expression -> grompp should fail
    };
    ASSERT_NE(0, runGrompp(joinStrings(inputMdpFile, "\n")));
}
#endif // HAVE_MUPARSER


} // namespace
} // namespace gmx
