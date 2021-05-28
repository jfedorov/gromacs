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
 * Tests for gmx grompp.
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 */

#include "gmxpre.h"

#include "gromacs/gmxpreprocess/grompp.h"

#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/cmdlinetest.h"
#include "testutils/conftest.h"
#include "testutils/refdata.h"
#include "testutils/stdiohelper.h"
#include "testutils/testfilemanager.h"


namespace gmx
{
namespace test
{
namespace
{

class GromppTest : public CommandLineTestBase
{
public:
    void setTopology(std::string_view top)
    {
        caller_.addOption("-p", top.data());
    }
    void setInputCoordinates(std::string_view coords)
    {
        caller_.addOption("-c", coords.data());
    }
    void setRestraintCoordinates(std::string_view coords)
    {
        caller_.addOption("-r", coords.data());
    }

    void writeAndSetSimpleMdp()
    {
        std::string mdp             = fileManager().getTemporaryFilePath(".mdp");
        std::string mdpFileContents = gmx::formatString(
                "cutoff-scheme = verlet\n"
                "rcoulomb      = 0.85\n"
                "rvdw          = 0.85\n"
                "rlist         = 0.85\n");
        gmx::TextWriter::writeFileFromString(mdp, mdpFileContents);
        caller_.addOption("-f", mdp);
    }
    //! Returns gmx_grompp's return code.
    int executeGrompp() { return gmx_grompp(caller_.argc(), caller_.argv()); }

private:
    CommandLine caller_ = commandLine();
};
TEST_F(GromppTest, SucceedsWithSimpleRestraints)
{
    writeAndSetSimpleMdp();
    setTopology(TestFileManager::getInputFilePath("4water_restraints.top"));
    setInputCoordinates(TestFileManager::getInputFilePath("4water.gro"));
    setRestraintCoordinates(TestFileManager::getInputFilePath("4water.gro"));
    EXPECT_EQ(executeGrompp(), 0);
}

TEST_F(GromppTest, FailsWithOverlappingRestraints)
{
    writeAndSetSimpleMdp();
    setTopology(TestFileManager::getInputFilePath("4water_overlapping_restraints.top"));
    setInputCoordinates(TestFileManager::getInputFilePath("4water.gro"));
    setRestraintCoordinates(TestFileManager::getInputFilePath("4water.gro"));
    EXPECT_THROW_GMX(executeGrompp(), InvalidInputError);
}

} // namespace

} // namespace test

} // namespace gmx
