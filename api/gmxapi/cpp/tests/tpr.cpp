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

#include "gromacs/fileio/tpxio.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/topology/topology.h"
#include "testingconfiguration.h"

#include "gmxapi/compat/tpr.h"

namespace gmxapi
{

namespace testing
{

namespace
{

TEST_F(GmxApiTest, TprReader)
{
    const int nsteps = 2;
    makeTprFile(nsteps);
    auto& tprFilename = this->runner_.tprFileName_;

    t_inputrec irInstance;
    gmx_mtop_t mtop;
    t_state    state;
    read_tpx_state(tprFilename.c_str(), &irInstance, &state, &mtop);
    ASSERT_EQ(state.x.size(), state.natoms);
    ASSERT_EQ(state.natoms, 8);

    // Check assumptions about memory layout for t_state.
    // Coordinate storage is contiguous.
    EXPECT_EQ(reinterpret_cast<size_t>(static_cast<void*>(&state.x[1]))
                      - reinterpret_cast<size_t>(static_cast<void*>(&state.x[0])),
              sizeof(gmx::RVec));
    EXPECT_EQ(state.x.data(), &state.x[0]);
    EXPECT_EQ(static_cast<void*>(&state.x[0]), static_cast<void*>(&state.x[0][0]));
    EXPECT_EQ(state.x.end() - state.x.begin(), state.natoms);

    std::vector<gmx::RVec> value;
    value.assign(state.x.begin(), state.x.end());
    EXPECT_EQ(state.x.size(), state.natoms);

    auto structureSource = gmxapicompat::getStructureSource(*gmxapicompat::readTprFile(tprFilename));
    auto coordinatesBuffer = gmxapicompat::coordinates(*structureSource, real());
    EXPECT_EQ(coordinatesBuffer.itemSize, sizeof(real));
    EXPECT_EQ(coordinatesBuffer.shape[1], 3);
    EXPECT_EQ(coordinatesBuffer.shape[0], state.natoms);
    EXPECT_EQ(coordinatesBuffer.strides[1], sizeof(real));
    EXPECT_EQ(coordinatesBuffer.strides[0], sizeof(gmx::RVec));

    for (auto i = 0; i < state.natoms; ++i)
    {
        for (auto j = 0; j < 3; ++j)
        {
            EXPECT_EQ(value[i][j], *((static_cast<real*>(coordinatesBuffer.ptr) + j) + (3 * i)));
        }
    }

    std::vector<real> coordinates;
    const int         num_elements = coordinatesBuffer.shape[0] * coordinatesBuffer.shape[1];
    EXPECT_EQ(num_elements, state.natoms * 3);
    real* begin = static_cast<real*>(coordinatesBuffer.ptr);
    real* end   = begin + num_elements;
    coordinates.assign(begin, end);
    for (auto i = 0; i < state.natoms * 3; ++i)
    {
        EXPECT_EQ(coordinates[i], value[i / 3][i % 3]);
    }
}

TEST_F(GmxApiTest, TprWriter)
{
    auto&             fileManager    = this->fileManager_;
    auto              outputFilePath = fileManager.getTemporaryFilePath(".new.tpr");
    std::vector<real> coordinates;
    const int         nsteps = 2;
    makeTprFile(nsteps);
    auto& inputFilename = this->runner_.tprFileName_;
    ASSERT_STRNE(inputFilename.c_str(), outputFilePath.c_str());

    {
        auto structureSource =
                gmxapicompat::getStructureSource(*gmxapicompat::readTprFile(inputFilename));
        auto      coordinatesBuffer = gmxapicompat::coordinates(*structureSource, real());
        const int num_elements      = coordinatesBuffer.shape[0] * coordinatesBuffer.shape[1];
        real*     begin             = static_cast<real*>(coordinatesBuffer.ptr);
        real*     end               = begin + num_elements;
        coordinates.assign(begin, end);

        std::vector<real> outputCoordinates;
        outputCoordinates.reserve(coordinates.size());
        for (const auto& x : coordinates)
        {
            outputCoordinates.emplace_back(x + 1.);
        }
        coordinatesBuffer.ptr = outputCoordinates.data();

        auto tprBuilder = gmxapicompat::editTprFile(inputFilename);
        tprBuilder->set(coordinatesBuffer);

        tprBuilder->write(outputFilePath);
    }

    {
        auto inputStructureSource =
                gmxapicompat::getStructureSource(*gmxapicompat::readTprFile(inputFilename));
        auto inputCoordinatesBuffer = gmxapicompat::coordinates(*inputStructureSource, real());
        ASSERT_EQ(inputCoordinatesBuffer.strides[0], 3 * sizeof(real));
        ASSERT_EQ(inputCoordinatesBuffer.strides[1], sizeof(real));
        real* inputCoordinates = static_cast<real*>(inputCoordinatesBuffer.ptr);

        auto outputStructureSource =
                gmxapicompat::getStructureSource(*gmxapicompat::readTprFile(outputFilePath));
        auto  outputCoordinatesBuffer = gmxapicompat::coordinates(*outputStructureSource, real());
        real* outputCoordinates       = static_cast<real*>(outputCoordinatesBuffer.ptr);

        const size_t num_items = inputCoordinatesBuffer.shape[0] * inputCoordinatesBuffer.shape[1];
        for (size_t i = 0; i < num_items; ++i)
        {
            EXPECT_FLOAT_EQ(coordinates[i] + 1., outputCoordinates[i]);
        }
    }
}

} // end anonymous namespace
} // end namespace testing
} // end namespace gmxapi
