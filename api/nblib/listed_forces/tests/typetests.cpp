/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020,2021, by the GROMACS development team, led by
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
 * This implements basic nblib box tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "nblib/listed_forces/dataflow.hpp"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace nblib
{

//! Number of atoms used in these tests.
constexpr int c_numAtoms = 4;

namespace
{

//! Coordinates for testing
std::vector<std::vector<gmx::RVec>> c_coordinatesForTests = {
    { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.2 }, { 0.005, 0.0, 0.1 }, { -0.001, 0.1, 0.0 } },
    { { 0.5, 0.0, 0.0 }, { 0.5, 0.0, 0.15 }, { 0.5, 0.07, 0.22 }, { 0.5, 0.18, 0.22 } },
    { { -0.1143, -0.0282, 0.0 }, { 0.0, 0.0434, 0.0 }, { 0.1185, -0.0138, 0.0 }, { -0.0195, 0.1498, 0.0 } }
};
// Parameters for harmonic bonds
std::vector<InteractionIndex<HarmonicBondType>> c_HarmonicBondIndices{ { 0, 1, 0 }, { 1, 2, 0 }, { 2, 3, 0 } };
std::vector<std::vector<HarmonicBondType>> c_InputHarmonicBond = { { HarmonicBondType(500, 0.15) } };

// Parameters for harmonic angles
std::vector<InteractionIndex<HarmonicAngle>> c_HarmonicAngleIndices{ { 0, 1, 2, 0 }, { 1, 2, 3, 0 } };
std::vector<std::vector<HarmonicAngle>> c_InputHarmonicAngle = { { HarmonicAngle(Degrees(100), 50.0) } };

//! Function types for testing dihedrals. Add new terms at the end.
std::vector<std::vector<ProperDihedral>> c_InputDihs = { { { ProperDihedral(Degrees(-105.0), 15.0, 2) } } /*, { ImproperDihedral(100.0, 50.0) }*/ };
// Todo: update test setup to allow more than one interaction type and add the following to the inputs
// std::vector<std::vector<RyckaertBellemanDihedral>> c_InputDihs = { { RyckaertBellemanDihedral({ -7.35, 13.6, 8.4, -16.7, 1.3, 12.4 }) } };

} // namespace

template<class Interaction>
class ListedForcesBase
{
public:
    std::vector<Interaction>                   input_;
    std::vector<gmx::RVec>                     x_;
    std::vector<InteractionIndex<Interaction>> indices_;
    PbcHolder                                  pbcHolder_;
    gmx::test::TestReferenceData               refData_;
    gmx::test::TestReferenceChecker            checker_;
    std::vector<gmx::RVec>                     forces_;
    real                                       energy_;

    ListedForcesBase(std::vector<Interaction>                   input,
                     std::vector<gmx::RVec>                     coordinates,
                     std::vector<InteractionIndex<Interaction>> indices) :
        input_(std::move(input)),
        x_(std::move(coordinates)),
        indices_(std::move(indices)),
        pbcHolder_(PbcType::Xyz, Box(1.5)),
        checker_(refData_.rootChecker()),
        forces_(c_numAtoms, gmx::RVec{ 0, 0, 0 })
    {
        energy_ = computeForces(gmx::ArrayRef<const InteractionIndex<Interaction>>(indices_),
                                gmx::ArrayRef<const Interaction>(input_),
                                x_,
                                &forces_,
                                pbcHolder_);
    }

    void checkForcesAndEnergies()
    {
        // We need quite specific tolerances here since angle functions
        // etc. are not very precise and reproducible.
        auto tolerances = gmx::test::FloatingPointTolerance(
                singleAbsoluteTolerance_, 1.0e-12, singleRelativeTolerance_, 1.0e-12, 1000, 100, false);
        checker_.setDefaultTolerance(tolerances);
        checker_.checkReal(energy_, "Epot");
        checker_.checkSequence(std::begin(forces_), std::end(forces_), "forces");
    }

    void setSingleTolerance(float relative, float absolute)
    {
        singleRelativeTolerance_ = relative;
        singleAbsoluteTolerance_ = absolute;
    }

private:
    float singleRelativeTolerance_ = 1.0e-12;
    float singleAbsoluteTolerance_ = 1.0e-12;
};

class ProperDihedralTest :
    public ListedForcesBase<ProperDihedral>,
    public testing::TestWithParam<std::tuple<std::vector<ProperDihedral>, std::vector<gmx::RVec>>>
{
    using Base = ListedForcesBase<ProperDihedral>;

public:
    ProperDihedralTest() :
        Base(std::get<0>(GetParam()), std::get<1>(GetParam()), { { 0, 1, 2, 3, 0 } })
    {
    }
};

TEST_P(ProperDihedralTest, CheckListed)
{
    checkForcesAndEnergies();
}

INSTANTIATE_TEST_CASE_P(FourCenter,
                        ProperDihedralTest,
                        ::testing::Combine(::testing::ValuesIn(c_InputDihs),
                                           ::testing::ValuesIn(c_coordinatesForTests)));

class HarmonicBondTest :
    public ListedForcesBase<HarmonicBondType>,
    public testing::TestWithParam<std::tuple<std::vector<HarmonicBondType>, std::vector<gmx::RVec>>>
{
    using Base = ListedForcesBase<HarmonicBondType>;

public:
    HarmonicBondTest() :
        Base(std::get<0>(GetParam()), std::get<1>(GetParam()), c_HarmonicBondIndices)
    {
    }
};

TEST_P(HarmonicBondTest, CheckListed)
{
    checkForcesAndEnergies();
}

INSTANTIATE_TEST_CASE_P(TwoCenter,
                        HarmonicBondTest,
                        ::testing::Combine(::testing::ValuesIn(c_InputHarmonicBond),
                                           ::testing::ValuesIn(c_coordinatesForTests)));

class HarmonicAngleTest :
    public ListedForcesBase<HarmonicAngle>,
    public testing::TestWithParam<std::tuple<std::vector<HarmonicAngle>, std::vector<gmx::RVec>>>
{
    using Base = ListedForcesBase<HarmonicAngle>;

public:
    HarmonicAngleTest() :
        Base(std::get<0>(GetParam()), std::get<1>(GetParam()), c_HarmonicAngleIndices)
    {
    }
};

TEST_P(HarmonicAngleTest, CheckListed)
{
    setSingleTolerance(1e-12, 2e-3);
    checkForcesAndEnergies();
}

INSTANTIATE_TEST_CASE_P(ThreeCenter,
                        HarmonicAngleTest,
                        ::testing::Combine(::testing::ValuesIn(c_InputHarmonicAngle),
                                           ::testing::ValuesIn(c_coordinatesForTests)));

} // namespace nblib
