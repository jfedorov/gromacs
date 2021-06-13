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
 * \brief Implements a force calculator based on GROMACS data structures.
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "nblib/gmxcalculator.h"
#include "gromacs/ewald/ewald_utils.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/mdlib/rf_util.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/utility/listoflists.h"
#include "gromacs/utility/range.h"
#include "nblib/exception.h"
#include "nbnxmsetuphelpers.h"
#include "nblib/simulationstate.h"

namespace nblib
{

GmxForceCalculator::GmxForceCalculator()
{
    enerd_            = std::make_unique<gmx_enerdata_t>(1, 0);
    forcerec_         = std::make_unique<t_forcerec>();
    interactionConst_ = std::make_unique<interaction_const_t>();
    stepWork_         = std::make_unique<gmx::StepWorkload>();
    nrnb_             = std::make_unique<t_nrnb>();
}

GmxForceCalculator::~GmxForceCalculator() = default;

void GmxForceCalculator::compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                                 const Box&                     box,
                                 gmx::ArrayRef<gmx::RVec>       forceOutput)
{
    if (coordinateInput.size() != forceOutput.size())
    {
        throw InputException("coordinate array and force buffer size mismatch");
    }

    // update the box if changed
    if (!(box_ == box))
    {
        box_ = box;
        updateForcerec(forcerec_.get(), box.legacyMatrix());
    }

    // update the coordinates in the backend
    nbv_->convertCoordinates(gmx::AtomLocality::Local, coordinateInput);

    nbv_->dispatchNonbondedKernel(
            gmx::InteractionLocality::Local,
            *interactionConst_,
            *stepWork_,
            enbvClearFYes,
            forcerec_->shift_vec,
            enerd_->grpp.energyGroupPairTerms[forcerec_->haveBuckingham ? NonBondedEnergyTerms::BuckinghamSR
                                                                        : NonBondedEnergyTerms::LJSR],
            enerd_->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR],
            nrnb_.get());

    nbv_->atomdata_add_nbat_f_to_f(gmx::AtomLocality::All, forceOutput);
}

void GmxForceCalculator::setParticlesOnGrid(gmx::ArrayRef<const gmx::RVec> coordinates, const Box& box)
{
    const auto* legacyBox = box.legacyMatrix();
    box_                  = box;
    updateForcerec(forcerec_.get(), box.legacyMatrix());
    if (TRICLINIC(legacyBox))
    {
        throw InputException("Only rectangular unit-cells are supported here");
    }

    const rvec lowerCorner = { 0, 0, 0 };
    const rvec upperCorner = { legacyBox[dimX][dimX], legacyBox[dimY][dimY], legacyBox[dimZ][dimZ] };

    const real particleDensity = static_cast<real>(coordinates.size()) / det(legacyBox);

    nbnxn_put_on_grid(nbv_.get(),
                      legacyBox,
                      0,
                      lowerCorner,
                      upperCorner,
                      nullptr,
                      { 0, int(coordinates.size()) },
                      particleDensity,
                      particleInfo_,
                      coordinates,
                      0,
                      nullptr);

    // Construct pair lists
    std::vector<int> exclusionRanges = exclusions_.ListRanges;
    std::vector<int> exclusionElements = exclusions_.ListElements;
    gmx::ListOfLists<int> exclusions(std::move(exclusionRanges), std::move(exclusionElements));
    nbv_->constructPairlist(gmx::InteractionLocality::Local, exclusions, 0, nrnb_.get());

    // Set Particle Types and Charges and VdW params
    nbv_->setAtomProperties(particleTypeIdOfAllParticles_, charges_, particleInfo_);
}

} // namespace nblib
