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
#include "gromacs/ewald/ewald_utils.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gpu_utils/device_stream_manager.h"
#include "gromacs/mdlib/rf_util.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/pairlistset.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/nbnxm/pairsearch.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/listoflists.h"
#include "gromacs/utility/range.h"
#include "nblib/exception.h"
#include "nblib/gmxcalculator.h"
#include "nblib/nbnxmsetuphelpers.h"
#include "nblib/pbc.hpp"

namespace nblib
{

//! \brief client-side provided system description data
struct SystemDescription
{
    //! number of particles
    size_t numParticles_{ 0 };

    //! number of particle types
    size_t numParticleTypes_{ 0 };

    //! particle type id of all particles
    std::vector<int> particleTypeIdOfAllParticles_;

    //! Storage for parameters for short range interactions.
    std::vector<real> nonBondedParams_;

    //! electrostatic charges
    std::vector<real> charges_;

    //! flag for each particle to set LJ and Q interactions
    std::vector<int64_t> particleInfo_;

    //! Legacy matrix for box
    Box box_{ 0 };
} __attribute__((aligned(128)));


/*! \brief GROMACS non-bonded force calculation backend
 *
 * This class encapsulates the various GROMACS data structures and their interplay
 * from the NBLIB user. The class is a private member of the ForceCalculator and
 * is not intended for the public interface.
 *
 * Handles the task of storing the simulation problem description using the internal
 * representation used within GROMACS. It currently supports short range non-bonded
 * interactions (PP) on a single node CPU.
 *
 */
struct GmxBackendData
{
    //! exclusions in gmx format
    gmx::ListOfLists<int> exclusions_;

    //! Non-Bonded Verlet object for force calculation
    std::unique_ptr<nonbonded_verlet_t> nbv_;

    //! Only shift_vec is used
    t_forcerec forcerec_;

    //! Parameters for various interactions in the system
    interaction_const_t interactionConst_;

    //! Tasks to perform in an MD Step
    gmx::StepWorkload stepWork_;

    gmx::SimulationWorkload simulationWork_;

    //! Energies of different interaction types; currently only needed as an argument for dispatchNonbondedKernel
    gmx_enerdata_t enerd_{ 1, 0 };

    //! Non-bonded flop counter; currently only needed as an argument for dispatchNonbondedKernel
    t_nrnb nrnb_;

    //! Keep track of whether updatePairlist has been called at least once
    bool updatePairlistCalled{ false };
} __attribute__((aligned(128)));


//! \brief stores system description and parameters data needed to calculate non-bonded forces
struct NBForceCalculatorData
{
    //! \brief client-side provided system description data
    SystemDescription system_;

    //! \brief Gmx backend objects, employed for calculating the forces
    GmxBackendData backend_;
};

static void initCommonData(SystemDescription&     system,
                           gmx::ArrayRef<int>     particleTypeIdOfAllParticles,
                           gmx::ArrayRef<real>    nonBondedParams,
                           gmx::ArrayRef<real>    charges,
                           gmx::ArrayRef<int64_t> particleInteractionFlags)
{
    std::array inputSizes{ particleTypeIdOfAllParticles.size(),
                           charges.size(),
                           particleInteractionFlags.size() };
    if (static_cast<unsigned long>(std::count(begin(inputSizes), end(inputSizes), inputSizes[0]))
        != inputSizes.size())
    {
        throw InputException("input array size inconsistent");
    }

    int numParticleTypes = int(std::sqrt(std::round(nonBondedParams.size() / 2)));
    if (2 * numParticleTypes * numParticleTypes != int(nonBondedParams.size()))
    {
        throw InputException("Wrong size of nonBondedParams");
    }

    system.numParticles_     = particleTypeIdOfAllParticles.size();
    system.numParticleTypes_ = numParticleTypes;

    system.particleTypeIdOfAllParticles_ =
            std::vector<int>(particleTypeIdOfAllParticles.begin(), particleTypeIdOfAllParticles.end());

    system.nonBondedParams_ = std::vector<real>(nonBondedParams.begin(), nonBondedParams.end());
    system.charges_         = std::vector<real>(charges.begin(), charges.end());
    system.particleInfo_ =
            std::vector<int64_t>(particleInteractionFlags.begin(), particleInteractionFlags.end());
}

static void initCommonBackend(GmxBackendData&        backend,
                              const NBKernelOptions& options,
                              int                    numEnergyGroups,
                              gmx::ArrayRef<int>     exclusionRanges,
                              gmx::ArrayRef<int>     exclusionElements)
{
    // Set hardware params from the execution context
    setGmxNonBondedNThreads(options.numOpenMPThreads);

    // Set interaction constants struct
    backend.interactionConst_ = createInteractionConst(options);

    // Set up StepWorkload data
    backend.stepWork_ = createStepWorkload(options);

    // Set up gmx_enerdata_t (holds energy information)
    backend.enerd_ = gmx_enerdata_t{ numEnergyGroups, 0 };

    // Construct pair lists
    std::vector<int> exclusionRanges_(exclusionRanges.begin(), exclusionRanges.end());
    std::vector<int> exclusionElements_(exclusionElements.begin(), exclusionElements.end());
    backend.exclusions_ =
            gmx::ListOfLists<int>(std::move(exclusionRanges_), std::move(exclusionElements_));
}


class GmxNBForceCalculatorCpuImpl final
{
public:
    GmxNBForceCalculatorCpuImpl(gmx::ArrayRef<int>     particleTypeIdOfAllParticles,
                                gmx::ArrayRef<real>    nonBondedParams,
                                gmx::ArrayRef<real>    charges,
                                gmx::ArrayRef<int64_t> particleInteractionFlags,
                                gmx::ArrayRef<int>     exclusionRanges,
                                gmx::ArrayRef<int>     exclusionElements,
                                const NBKernelOptions& options);

    //! calculates a new pair list based on new coordinates (for every NS step)
    void updatePairlist(gmx::ArrayRef<const gmx::RVec> coordinates, const Box& box);

    //! Compute forces and return
    void compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                 const Box&                     box,
                 gmx::ArrayRef<gmx::RVec>       forceOutput);

private:
    //! Common data
    NBForceCalculatorData data_;
};

GmxNBForceCalculatorCpuImpl::GmxNBForceCalculatorCpuImpl(gmx::ArrayRef<int> particleTypeIdOfAllParticles,
                                                         gmx::ArrayRef<real> nonBondedParams,
                                                         gmx::ArrayRef<real> charges,
                                                         gmx::ArrayRef<int64_t> particleInteractionFlags,
                                                         gmx::ArrayRef<int>     exclusionRanges,
                                                         gmx::ArrayRef<int>     exclusionElements,
                                                         const NBKernelOptions& options)
{
    int numEnergyGroups = findNumEnergyGroups(particleInteractionFlags);
    initCommonData(data_.system_, particleTypeIdOfAllParticles, nonBondedParams, charges, particleInteractionFlags);
    initCommonBackend(data_.backend_, options, numEnergyGroups, exclusionRanges, exclusionElements);

    // Set up non-bonded verlet in the backend
    data_.backend_.nbv_ = createNbnxmCPU(
            data_.system_.numParticleTypes_, options, numEnergyGroups, data_.system_.nonBondedParams_);
}

void GmxNBForceCalculatorCpuImpl::updatePairlist(gmx::ArrayRef<const gmx::RVec> coordinates, const Box& box)
{
    const auto* legacyBox = box.legacyMatrix();
    data_.system_.box_    = box;
    updateForcerec(&data_.backend_.forcerec_, box.legacyMatrix());
    if (TRICLINIC(legacyBox))
    {
        throw InputException("Only rectangular unit-cells are supported here");
    }

    const rvec lowerCorner = { 0, 0, 0 };
    const rvec upperCorner = { legacyBox[dimX][dimX], legacyBox[dimY][dimY], legacyBox[dimZ][dimZ] };

    const real particleDensity = static_cast<real>(coordinates.size()) / det(legacyBox);

    // Put particles on a grid based on bounds specified by the box
    nbnxn_put_on_grid(data_.backend_.nbv_.get(),
                      legacyBox,
                      0,
                      lowerCorner,
                      upperCorner,
                      nullptr,
                      { 0, int(coordinates.size()) },
                      particleDensity,
                      data_.system_.particleInfo_,
                      coordinates,
                      0,
                      nullptr);

    data_.backend_.nbv_->constructPairlist(
            gmx::InteractionLocality::Local, data_.backend_.exclusions_, 0, &data_.backend_.nrnb_);

    // Set Particle Types and Charges and VdW params
    data_.backend_.nbv_->setAtomProperties(data_.system_.particleTypeIdOfAllParticles_,
                                           data_.system_.charges_,
                                           data_.system_.particleInfo_);
    data_.backend_.updatePairlistCalled = true;
}

void GmxNBForceCalculatorCpuImpl::compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                                          const Box&                     box,
                                          gmx::ArrayRef<gmx::RVec>       forceOutput)
{
    if (coordinateInput.size() != forceOutput.size())
    {
        throw InputException("coordinate array and force buffer size mismatch");
    }

    if (!data_.backend_.updatePairlistCalled)
    {
        throw InputException("compute called without updating pairlist at least once");
    }

    // update the box if changed
    if (!(data_.system_.box_ == box))
    {
        data_.system_.box_ = box;
        updateForcerec(&data_.backend_.forcerec_, box.legacyMatrix());
    }

    data_.backend_.stepWork_.computeVirial = false;
    data_.backend_.stepWork_.computeEnergy = false;

    // update the coordinates in the backend
    data_.backend_.nbv_->convertCoordinates(gmx::AtomLocality::Local, coordinateInput);

    data_.backend_.nbv_->dispatchNonbondedKernel(
            gmx::InteractionLocality::Local,
            data_.backend_.interactionConst_,
            data_.backend_.stepWork_,
            enbvClearFYes,
            data_.backend_.forcerec_.shift_vec,
            data_.backend_.enerd_.grpp.energyGroupPairTerms[data_.backend_.forcerec_.haveBuckingham
                                                                    ? NonBondedEnergyTerms::BuckinghamSR
                                                                    : NonBondedEnergyTerms::LJSR],
            data_.backend_.enerd_.grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR],
            &data_.backend_.nrnb_);

    data_.backend_.nbv_->atomdata_add_nbat_f_to_f(gmx::AtomLocality::All, forceOutput);
}


GmxNBForceCalculatorCpu::GmxNBForceCalculatorCpu(gmx::ArrayRef<int>  particleTypeIdOfAllParticles,
                                                 gmx::ArrayRef<real> nonBondedParams,
                                                 gmx::ArrayRef<real> charges,
                                                 gmx::ArrayRef<int64_t> particleInteractionFlags,
                                                 gmx::ArrayRef<int>     exclusionRanges,
                                                 gmx::ArrayRef<int>     exclusionElements,
                                                 const NBKernelOptions& options)
{
    impl_ = std::make_unique<GmxNBForceCalculatorCpuImpl>(particleTypeIdOfAllParticles,
                                                          nonBondedParams,
                                                          charges,
                                                          particleInteractionFlags,
                                                          exclusionRanges,
                                                          exclusionElements,
                                                          options);
}

GmxNBForceCalculatorCpu::~GmxNBForceCalculatorCpu() = default;

//! calculates a new pair list based on new coordinates (for every NS step)
void GmxNBForceCalculatorCpu::updatePairlist(gmx::ArrayRef<const gmx::RVec> coordinates, const Box& box)
{
    impl_->updatePairlist(coordinates, box);
}

//! Compute forces and return
void GmxNBForceCalculatorCpu::compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                                      const Box&                     box,
                                      gmx::ArrayRef<gmx::RVec>       forceOutput)
{
    impl_->compute(coordinateInput, box, forceOutput);
}

} // namespace nblib
