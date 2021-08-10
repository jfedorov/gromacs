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
#include "nblib/virials.h"

namespace nblib
{

//! \brief client-side provided system description data
struct SystemDescription
{
    SystemDescription() = default;

    SystemDescription(gmx::ArrayRef<int>     particleTypeIdOfAllParticles,
                      gmx::ArrayRef<real>    nonBondedParams,
                      gmx::ArrayRef<real>    charges,
                      gmx::ArrayRef<int64_t> particleInteractionFlags);

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
};


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
    GmxBackendData() = default;
    GmxBackendData(const NBKernelOptions& options,
                   int                    numEnergyGroups,
                   gmx::ArrayRef<int>     exclusionRanges,
                   gmx::ArrayRef<int>     exclusionElements);

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
};


SystemDescription::SystemDescription(gmx::ArrayRef<int>     particleTypeIdOfAllParticles,
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

    int numParticleTypes = int(std::round(std::sqrt(nonBondedParams.size() / 2)));
    if (2 * numParticleTypes * numParticleTypes != int(nonBondedParams.size()))
    {
        throw InputException("Wrong size of nonBondedParams");
    }

    numParticles_     = particleTypeIdOfAllParticles.size();
    numParticleTypes_ = numParticleTypes;

    particleTypeIdOfAllParticles_ =
            std::vector<int>(particleTypeIdOfAllParticles.begin(), particleTypeIdOfAllParticles.end());

    nonBondedParams_ = std::vector<real>(nonBondedParams.begin(), nonBondedParams.end());
    charges_         = std::vector<real>(charges.begin(), charges.end());
    particleInfo_ =
            std::vector<int64_t>(particleInteractionFlags.begin(), particleInteractionFlags.end());
}

GmxBackendData::GmxBackendData(const NBKernelOptions& options,
                               int                    numEnergyGroups,
                               gmx::ArrayRef<int>     exclusionRanges,
                               gmx::ArrayRef<int>     exclusionElements)
{
    // Set hardware params from the execution context
    setGmxNonBondedNThreads(options.numOpenMPThreads);

    // Set interaction constants struct
    interactionConst_ = createInteractionConst(options);

    // Set up StepWorkload data
    stepWork_ = createStepWorkload(options);

    // Set up gmx_enerdata_t (holds energy information)
    enerd_ = gmx_enerdata_t{ numEnergyGroups, 0 };

    // Construct pair lists
    std::vector<int> exclusionRanges_(exclusionRanges.begin(), exclusionRanges.end());
    std::vector<int> exclusionElements_(exclusionElements.begin(), exclusionElements.end());
    exclusions_ = gmx::ListOfLists<int>(std::move(exclusionRanges_), std::move(exclusionElements_));
}


class GmxNBForceCalculatorCpu::CpuImpl final
{
public:
    CpuImpl(gmx::ArrayRef<int>     particleTypeIdOfAllParticles,
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

    //! Compute forces and virial tensor
    void compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                 const Box&                     box,
                 gmx::ArrayRef<gmx::RVec>       forceOutput,
                 gmx::ArrayRef<real>            virialOutput);

    //! Compute forces, virial tensor and potential energies
    void compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                 const Box&                     box,
                 gmx::ArrayRef<gmx::RVec>       forceOutput,
                 gmx::ArrayRef<real>            virialOutput,
                 gmx::ArrayRef<real>            energyOutput);

private:
    //! \brief client-side provided system description data
    SystemDescription system_;

    //! \brief Gmx backend objects, employed for calculating the forces
    GmxBackendData backend_;
};

GmxNBForceCalculatorCpu::CpuImpl::CpuImpl(gmx::ArrayRef<int>     particleTypeIdOfAllParticles,
                                          gmx::ArrayRef<real>    nonBondedParams,
                                          gmx::ArrayRef<real>    charges,
                                          gmx::ArrayRef<int64_t> particleInteractionFlags,
                                          gmx::ArrayRef<int>     exclusionRanges,
                                          gmx::ArrayRef<int>     exclusionElements,
                                          const NBKernelOptions& options) :
    system_(SystemDescription(particleTypeIdOfAllParticles, nonBondedParams, charges, particleInteractionFlags)),
    backend_(GmxBackendData(options, findNumEnergyGroups(particleInteractionFlags), exclusionRanges, exclusionElements))
{
    // Set up non-bonded verlet in the backend
    backend_.nbv_ = createNbnxmCPU(system_.numParticleTypes_,
                                   options,
                                   findNumEnergyGroups(particleInteractionFlags),
                                   system_.nonBondedParams_);
}

void GmxNBForceCalculatorCpu::CpuImpl::updatePairlist(gmx::ArrayRef<const gmx::RVec> coordinates,
                                                      const Box&                     box)
{
    const auto* legacyBox = box.legacyMatrix();
    system_.box_          = box;
    updateForcerec(&backend_.forcerec_, box.legacyMatrix());
    if (TRICLINIC(legacyBox))
    {
        throw InputException("Only rectangular unit-cells are supported here");
    }

    const rvec lowerCorner = { 0, 0, 0 };
    const rvec upperCorner = { legacyBox[dimX][dimX], legacyBox[dimY][dimY], legacyBox[dimZ][dimZ] };

    const real particleDensity = static_cast<real>(coordinates.size()) / det(legacyBox);

    // Put particles on a grid based on bounds specified by the box
    nbnxn_put_on_grid(backend_.nbv_.get(),
                      legacyBox,
                      0,
                      lowerCorner,
                      upperCorner,
                      nullptr,
                      { 0, int(coordinates.size()) },
                      particleDensity,
                      system_.particleInfo_,
                      coordinates,
                      0,
                      nullptr);

    backend_.nbv_->constructPairlist(
            gmx::InteractionLocality::Local, backend_.exclusions_, 0, &backend_.nrnb_);

    // Set Particle Types and Charges and VdW params
    backend_.nbv_->setAtomProperties(
            system_.particleTypeIdOfAllParticles_, system_.charges_, system_.particleInfo_);
    backend_.updatePairlistCalled = true;
}

void GmxNBForceCalculatorCpu::CpuImpl::compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                                               const Box&                     box,
                                               gmx::ArrayRef<gmx::RVec>       forceOutput,
                                               gmx::ArrayRef<real>            virialOutput,
                                               gmx::ArrayRef<real>            energyOutput)
{
    if (coordinateInput.size() != forceOutput.size())
    {
        throw InputException("coordinate array and force buffer size mismatch");
    }

    if (!backend_.updatePairlistCalled)
    {
        throw InputException("compute called without updating pairlist at least once");
    }

    // update the box if changed
    if (!(system_.box_ == box))
    {
        system_.box_ = box;
        updateForcerec(&backend_.forcerec_, box.legacyMatrix());
    }

    bool computeVirial               = !virialOutput.empty();
    bool computeEnergies             = !energyOutput.empty();
    backend_.stepWork_.computeVirial = computeVirial;
    backend_.stepWork_.computeEnergy = computeEnergies;

    // update the coordinates in the backend
    backend_.nbv_->convertCoordinates(gmx::AtomLocality::Local, coordinateInput);

    backend_.nbv_->dispatchNonbondedKernel(
            gmx::InteractionLocality::Local,
            backend_.interactionConst_,
            backend_.stepWork_,
            enbvClearFYes,
            backend_.forcerec_.shift_vec,
            backend_.enerd_.grpp.energyGroupPairTerms[backend_.forcerec_.haveBuckingham ? NonBondedEnergyTerms::BuckinghamSR
                                                                                        : NonBondedEnergyTerms::LJSR],
            backend_.enerd_.grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR],
            &backend_.nrnb_);

    backend_.nbv_->atomdata_add_nbat_f_to_f(gmx::AtomLocality::All, forceOutput);

    if (computeVirial)
    {
        // calculate shift forces and turn into an array ref
        std::vector<Vec3> shiftForcesVector(gmx::c_numShiftVectors, Vec3(0.0, 0.0, 0.0));
        nbnxn_atomdata_add_nbat_fshift_to_fshift(*backend_.nbv_->nbat, shiftForcesVector);
        auto shiftForcesRef = constArrayRefFromArray(shiftForcesVector.data(), shiftForcesVector.size());

        std::vector<Vec3> shiftVectorsArray(gmx::c_numShiftVectors);

        // copy shift vectors from ForceRec
        std::copy(backend_.forcerec_.shift_vec.begin(),
                  backend_.forcerec_.shift_vec.end(),
                  shiftVectorsArray.begin());

        computeVirialTensor(
                coordinateInput, forceOutput, shiftVectorsArray, shiftForcesRef, box, virialOutput);
    }

    // extract term energies (per interaction type)
    if (computeEnergies)
    {
        int nGroupPairs = backend_.enerd_.grpp.nener;
        if (int(energyOutput.size()) != int(NonBondedEnergyTerms::Count) * nGroupPairs)
        {
            throw InputException("Array size for energy output is wrong\n");
        }

        for (int eg = 0; eg < int(NonBondedEnergyTerms::Count); ++eg)
        {
            std::copy(begin(backend_.enerd_.grpp.energyGroupPairTerms[eg]),
                      end(backend_.enerd_.grpp.energyGroupPairTerms[eg]),
                      energyOutput.begin() + eg * nGroupPairs);
        }
    }
}

void GmxNBForceCalculatorCpu::CpuImpl::compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                                               const Box&                     box,
                                               gmx::ArrayRef<gmx::RVec>       forceOutput)
{
    // compute forces and fill in force buffer
    compute(coordinateInput, box, forceOutput, gmx::ArrayRef<real>{}, gmx::ArrayRef<real>{});
}

void GmxNBForceCalculatorCpu::CpuImpl::compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                                               const Box&                     box,
                                               gmx::ArrayRef<gmx::RVec>       forceOutput,
                                               gmx::ArrayRef<real>            virialOutput)
{
    // compute forces and fill in force buffer
    compute(coordinateInput, box, forceOutput, virialOutput, gmx::ArrayRef<real>{});
}


class GmxNBForceCalculatorGpu::GpuImpl final
{
public:
    GpuImpl(gmx::ArrayRef<int>       particleTypeIdOfAllParticles,
            gmx::ArrayRef<real>      nonBondedParams,
            gmx::ArrayRef<real>      charges,
            gmx::ArrayRef<int64_t>   particleInteractionFlags,
            gmx::ArrayRef<int>       exclusionRanges,
            gmx::ArrayRef<int>       exclusionElements,
            const NBKernelOptions&   options,
            const DeviceInformation& deviceInfo);

    //! calculates a new pair list based on new coordinates (for every NS step)
    void updatePairlist(gmx::ArrayRef<const gmx::RVec> coordinates, const Box& box);

    //! Compute forces and return
    void compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                 const Box&                     box,
                 gmx::ArrayRef<gmx::RVec>       forceOutput);

private:
    //! \brief client-side provided system description data
    SystemDescription system_;

    //! \brief Gmx backend objects, employed for calculating the forces
    GmxBackendData backend_;

    //! Stream and context manager
    std::shared_ptr<gmx::DeviceStreamManager> deviceStreamManager_;
};

GmxNBForceCalculatorGpu::GpuImpl::GpuImpl(gmx::ArrayRef<int>       particleTypeIdOfAllParticles,
                                          gmx::ArrayRef<real>      nonBondedParams,
                                          gmx::ArrayRef<real>      charges,
                                          gmx::ArrayRef<int64_t>   particleInteractionFlags,
                                          gmx::ArrayRef<int>       exclusionRanges,
                                          gmx::ArrayRef<int>       exclusionElements,
                                          const NBKernelOptions&   options,
                                          const DeviceInformation& deviceInfo) :
    system_(SystemDescription(particleTypeIdOfAllParticles, nonBondedParams, charges, particleInteractionFlags)),
    // multiple energy groups not supported on GPUs
    backend_(GmxBackendData(options, 1, exclusionRanges, exclusionElements))
{
    backend_.simulationWork_ = createSimulationWorkloadGpu(options);

    // set DeviceInformation and create the DeviceStreamManager
    deviceStreamManager_ = createDeviceStreamManager(deviceInfo, backend_.simulationWork_);

    // create the nonbonded_verlet_t instance, corresponds roughly to init_nb_verlet in GROMACS
    backend_.nbv_ = createNbnxmGPU(system_.numParticleTypes_,
                                   options,
                                   system_.nonBondedParams_,
                                   backend_.interactionConst_,
                                   deviceStreamManager_);
}

void GmxNBForceCalculatorGpu::GpuImpl::updatePairlist(gmx::ArrayRef<const gmx::RVec> coordinates,
                                                      const Box&                     box)
{
    const auto* legacyBox = box.legacyMatrix();
    system_.box_          = box;
    updateForcerec(&backend_.forcerec_, box.legacyMatrix());
    if (TRICLINIC(legacyBox))
    {
        throw InputException("Only rectangular unit-cells are supported here");
    }

    const rvec lowerCorner = { 0, 0, 0 };
    const rvec upperCorner = { legacyBox[dimX][dimX], legacyBox[dimY][dimY], legacyBox[dimZ][dimZ] };

    const real particleDensity = real(coordinates.size()) / det(legacyBox);

    nbnxn_put_on_grid(backend_.nbv_.get(),
                      legacyBox,
                      0,
                      lowerCorner,
                      upperCorner,
                      nullptr,
                      { 0, int(coordinates.size()) },
                      particleDensity,
                      system_.particleInfo_,
                      coordinates,
                      0,
                      nullptr);

    // construct the pairlist
    backend_.nbv_->constructPairlist(
            gmx::InteractionLocality::Local, backend_.exclusions_, 0, &backend_.nrnb_);

    // particle types, charges and info flags are stored in the nbnxm_atomdata_t member of
    // nonbonded_verlet_t. For now, we replicate the current GROMACS implementation to
    // add types, charges and infos to nbnxm_atomdata_t separately from the nonbondedParameters_.
    backend_.nbv_->setAtomProperties(
            system_.particleTypeIdOfAllParticles_, system_.charges_, system_.particleInfo_);

    // allocates f, xq, atom_types and lj_comb in the nbnxm_atomdata_gpu on the GPU
    // also uploads nbfp (LJ) parameters to the GPU
    Nbnxm::gpu_init_atomdata(backend_.nbv_->gpu_nbv, backend_.nbv_->nbat.get());

    // this is needed to for atomdata_add_nbat_f_to_f, otherwise it thinks there's no work done
    // it just sets a flag to true in case the pair list is not empty
    backend_.nbv_->setupGpuShortRangeWork(nullptr, gmx::InteractionLocality::Local);
    backend_.updatePairlistCalled = true;
}

void GmxNBForceCalculatorGpu::GpuImpl::compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                                               const Box&                     box,
                                               gmx::ArrayRef<gmx::RVec>       forceOutput)
{
    if (!backend_.updatePairlistCalled)
    {
        throw InputException("compute called without updating pairlist at least once");
    }

    // update the box if changed
    if (!(system_.box_ == box))
    {
        system_.box_ = box;
        updateForcerec(&backend_.forcerec_, box.legacyMatrix());
    }

    // needed for non-neighbor search steps
    backend_.nbv_->convertCoordinates(gmx::AtomLocality::Local, coordinateInput);

    // copy the coordinates to GPU buffer
    Nbnxm::gpu_copy_xq_to_gpu(backend_.nbv_->gpu_nbv, backend_.nbv_->nbat.get(), gmx::AtomLocality::Local);

    // compute forces
    backend_.nbv_->dispatchNonbondedKernel(
            gmx::InteractionLocality::Local,
            backend_.interactionConst_,
            backend_.stepWork_,
            enbvClearFYes,
            backend_.forcerec_.shift_vec,
            backend_.enerd_.grpp.energyGroupPairTerms[backend_.forcerec_.haveBuckingham ? NonBondedEnergyTerms::BuckinghamSR
                                                                                        : NonBondedEnergyTerms::LJSR],
            backend_.enerd_.grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR],
            &backend_.nrnb_);

    // copy force from the Gpu Nbnxm into the Cpu version
    Nbnxm::gpu_launch_cpyback(
            backend_.nbv_->gpu_nbv, backend_.nbv_->nbat.get(), backend_.stepWork_, gmx::AtomLocality::Local);

    gmx::ArrayRef<gmx::RVec> nullShiftForce;
    Nbnxm::gpu_wait_finish_task(
            backend_.nbv_->gpu_nbv,
            backend_.stepWork_,
            gmx::AtomLocality::Local,
            backend_.enerd_.grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR].data(),
            backend_.enerd_.grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR].data(),
            nullShiftForce,
            nullptr);

    // copy non-bonded forces from the nbv-internal buffer to the user provided force output buffer
    backend_.nbv_->atomdata_add_nbat_f_to_f(gmx::AtomLocality::Local, forceOutput);
}


GmxNBForceCalculatorCpu::GmxNBForceCalculatorCpu(gmx::ArrayRef<int>  particleTypeIdOfAllParticles,
                                                 gmx::ArrayRef<real> nonBondedParams,
                                                 gmx::ArrayRef<real> charges,
                                                 gmx::ArrayRef<int64_t> particleInteractionFlags,
                                                 gmx::ArrayRef<int>     exclusionRanges,
                                                 gmx::ArrayRef<int>     exclusionElements,
                                                 const NBKernelOptions& options)
{
    impl_ = std::make_unique<CpuImpl>(particleTypeIdOfAllParticles,
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

//! Compute forces and virial tensor
void GmxNBForceCalculatorCpu::compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                                      const Box&                     box,
                                      gmx::ArrayRef<gmx::RVec>       forceOutput,
                                      gmx::ArrayRef<real>            virialOutput)
{
    impl_->compute(coordinateInput, box, forceOutput, virialOutput);
}

//! Compute forces, virial tensor and potential energies
void GmxNBForceCalculatorCpu::compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                                      const Box&                     box,
                                      gmx::ArrayRef<gmx::RVec>       forceOutput,
                                      gmx::ArrayRef<real>            virialOutput,
                                      gmx::ArrayRef<real>            energyOutput)
{
    impl_->compute(coordinateInput, box, forceOutput, virialOutput, energyOutput);
}


GmxNBForceCalculatorGpu::GmxNBForceCalculatorGpu(gmx::ArrayRef<int>  particleTypeIdOfAllParticles,
                                                 gmx::ArrayRef<real> nonBondedParams,
                                                 gmx::ArrayRef<real> charges,
                                                 gmx::ArrayRef<int64_t>   particleInteractionFlags,
                                                 gmx::ArrayRef<int>       exclusionRanges,
                                                 gmx::ArrayRef<int>       exclusionElements,
                                                 const NBKernelOptions&   options,
                                                 const DeviceInformation& deviceInfo)
{
    impl_ = std::make_unique<GpuImpl>(particleTypeIdOfAllParticles,
                                      nonBondedParams,
                                      charges,
                                      particleInteractionFlags,
                                      exclusionRanges,
                                      exclusionElements,
                                      options,
                                      deviceInfo);
}

GmxNBForceCalculatorGpu::~GmxNBForceCalculatorGpu() = default;

//! calculates a new pair list based on new coordinates (for every NS step)
void GmxNBForceCalculatorGpu::updatePairlist(gmx::ArrayRef<const gmx::RVec> coordinates, const Box& box)
{
    impl_->updatePairlist(coordinates, box);
}

//! Compute forces and return
void GmxNBForceCalculatorGpu::compute(gmx::ArrayRef<const gmx::RVec> coordinateInput,
                                      const Box&                     box,
                                      gmx::ArrayRef<gmx::RVec>       forceOutput) const
{
    impl_->compute(coordinateInput, box, forceOutput);
}

} // namespace nblib
