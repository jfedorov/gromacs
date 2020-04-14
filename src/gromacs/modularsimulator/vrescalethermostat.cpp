/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020, by the GROMACS development team, led by
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
 * \brief Defines the v-rescale thermostat for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "vrescalethermostat.h"

#include "gromacs/domdec/domdec_network.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdtypes/checkpointdata.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/utility/fatalerror.h"

#include "builders.h"
#include "checkpointhelper.h"
#include "energyelement.h"

namespace gmx
{

VRescaleThermostat::VRescaleThermostat(int         nstcouple,
                                       int64_t     seed,
                                       int         numTemperatureGroups,
                                       double      couplingTimeStep,
                                       const real* referenceTemperature,
                                       const real* couplingTime,
                                       const real* numDegreesOfFreedom,
                                       int         inputThermostatType) :
    nstcouple_(nstcouple),
    offset_(),
    useFullStepKE_(),
    seed_(seed),
    numTemperatureGroups_(numTemperatureGroups),
    couplingTimeStep_(couplingTimeStep),
    referenceTemperature_(referenceTemperature, referenceTemperature + numTemperatureGroups),
    couplingTime_(couplingTime, couplingTime + numTemperatureGroups),
    numDegreesOfFreedom_(numDegreesOfFreedom, numDegreesOfFreedom + numTemperatureGroups),
    thermostatIntegral_(numTemperatureGroups, 0.0),
    energyElement_(nullptr),
    propagatorCallback_(nullptr)
{
    if (inputThermostatType != etcVRESCALE)
    {
        GMX_THROW(ElementNotNeededException(
                "VRescaleThermostat is not needed without v-rescale temperature control."));
    }
}

void VRescaleThermostat::scheduleTask(Step step, Time gmx_unused time, const RegisterRunFunctionPtr& registerRunFunction)
{
    /* The thermostat will need a valid kinetic energy when it is running.
     * Currently, computeGlobalCommunicationPeriod() is making sure this
     * happens on time.
     * TODO: Once we're switching to a new global communication scheme, we
     *       will want the thermostat to signal that global reduction
     *       of the kinetic energy is needed.
     *
     */
    if (do_per_step(step + nstcouple_ + offset_, nstcouple_))
    {
        // do T-coupling this step
        (*registerRunFunction)(
                std::make_unique<SimulatorRunFunction>([this, step]() { setLambda(step); }));

        // Let propagator know that we want to do T-coupling
        (*propagatorCallback_)(step);
    }
}

void VRescaleThermostat::setLambda(Step step)
{
    real currentKineticEnergy, referenceKineticEnergy, newKineticEnergy;

    auto ekind = energyElement_->ekindata();

    for (int i = 0; (i < numTemperatureGroups_); i++)
    {
        if (useFullStepKE_)
        {
            currentKineticEnergy = trace(ekind->tcstat[i].ekinf);
        }
        else
        {
            currentKineticEnergy = trace(ekind->tcstat[i].ekinh);
        }

        if (couplingTime_[i] >= 0 && numDegreesOfFreedom_[i] > 0 && currentKineticEnergy > 0)
        {
            referenceKineticEnergy = 0.5 * referenceTemperature_[i] * BOLTZ * numDegreesOfFreedom_[i];

            newKineticEnergy = vrescale_resamplekin(currentKineticEnergy, referenceKineticEnergy,
                                                    numDegreesOfFreedom_[i],
                                                    couplingTime_[i] / couplingTimeStep_, step, seed_);

            // Analytically newKineticEnergy >= 0, but we check for rounding errors
            if (newKineticEnergy <= 0)
            {
                lambda_[i] = 0.0;
            }
            else
            {
                lambda_[i] = std::sqrt(newKineticEnergy / currentKineticEnergy);
            }

            thermostatIntegral_[i] -= newKineticEnergy - currentKineticEnergy;

            if (debug)
            {
                fprintf(debug, "TC: group %d: Ekr %g, Ek %g, Ek_new %g, Lambda: %g\n", i,
                        referenceKineticEnergy, currentKineticEnergy, newKineticEnergy, lambda_[i]);
            }
        }
        else
        {
            lambda_[i] = 1.0;
        }
    }
}

template<CheckpointDataOperation operation>
void VRescaleThermostat::doCheckpointData(CheckpointData* checkpointData, const t_commrec* cr)
{
    if (MASTER(cr))
    {
        checkpointData->arrayRef<operation>("thermostat integral",
                                            makeCheckpointArrayRef<operation>(thermostatIntegral_));
    }
    if (operation == CheckpointDataOperation::Read && DOMAINDECOMP(cr))
    {
        dd_bcast(cr->dd, thermostatIntegral_.size() * sizeof(double), thermostatIntegral_.data());
    }
}

void VRescaleThermostat::writeCheckpoint(CheckpointData checkpointData, const t_commrec* cr)
{
    doCheckpointData<CheckpointDataOperation::Write>(&checkpointData, cr);
}

void VRescaleThermostat::readCheckpoint(CheckpointData checkpointData, const t_commrec* cr)
{
    doCheckpointData<CheckpointDataOperation::Read>(&checkpointData, cr);
}

const std::vector<double>& VRescaleThermostat::thermostatIntegral() const
{
    return thermostatIntegral_;
}

void VRescaleThermostatBuilder::connectWithBuilders(ElementAndSignallerBuilders* builders)
{
    setEnergyElementBuilder(builders->energyElement.get());
    registerWithCheckpointHelper(builders->checkpointHelper.get());
}

void VRescaleThermostatBuilder::setEnergyElementBuilder(EnergyElementBuilder* energyElementBuilder)
{
    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Tried to set EnergyElement after VRescaleThermostat was built.");
    if (vrThermostat_)
    {
        vrThermostat_->energyElement_ = energyElementBuilder->getPointer();
        energyElementBuilder->setVRescaleThermostat(vrThermostat_.get());
    }
}

void VRescaleThermostatBuilder::registerWithCheckpointHelper(CheckpointHelperBuilder* checkpointHelperBuilder)
{
    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Tried to register to CheckpointHelper after VRescaleThermostat was built.");
    if (vrThermostat_)
    {
        checkpointHelperBuilder->registerClient(compat::make_not_null(vrThermostat_.get()),
                                                vrThermostat_->identifier);
    }
    registeredWithCheckpointHelper_ = true;
}

std::unique_ptr<VRescaleThermostat> VRescaleThermostatBuilder::build(int offset, bool useFullStepKE)
{
    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Called build() without available VRescaleThermostat.");
    if (vrThermostat_)
    {
        GMX_RELEASE_ASSERT(vrThermostat_->energyElement_,
                           "Tried to build VRescaleThermostat before setting EnergyElement.");
        GMX_RELEASE_ASSERT(registeredWithCheckpointHelper_,
                           "Tried to build VRescaleThermostat before registering with "
                           "CheckpointHelper.");
        GMX_RELEASE_ASSERT(
                registeredWithPropagator_,
                "Tried to build VRescaleThermostat before registering with a propagator.");
        vrThermostat_->offset_        = offset;
        vrThermostat_->useFullStepKE_ = useFullStepKE;
    }
    registrationPossible_ = false;
    return std::move(vrThermostat_);
}

VRescaleThermostatBuilder::~VRescaleThermostatBuilder()
{
    // If the element was built, but not consumed, we risk dangling pointers
    GMX_RELEASE_ASSERT(!vrThermostat_, "VRescaleThermostat was constructed, but not used.");
}

} // namespace gmx
