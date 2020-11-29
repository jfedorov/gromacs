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
 * \brief Defines classes related to MTTK pressure coupling
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "mttk.h"

#include "gromacs/mdtypes/commrec.h"
#include "gromacs/domdec/domdec_network.h"
#include "gromacs/math/utilities.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/coupling.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/enerdata.h"

#include "energydata.h"
#include "velocityscalingtemperaturecoupling.h"
#include "nosehooverchains.h"
#include "simulatoralgorithm.h"

namespace gmx
{
/*! \brief Expected time precision
 *
 * Times are typically incremented in the order of 1e-3 ps (1 fs), so
 * 1e-6 should be sufficiently tight.
 */
static constexpr real c_timePrecision = 1e-6;

/*! \brief Check whether two times are nearly equal
 *
 * Times are considered close if their absolute difference is smaller
 * than c_timePrecision.
 *
 * \param time           The test time
 * \param referenceTime  The reference time
 * \return bool          Whether the absolute difference is < 1e-6
 */
static inline bool timesClose(Time time, Time referenceTime)
{
    return (time - referenceTime) * (time - referenceTime) < c_timePrecision * c_timePrecision;
}

void MttkData::build(LegacySimulatorData*                    legacySimulatorData,
                     ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                     StatePropagatorData*                    statePropagatorData,
                     EnergyData*                             energyData)
{
    // Uses reference temperature of first T-group
    const real referenceTemperature = legacySimulatorData->inputrec->opts.ref_t[0];
    const real referencePressure    = trace(legacySimulatorData->inputrec->ref_p) / DIM;
    // Weights are set based on initial volume
    real initialVolume = det(statePropagatorData->constBox());

    // When using domain decomposition, statePropagatorData might not have the inital
    // box yet, so we get it from the legacy state_global instead.
    // TODO: Make sure we have a valid state in statePropagatorData at all times (#3421)
    if (DOMAINDECOMP(legacySimulatorData->cr))
    {
        if (MASTER(legacySimulatorData->cr))
        {
            initialVolume = det(legacySimulatorData->state_global->box);
        }
        dd_bcast(legacySimulatorData->cr->dd, int(sizeof(real)), &initialVolume);
    }
    builderHelper->storeSimulationData(
            MttkData::dataID(),
            std::make_unique<MttkData>(
                    referenceTemperature,
                    referencePressure,
                    legacySimulatorData->inputrec->nstpcouple * legacySimulatorData->inputrec->delta_t,
                    legacySimulatorData->inputrec->tau_p,
                    initialVolume,
                    legacySimulatorData->inputrec->compress,
                    energyData,
                    statePropagatorData));
}

std::string MttkData::dataID()
{
    return "MttkData";
}

MttkData::MttkData(real                       referenceTemperature,
                   real                       referencePressure,
                   real                       couplingTimeStep,
                   real                       couplingTime,
                   real                       initialVolume,
                   const tensor               compressibility,
                   EnergyData*                energyData,
                   const StatePropagatorData* statePropagatorData) :
    couplingTimeStep_(couplingTimeStep),
    etaVelocity_(0.0),
    invMass_((PRESFAC * trace(compressibility) * BOLTZ * referenceTemperature)
             / (DIM * initialVolume * gmx::square(couplingTime / M_2PI))),
    etaVelocityTime_(0.0),
    temperatureCouplingIntegral_(0.0),
    integralTime_(0.0),
    referencePressure_(referencePressure),
    boxVelocity_{ { 0 } },
    statePropagatorData_(statePropagatorData)
{
    energyData->addConservedEnergyContribution(
            [this](Step /*unused*/, Time time) { return temperatureCouplingIntegral(time); });
    energyData->setParrinelloRahmanBoxVelocities([this]() { return boxVelocity_; });
}

void MttkData::calculateIntegralIfNeeded()
{
    // Check whether coordinate time divided by the time step is close to integer
    const bool calculationNeeded = timesClose(
            lround(etaVelocityTime_ / couplingTimeStep_) * couplingTimeStep_, etaVelocityTime_);

    if (calculationNeeded)
    {
        const real volume = det(statePropagatorData_->constBox());
        // Calculate current value of barostat integral
        temperatureCouplingIntegral_ = kineticEnergy() + volume * referencePressure_ / PRESFAC;
        integralTime_                = etaVelocityTime_;
    }
}

real MttkData::kineticEnergy() const
{
    return 0.5 * etaVelocity_ * etaVelocity_ / invMass_;
}

void MttkData::scale(real scalingFactor)
{
    etaVelocity_ *= scalingFactor;
    calculateIntegralIfNeeded();
}

real MttkData::etaVelocity() const
{
    return etaVelocity_;
}

real MttkData::invEtaMass() const
{
    return invMass_;
}

void MttkData::setEtaVelocity(real etaVelocity, real etaVelocityTimeIncrement)
{
    etaVelocity_ = etaVelocity;
    etaVelocityTime_ += etaVelocityTimeIncrement;
    calculateIntegralIfNeeded();
}

double MttkData::temperatureCouplingIntegral(Time time) const
{
    /* When using nstpcouple >= nstcalcenergy, we accept that the coupling
     * integral might be ahead of the current energy calculation step. The
     * extended system degrees of freedom are either in sync or ahead of the
     * rest of the system.
     */
    if (!(time <= integralTime_ || timesClose(integralTime_, time)))
    {
        GMX_THROW(ModSimRuntimeError("MttkData conserved energy time mismatch."));
    }
    return temperatureCouplingIntegral_;
}

real MttkData::referencePressure() const
{
    return referencePressure_;
}

rvec* MttkData::boxVelocities()
{
    return boxVelocity_;
}

namespace
{
/*!
 * \brief Enum describing the contents MttkData writes to modular checkpoint
 *
 * When changing the checkpoint content, add a new element just above Count, and adjust the
 * checkpoint functionality.
 */
enum class CheckpointVersion
{
    Base, //!< First version of modular checkpointing
    Count //!< Number of entries. Add new versions right above this!
};
constexpr auto c_currentVersion = CheckpointVersion(int(CheckpointVersion::Count) - 1);
} // namespace

template<CheckpointDataOperation operation>
void MttkData::doCheckpointData(CheckpointData<operation>* checkpointData)
{
    checkpointVersion(checkpointData, "MttkData version", c_currentVersion);
    checkpointData->scalar("veta", &etaVelocity_);
    // Mass is calculated from initial volume, so need to save it for exact continuation
    checkpointData->scalar("mass", &invMass_);
    checkpointData->scalar("time", &etaVelocityTime_);
}

void MttkData::saveCheckpointState(std::optional<WriteCheckpointData> checkpointData, const t_commrec* cr)
{
    if (MASTER(cr))
    {
        doCheckpointData<CheckpointDataOperation::Write>(&checkpointData.value());
    }
}

void MttkData::restoreCheckpointState(std::optional<ReadCheckpointData> checkpointData, const t_commrec* cr)
{
    if (MASTER(cr))
    {
        doCheckpointData<CheckpointDataOperation::Read>(&checkpointData.value());
    }
    if (DOMAINDECOMP(cr))
    {
        dd_bcast(cr->dd, int(sizeof(real)), &etaVelocity_);
        dd_bcast(cr->dd, int(sizeof(real)), &invMass_);
        dd_bcast(cr->dd, int(sizeof(Time)), &etaVelocityTime_);
    }
    // calculate integral?
}

const std::string& MttkData::clientID()
{
    return identifier_;
}

void MttkPropagatorConnection::build(ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                     const PropagatorTag& propagatorTagPrePosition,
                                     const PropagatorTag& propagatorTagPostPosition,
                                     int                  positionOffset,
                                     const PropagatorTag& propagatorTagPreVelocity1,
                                     const PropagatorTag& propagatorTagPostVelocity1,
                                     int                  velocityOffset1,
                                     const PropagatorTag& propagatorTagPreVelocity2,
                                     const PropagatorTag& propagatorTagPostVelocity2,
                                     int                  velocityOffset2)
{
    if (propagatorTagPrePosition == propagatorTagPostPosition
        && propagatorTagPrePosition != PropagatorTag(""))
    {
        GMX_THROW(NotImplementedError(
                "Pre- and post-step position scaling in same element is not supported."));
    }
    if ((propagatorTagPreVelocity1 == propagatorTagPostVelocity1
         && propagatorTagPreVelocity1 != PropagatorTag(""))
        || (propagatorTagPreVelocity2 == propagatorTagPostVelocity2
            && propagatorTagPreVelocity2 != PropagatorTag("")))
    {
        GMX_THROW(NotImplementedError(
                "Pre- and post-step velocity scaling in same element is not implemented."));
    }

    // Store object with simulation algorithm for safe pointer capturing
    builderHelper->storeSimulationData(MttkPropagatorConnection::dataID(),
                                       std::make_unique<MttkPropagatorConnection>());
    auto* object = builderHelper
                           ->simulationData<MttkPropagatorConnection>(MttkPropagatorConnection::dataID())
                           .value();

    builderHelper->registerTemperaturePressureControl(
            [object, propagatorTagPrePosition, positionOffset](const PropagatorConnection& connection) {
                object->connectWithPropagatorPositionPreStepScaling(
                        connection, propagatorTagPrePosition, positionOffset);
            });
    builderHelper->registerTemperaturePressureControl(
            [object, propagatorTagPostPosition, positionOffset](const PropagatorConnection& connection) {
                object->connectWithPropagatorPositionPostStepScaling(
                        connection, propagatorTagPostPosition, positionOffset);
            });
    builderHelper->registerTemperaturePressureControl(
            [object, propagatorTagPreVelocity1, velocityOffset1](const PropagatorConnection& connection) {
                object->connectWithPropagatorVelocityPreStepScaling(
                        connection, propagatorTagPreVelocity1, velocityOffset1);
            });
    builderHelper->registerTemperaturePressureControl(
            [object, propagatorTagPostVelocity1, velocityOffset1](const PropagatorConnection& connection) {
                object->connectWithPropagatorVelocityPostStepScaling(
                        connection, propagatorTagPostVelocity1, velocityOffset1);
            });
    builderHelper->registerTemperaturePressureControl(
            [object, propagatorTagPreVelocity2, velocityOffset2](const PropagatorConnection& connection) {
                object->connectWithPropagatorVelocityPreStepScaling(
                        connection, propagatorTagPreVelocity2, velocityOffset2);
            });
    builderHelper->registerTemperaturePressureControl(
            [object, propagatorTagPostVelocity2, velocityOffset2](const PropagatorConnection& connection) {
                object->connectWithPropagatorVelocityPostStepScaling(
                        connection, propagatorTagPostVelocity2, velocityOffset2);
            });
}

void MttkPropagatorConnection::propagatorCallback(Step step) const
{
    for (const auto& callback : propagatorCallbacks_)
    {
        std::get<0>(callback)(step + std::get<1>(callback));
    }
}

void MttkPropagatorConnection::setPositionScaling(real preStepScaling, real postStepScaling)
{
    for (const auto& scalingFactor : startPositionScalingFactors_)
    {
        std::fill(scalingFactor.begin(), scalingFactor.end(), preStepScaling);
    }
    for (const auto& scalingFactor : endPositionScalingFactors_)
    {
        std::fill(scalingFactor.begin(), scalingFactor.end(), postStepScaling);
    }
}

void MttkPropagatorConnection::setVelocityScaling(real preStepScaling, real postStepScaling)
{
    for (const auto& scalingFactor : startVelocityScalingFactors_)
    {
        std::fill(scalingFactor.begin(), scalingFactor.end(), preStepScaling);
    }
    for (const auto& scalingFactor : endVelocityScalingFactors_)
    {
        std::fill(scalingFactor.begin(), scalingFactor.end(), postStepScaling);
    }
}

std::string MttkPropagatorConnection::dataID()
{
    return "MttkPropagatorConnection";
}

void MttkPropagatorConnection::connectWithPropagatorVelocityPreStepScaling(const PropagatorConnection& connectionData,
                                                                           const PropagatorTag& propagatorTag,
                                                                           int offset)
{
    if (connectionData.tag == propagatorTag && connectionData.startVelocityScaling)
    {
        connectionData.setNumVelocityScalingVariables(1, ScaleVelocities::PreStepOnly);
        startVelocityScalingFactors_.emplace_back(connectionData.getViewOnStartVelocityScaling());
        propagatorCallbacks_.emplace_back(
                std::make_tuple(connectionData.getVelocityScalingCallback(), offset));
    }
}

void MttkPropagatorConnection::connectWithPropagatorVelocityPostStepScaling(const PropagatorConnection& connectionData,
                                                                            const PropagatorTag& propagatorTag,
                                                                            int offset)
{
    if (connectionData.tag == propagatorTag && connectionData.startVelocityScaling)
    {
        // Although we're using this propagator for scaling after the update, we're using
        // getViewOnStartVelocityScaling() - getViewOnEndVelocityScaling() is only
        // used for propagators doing BOTH start and end scaling
        connectionData.setNumVelocityScalingVariables(1, ScaleVelocities::PreStepOnly);
        endVelocityScalingFactors_.emplace_back(connectionData.getViewOnStartVelocityScaling());
        propagatorCallbacks_.emplace_back(
                std::make_tuple(connectionData.getVelocityScalingCallback(), offset));
    }
}

void MttkPropagatorConnection::connectWithPropagatorPositionPreStepScaling(const PropagatorConnection& connectionData,
                                                                           const PropagatorTag& propagatorTag,
                                                                           int offset)
{
    if (connectionData.tag == propagatorTag && connectionData.positionScaling)
    {
        connectionData.setNumPositionScalingVariables(1);
        startPositionScalingFactors_.emplace_back(connectionData.getViewOnPositionScaling());
        propagatorCallbacks_.emplace_back(
                std::make_tuple(connectionData.getPositionScalingCallback(), offset));
    }
}

void MttkPropagatorConnection::connectWithPropagatorPositionPostStepScaling(const PropagatorConnection& connectionData,
                                                                            const PropagatorTag& propagatorTag,
                                                                            int offset)
{
    if (connectionData.tag == propagatorTag && connectionData.positionScaling)
    {
        connectionData.setNumPositionScalingVariables(1);
        endPositionScalingFactors_.emplace_back(connectionData.getViewOnPositionScaling());
        propagatorCallbacks_.emplace_back(
                std::make_tuple(connectionData.getPositionScalingCallback(), offset));
    }
}

void MttkElement::propagateEtaVelocity(Step step)
{
    const auto* ekind         = energyData_->ekindata();
    const auto* virial        = energyData_->totalVirial(step);
    const real  currentVolume = det(statePropagatorData_->constBox());
    // Tuckerman et al. 2006, Eq 5.8
    // Note that we're using the dof of the first temperature group only
    const real alpha = 1.0 + DIM / (numDegreesOfFreedom_);
    // Also here, using first group only
    const real kineticEnergyFactor = alpha * ekind->tcstat[0].ekinscalef_nhc;
    // Now, we're using full system kinetic energy!
    tensor modifiedKineticEnergy;
    msmul(ekind->ekin, kineticEnergyFactor, modifiedKineticEnergy);

    tensor currentPressureTensor;

    const real currentPressure =
            calc_pres(pbcType_, numWalls_, statePropagatorData_->constBox(), modifiedKineticEnergy, virial, currentPressureTensor)
            + energyData_->enerdata()->term[F_PDISPCORR];

    const real etaAcceleration = DIM * currentVolume * (mttkData_->invEtaMass() / PRESFAC)
                                 * (currentPressure - mttkData_->referencePressure());
    mttkData_->setEtaVelocity(mttkData_->etaVelocity() + propagationTimeStep_ * etaAcceleration,
                              propagationTimeStep_);

    /* Tuckerman et al. 2006, eqs 5.11 and 5.13:
     *
     * r(t+dt)   = r(t)*exp(v_eta*dt) + dt*v*exp(v_eta*dt/2) * [sinh(v_eta*dt/2) / (v_eta*dt/2)]
     * v(t+dt/2) = v(t)*exp(-a*v_eta*dt/2) +
     *             dt/2*f/m*exp(-a*v_eta*dt/4) * [sinh(a*v_eta*dt/4) / (a*v_eta*dt/4)]
     * with a = 1 + 1/Natoms
     *
     * For r, let
     *   s1 = exp(v_eta*dt/2)
     *   s2 = [sinh(v_eta*dt/2) / (v_eta*dt/2)]
     * so we can use
     *   r(t) *= s1/s2
     *   r(t+dt) = r(t) + dt*v
     *   r(t+dt) *= s1*s2  <=>  r(t+dt) = s1*s2 * (r(t)*s1/s2 + dt*v) = s1^2*r(t) + dt*v*s1*s2
     *
     * For v, let
     *   s1 = exp(-a*v_eta*dt/4)
     *   s2 = [sinh(a*v_eta*dt/4) / (a*v_eta*dt/4)]
     * so we can use
     *   v(t) *= s1/s2
     *   v(t+dt/2) = v(t) + dt/2*f/m
     *   v(t+dt/2) *= s1*s2  <=>  v(t+dt/2) = s1^2*v(t) + dt/2*f/m*s1*s2
     *
     * In legacy simulator, this scaling is applied every step, even if the barostat is updated
     * less frequently, so we are mirroring this by using the simulation time step for dt and
     * requesting scaling every step. This could likely be applied impulse-style by using the
     * coupling time step for dt and only applying it when the barostat gets updated.
     */
    const real scalingPos1 = std::exp(0.5 * simulationTimeStep_ * mttkData_->etaVelocity());
    const real scalingPos2 = gmx::series_sinhx(0.5 * simulationTimeStep_ * mttkData_->etaVelocity());
    const real scalingVel1 = std::exp(-alpha * 0.25 * simulationTimeStep_ * mttkData_->etaVelocity());
    const real scalingVel2 =
            gmx::series_sinhx(alpha * 0.25 * simulationTimeStep_ * mttkData_->etaVelocity());

    mttkPropagatorConnection_->setPositionScaling(scalingPos1 / scalingPos2, scalingPos1 * scalingPos2);
    mttkPropagatorConnection_->setVelocityScaling(scalingVel1 / scalingVel2, scalingVel1 * scalingVel2);
}

MttkElement::MttkElement(int                        nstcouple,
                         int                        offset,
                         real                       propagationTimeStep,
                         real                       simulationTimeStep,
                         ScheduleOnInitStep         scheduleOnInitStep,
                         Step                       initStep,
                         const StatePropagatorData* statePropagatorData,
                         EnergyData*                energyData,
                         MttkData*                  mttkData,
                         MttkPropagatorConnection*  mttkPropagatorConnection,
                         PbcType                    pbcType,
                         int                        numWalls,
                         real                       numDegreesOfFreedom) :
    pbcType_(pbcType),
    numWalls_(numWalls),
    numDegreesOfFreedom_(numDegreesOfFreedom),
    nstcouple_(nstcouple),
    offset_(offset),
    propagationTimeStep_(propagationTimeStep),
    simulationTimeStep_(simulationTimeStep),
    scheduleOnInitStep_(scheduleOnInitStep),
    initialStep_(initStep),
    statePropagatorData_(statePropagatorData),
    energyData_(energyData),
    mttkData_(mttkData),
    mttkPropagatorConnection_(mttkPropagatorConnection)
{
}

void MttkElement::scheduleTask(Step step, Time /*unused*/, const RegisterRunFunction& registerRunFunction)
{
    if (step == initialStep_ && scheduleOnInitStep_ == ScheduleOnInitStep::No)
    {
        return;
    }
    if (do_per_step(step + nstcouple_ + offset_, nstcouple_))
    {
        // do T-coupling this step
        registerRunFunction([this, step]() { propagateEtaVelocity(step); });
    }

    // Let propagators know that we want to scale (every step - see comment in propagateEtaVelocity)
    mttkPropagatorConnection_->propagatorCallback(step);
}

ISimulatorElement*
MttkElement::getElementPointerImpl(LegacySimulatorData*                    legacySimulatorData,
                                   ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                   StatePropagatorData* gmx_unused statePropagatorData,
                                   EnergyData*                     energyData,
                                   FreeEnergyPerturbationData* gmx_unused freeEnergyPerturbationData,
                                   GlobalCommunicationHelper* gmx_unused globalCommunicationHelper,
                                   Offset                                offset,
                                   ScheduleOnInitStep                    scheduleOnInitStep,
                                   const PropagatorTag&                  propagatorTagPrePosition,
                                   const PropagatorTag&                  propagatorTagPostPosition,
                                   Offset                                positionOffset,
                                   const PropagatorTag&                  propagatorTagPreVelocity1,
                                   const PropagatorTag&                  propagatorTagPostVelocity1,
                                   Offset                                velocityOffset1,
                                   const PropagatorTag&                  propagatorTagPreVelocity2,
                                   const PropagatorTag&                  propagatorTagPostVelocity2,
                                   Offset                                velocityOffset2)
{
    // Data is now owned by the caller of this method, who will handle lifetime (see ModularSimulatorAlgorithm)
    if (!builderHelper->simulationData<MttkData>(MttkData::dataID()))
    {
        MttkData::build(legacySimulatorData, builderHelper, statePropagatorData, energyData);
    }
    auto* mttkData = builderHelper->simulationData<MttkData>(MttkData::dataID()).value();
    if (!builderHelper->simulationData<MttkPropagatorConnection>(MttkPropagatorConnection::dataID()))
    {
        MttkPropagatorConnection::build(builderHelper,
                                        propagatorTagPrePosition,
                                        propagatorTagPostPosition,
                                        positionOffset,
                                        propagatorTagPreVelocity1,
                                        propagatorTagPostVelocity1,
                                        velocityOffset1,
                                        propagatorTagPreVelocity2,
                                        propagatorTagPostVelocity2,
                                        velocityOffset2);
    }
    auto* mttkPropagatorConnection =
            builderHelper
                    ->simulationData<MttkPropagatorConnection>(MttkPropagatorConnection::dataID())
                    .value();

    // Element is now owned by the caller of this method, who will handle lifetime (see ModularSimulatorAlgorithm)
    auto* element = static_cast<MttkElement*>(builderHelper->storeElement(std::make_unique<MttkElement>(
            legacySimulatorData->inputrec->nsttcouple,
            offset,
            legacySimulatorData->inputrec->delta_t * legacySimulatorData->inputrec->nstpcouple / 2,
            legacySimulatorData->inputrec->delta_t,
            scheduleOnInitStep,
            legacySimulatorData->inputrec->init_step,
            statePropagatorData,
            energyData,
            mttkData,
            mttkPropagatorConnection,
            legacySimulatorData->inputrec->pbcType,
            legacySimulatorData->inputrec->nwall,
            legacySimulatorData->inputrec->opts.nrdf[0])));

    return element;
}

MttkBoxScaling::MttkBoxScaling(real                 simulationTimeStep,
                               StatePropagatorData* statePropagatorData,
                               MttkData*            mttkData) :
    simulationTimeStep_(simulationTimeStep),
    statePropagatorData_(statePropagatorData),
    mttkData_(mttkData)
{
}

void MttkBoxScaling::scheduleTask(Step gmx_unused step,
                                  gmx_unused Time            time,
                                  const RegisterRunFunction& registerRunFunction)
{
    registerRunFunction([this]() { scaleBox(); });
}

void MttkBoxScaling::scaleBox()
{
    auto* box = statePropagatorData_->box();

    /* DIM * eta = ln V.  so DIM*eta_new = DIM*eta_old + DIM*dt*veta =>
       ln V_new = ln V_old + 3*dt*veta => V_new = V_old*exp(3*dt*veta) =>
       Side length scales as exp(veta*dt) */
    msmul(box, std::exp(mttkData_->etaVelocity() * simulationTimeStep_), box);

    /* Relate veta to boxv.  veta = d(eta)/dT = (1/DIM)*1/V dV/dT.
       o               If we assume isotropic scaling, and box length scaling
       factor L, then V = L^DIM (det(M)).  So dV/dt = DIM
       L^(DIM-1) dL/dt det(M), and veta = (1/L) dL/dt.  The
       determinant of B is L^DIM det(M), and the determinant
       of dB/dt is (dL/dT)^DIM det (M).  veta will be
       (det(dB/dT)/det(B))^(1/3).  Then since M =
       B_new*(vol_new)^(1/3), dB/dT_new = (veta_new)*B(new). */
    msmul(box, mttkData_->etaVelocity(), mttkData_->boxVelocities());
}

ISimulatorElement*
MttkBoxScaling::getElementPointerImpl(LegacySimulatorData*                    legacySimulatorData,
                                      ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                      StatePropagatorData*                    statePropagatorData,
                                      EnergyData*                             energyData,
                                      FreeEnergyPerturbationData* gmx_unused freeEnergyPerturbationData,
                                      GlobalCommunicationHelper* gmx_unused globalCommunicationHelper)
{
    // Data is now owned by the caller of this method, who will handle lifetime (see ModularSimulatorAlgorithm)
    if (!builderHelper->simulationData<MttkData>(MttkData::dataID()))
    {
        MttkData::build(legacySimulatorData, builderHelper, statePropagatorData, energyData);
    }

    return builderHelper->storeElement(std::make_unique<MttkBoxScaling>(
            legacySimulatorData->inputrec->delta_t,
            statePropagatorData,
            builderHelper->simulationData<MttkData>(MttkData::dataID()).value()));
}

} // namespace gmx
