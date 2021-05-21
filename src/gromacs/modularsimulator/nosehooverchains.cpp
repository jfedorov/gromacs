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
 * \brief Defines classes related to Nose-Hoover chains for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "nosehooverchains.h"

#include <numeric>

#include "gromacs/domdec/domdec_network.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/strconvert.h"

#include "energydata.h"
#include "mttk.h"
#include "simulatoralgorithm.h"
#include "trotterhelperfunctions.h"
#include "velocityscalingtemperaturecoupling.h"

namespace gmx
{
// Names of the NHC usage options
static constexpr EnumerationArray<NhcUsage, const char*> nhcUsageNames = { "System", "Barostat" };

//! View on the NHC degrees of freedom of a temperature group
struct NhcCoordinateView
{
    //! The positions
    ArrayRef<real> positions;
    //! The velocities
    ArrayRef<real> velocities;
    //! The inverse masses
    ArrayRef<const real> invMass;
    //! The temperature group
    const int temperatureGroup;

    //! No copy construction
    NhcCoordinateView(const NhcCoordinateView&) = delete;
    //! No copy assignment
    NhcCoordinateView& operator=(const NhcCoordinateView&) = delete;
    //! Move constructor is allowed
    NhcCoordinateView(NhcCoordinateView&&) noexcept = default;
    //! Move assignment is implicitly deleted (const int)
    NhcCoordinateView& operator=(NhcCoordinateView&&) noexcept = delete;
    //! Complete rule of 5
    ~NhcCoordinateView() noexcept = default;
};

NoseHooverChainsData::NoseHooverChainsData(int                  numTemperatureGroups,
                                           real                 couplingTimeStep,
                                           int                  chainLength,
                                           ArrayRef<const real> referenceTemperature,
                                           ArrayRef<const real> couplingTime,
                                           ArrayRef<const real> numDegreesOfFreedom,
                                           EnergyData*          energyData,
                                           NhcUsage             nhcUsage) :
    identifier_(formatString("NoseHooverChainsData-%s", nhcUsageNames[nhcUsage])),
    couplingTimeStep_(couplingTimeStep),
    chainLength_(chainLength),
    numTemperatureGroups_(numTemperatureGroups),
    referenceTemperature_(referenceTemperature),
    couplingTime_(couplingTime),
    numDegreesOfFreedom_(numDegreesOfFreedom)
{
    xi_.resize(numTemperatureGroups);
    xiVelocities_.resize(numTemperatureGroups);
    temperatureCouplingIntegral_.resize(numTemperatureGroups, 0.0);
    invXiMass_.resize(numTemperatureGroups);
    coordinateViewInUse_.resize(numTemperatureGroups, false);
    coordinateTime_.resize(numTemperatureGroups, 0.0);
    integralTime_.resize(numTemperatureGroups, 0.0);

    for (auto temperatureGroup = 0; temperatureGroup < numTemperatureGroups; ++temperatureGroup)
    {
        xi_[temperatureGroup].resize(chainLength, 0.0);
        xiVelocities_[temperatureGroup].resize(chainLength, 0.0);
        invXiMass_[temperatureGroup].resize(chainLength, 0.0);

        if (referenceTemperature_[temperatureGroup] > 0 && couplingTime_[temperatureGroup] > 0
            && this->numDegreesOfFreedom(temperatureGroup) > 0)
        {
            for (auto chainPosition = 0; chainPosition < chainLength; ++chainPosition)
            {
                const real numDof =
                        ((chainPosition == 0) ? this->numDegreesOfFreedom(temperatureGroup) : 1);
                invXiMass_[temperatureGroup][chainPosition] =
                        1.0
                        / (gmx::square(couplingTime_[temperatureGroup] / M_2PI)
                           * referenceTemperature_[temperatureGroup] * numDof * c_boltz);
                if (nhcUsage == NhcUsage::Barostat && chainPosition == 0)
                {
                    invXiMass_[temperatureGroup][chainPosition] /= DIM * DIM;
                }
            }
        }
    }

    energyData->addConservedEnergyContribution(
            [this](Step /*unused*/, Time time) { return temperatureCouplingIntegral(time); });
}

void NoseHooverChainsData::build(NhcUsage                                nhcUsage,
                                 LegacySimulatorData*                    legacySimulatorData,
                                 ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                 EnergyData*                             energyData)
{
    // Data is now owned by the caller of this method, who will handle lifetime (see ModularSimulatorAlgorithm)
    if (nhcUsage == NhcUsage::System)
    {
        builderHelper->storeSimulationData(
                NoseHooverChainsData::dataID(nhcUsage),
                std::make_unique<NoseHooverChainsData>(
                        legacySimulatorData->inputrec->opts.ngtc,
                        legacySimulatorData->inputrec->delta_t * legacySimulatorData->inputrec->nsttcouple,
                        legacySimulatorData->inputrec->opts.nhchainlength,
                        constArrayRefFromArray(legacySimulatorData->inputrec->opts.ref_t,
                                               legacySimulatorData->inputrec->opts.ngtc),
                        constArrayRefFromArray(legacySimulatorData->inputrec->opts.tau_t,
                                               legacySimulatorData->inputrec->opts.ngtc),
                        constArrayRefFromArray(legacySimulatorData->inputrec->opts.nrdf,
                                               legacySimulatorData->inputrec->opts.ngtc),
                        energyData,
                        nhcUsage));
    }
    else
    {
        const int numTemperatureGroups = 1;
        builderHelper->storeSimulationData(
                NoseHooverChainsData::dataID(nhcUsage),
                std::make_unique<NoseHooverChainsData>(
                        numTemperatureGroups,
                        legacySimulatorData->inputrec->delta_t * legacySimulatorData->inputrec->nstpcouple,
                        legacySimulatorData->inputrec->opts.nhchainlength,
                        constArrayRefFromArray(legacySimulatorData->inputrec->opts.ref_t, 1),
                        constArrayRefFromArray(legacySimulatorData->inputrec->opts.tau_t, 1),
                        ArrayRef<real>(),
                        energyData,
                        nhcUsage));
    }
}

NhcCoordinateView NoseHooverChainsData::coordinateView(int temperatureGroup)
{
    GMX_ASSERT(!coordinateViewInUse_[temperatureGroup],
               "xi view was already requested and not returned.");
    coordinateViewInUse_[temperatureGroup] = true;
    return { xi_[temperatureGroup], xiVelocities_[temperatureGroup], invXiMass_[temperatureGroup], temperatureGroup };
}

void NoseHooverChainsData::returnCoordinateView(NhcCoordinateView nhcCoordinateView, real timeIncrement)
{
    const int temperatureGroup             = nhcCoordinateView.temperatureGroup;
    coordinateViewInUse_[temperatureGroup] = false;
    coordinateTime_[temperatureGroup] += timeIncrement;
    if (isAtFullCouplingTimeStep(temperatureGroup))
    {
        calculateIntegral(temperatureGroup);
    }
}

inline int NoseHooverChainsData::numTemperatureGroups() const
{
    return numTemperatureGroups_;
}

inline real NoseHooverChainsData::referenceTemperature(int temperatureGroup) const
{
    GMX_ASSERT(temperatureGroup < numTemperatureGroups_, "Invalid temperature group");
    return referenceTemperature_[temperatureGroup];
}

inline real NoseHooverChainsData::numDegreesOfFreedom(int temperatureGroup) const
{
    GMX_ASSERT(temperatureGroup < numTemperatureGroups_, "Invalid temperature group");
    if (numDegreesOfFreedom_.empty() && temperatureGroup == 0)
    {
        // Barostat has a single degree of freedom
        return 1;
    }
    return numDegreesOfFreedom_[temperatureGroup];
}

inline bool NoseHooverChainsData::isAtFullCouplingTimeStep() const
{
    for (int temperatureGroup = 0; temperatureGroup < numTemperatureGroups_; ++temperatureGroup)
    {
        if (!isAtFullCouplingTimeStep(temperatureGroup))
        {
            return false;
        }
    }
    return true;
}

void NoseHooverChainsData::calculateIntegral(int temperatureGroup)
{
    // Calculate current value of thermostat integral
    double integral = 0.0;
    for (auto chainPosition = 0; chainPosition < chainLength_; ++chainPosition)
    {
        // Chain thermostats have only one degree of freedom
        const real numDegreesOfFreedomThisPosition =
                (chainPosition == 0) ? numDegreesOfFreedom(temperatureGroup) : 1;
        integral += 0.5 * gmx::square(xiVelocities_[temperatureGroup][chainPosition])
                            / invXiMass_[temperatureGroup][chainPosition]
                    + numDegreesOfFreedomThisPosition * xi_[temperatureGroup][chainPosition]
                              * c_boltz * referenceTemperature_[temperatureGroup];
    }
    temperatureCouplingIntegral_[temperatureGroup] = integral;
    integralTime_[temperatureGroup]                = coordinateTime_[temperatureGroup];
}

double NoseHooverChainsData::temperatureCouplingIntegral(Time gmx_used_in_debug time) const
{
    /* When using nsttcouple >= nstcalcenergy, we accept that the coupling
     * integral might be ahead of the current energy calculation step. The
     * extended system degrees of freedom are either in sync or ahead of the
     * rest of the system.
     */
    GMX_ASSERT(!std::any_of(integralTime_.begin(),
                            integralTime_.end(),
                            [time](real integralTime) {
                                return !(time <= integralTime || timesClose(integralTime, time));
                            }),
               "NoseHooverChainsData conserved energy time mismatch.");
    return std::accumulate(temperatureCouplingIntegral_.begin(), temperatureCouplingIntegral_.end(), 0.0);
}

bool NoseHooverChainsData::isAtFullCouplingTimeStep(int temperatureGroup) const
{
    // Check whether coordinate time divided by the time step is close to integer
    return timesClose(std::lround(coordinateTime_[temperatureGroup] / couplingTimeStep_) * couplingTimeStep_,
                      coordinateTime_[temperatureGroup]);
}

namespace
{
/*!
 * \brief Enum describing the contents NoseHooverChainsData writes to modular checkpoint
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
void NoseHooverChainsData::doCheckpointData(CheckpointData<operation>* checkpointData)
{
    checkpointVersion(checkpointData, "NoseHooverChainsData version", c_currentVersion);

    for (unsigned int temperatureGroup = 0; temperatureGroup < xi_.size(); ++temperatureGroup)
    {
        const auto temperatureGroupStr = toString(static_cast<int>(temperatureGroup));
        checkpointData->arrayRef("xi T-group " + temperatureGroupStr,
                                 makeCheckpointArrayRef<operation>(xi_[temperatureGroup]));
        checkpointData->arrayRef("xi velocities T-group " + temperatureGroupStr,
                                 makeCheckpointArrayRef<operation>(xiVelocities_[temperatureGroup]));
    }
    checkpointData->arrayRef("Coordinate times", makeCheckpointArrayRef<operation>(coordinateTime_));
}

void NoseHooverChainsData::saveCheckpointState(std::optional<WriteCheckpointData> checkpointData,
                                               const t_commrec*                   cr)
{
    if (MASTER(cr))
    {
        doCheckpointData<CheckpointDataOperation::Write>(&checkpointData.value());
    }
}

void NoseHooverChainsData::restoreCheckpointState(std::optional<ReadCheckpointData> checkpointData,
                                                  const t_commrec*                  cr)
{
    if (MASTER(cr))
    {
        doCheckpointData<CheckpointDataOperation::Read>(&checkpointData.value());
    }
    if (DOMAINDECOMP(cr))
    {
        for (unsigned int temperatureGroup = 0; temperatureGroup < xi_.size(); ++temperatureGroup)
        {
            dd_bcast(cr->dd, ssize(xi_[temperatureGroup]) * int(sizeof(real)), xi_[temperatureGroup].data());
            dd_bcast(cr->dd,
                     ssize(xiVelocities_[temperatureGroup]) * int(sizeof(real)),
                     xiVelocities_[temperatureGroup].data());
        }
        dd_bcast(cr->dd, ssize(coordinateTime_) * int(sizeof(real)), coordinateTime_.data());
    }
    for (unsigned int temperatureGroup = 0; temperatureGroup < xi_.size(); ++temperatureGroup)
    {
        calculateIntegral(temperatureGroup);
    }
}

const std::string& NoseHooverChainsData::clientID()
{
    return identifier_;
}

std::string NoseHooverChainsData::dataID(NhcUsage nhcUsage)
{
    return formatString("NoseHooverChainsData%s", nhcUsageNames[nhcUsage]);
}

/*! \brief Trotter operator for the NHC degrees of freedom
 *
 * This follows Tuckerman et al. 2006
 *
 * In NVT, the Trotter decomposition reads
 *   exp[iL dt] = exp[iLT dt/2] exp[iLv dt/2] exp[iLx dt] exp[iLv dt/2] exp[iLT dt/2]
 * iLv denotes the velocity propagation, iLx the position propagation
 * iLT denotes the thermostat propagation implemented here:
 *     v_xi[i](t-dt/2) = v_xi[i](t-dt) + dt_xi/2 * a_xi[i](t-dt);
 *     xi[i](t) = xi[i](t-dt) + dt_xi * v_xi[i](t-dt/2);
 *     v_sys *= exp(-dt/2 * v_xi[1](t-dt/2))
 *     v_xi[i](t) = v_xi[i](t-dt/2) + dt_xi/2 * a_xi[i](t);
 * where i = 1 ... N_chain, and
 *     a[i](t) = (M_xi * v_xi[i+1](t)^2 - 2*K_ref) / M_xi , i = 2 ... N_chain
 *     a[1](t) = (K_sys - K_ref) / M_xi
 * Note, iLT contains a term scaling the system velocities!
 *
 * In the legacy GROMACS simulator, the top of the loop marks the simulation
 * state at x(t), v(t-dt/2), f(t-1), mirroring the leap-frog implementation.
 * The loop then proceeds to calculate the forces at time t, followed by a
 * velocity half step (corresponding to the second exp[iLv dt/2] above).
 * For Tuckerman NHC NVT, this is followed by a thermostat propagation to reach
 * the full timestep t state. This is the state which is printed to file, so
 * we need to scale the velocities.
 * After writing to file, the next step effectively starts, by moving the thermostat
 * variables (half step), the velocities (half step) and the positions (full step),
 * which is equivalent to the first three terms of the Trotter decomposition above.
 * Currently, modular simulator is replicating the division of the simulator loop
 * used by the legacy simulator. The implementation here is independent of these
 * assumptions, but the builder of the simulator must be careful to ensure that
 * velocity scaling is applied before re-using the velocities after the thermostat.
 *
 * The time-scale separation between the particles and the thermostat requires the
 * NHC operator to have a higher-order factorization. The method used is the
 * Suzuki-Yoshida scheme which uses weighted time steps chosen to cancel out
 * lower-order error terms. Here, the fifth order SY scheme is used.
 */
static real applyNhc(real                 currentKineticEnergy,
                     ArrayRef<real>       xi,
                     ArrayRef<real>       xiVelocities,
                     ArrayRef<const real> invXiMass,
                     const real           numDofSystem,
                     const real           referenceTemperature,
                     const real           couplingTimeStep)
{
    if (currentKineticEnergy < 0)
    {
        return 1.0;
    }

    GMX_ASSERT(xi.size() == xiVelocities.size(),
               "Xi positions and velocities must have matching size.");
    const int chainLength = xi.size();

    constexpr unsigned int c_suzukiYoshidaOrder                         = 5;
    constexpr double       c_suzukiYoshidaWeights[c_suzukiYoshidaOrder] = {
        0.2967324292201065, 0.2967324292201065, -0.186929716880426, 0.2967324292201065, 0.2967324292201065
    };

    real velocityScalingFactor = 1.0;

    // Apply Suzuki-Yoshida scheme
    for (unsigned int syOuterLoop = 0; syOuterLoop < c_suzukiYoshidaOrder; ++syOuterLoop)
    {
        for (unsigned int syInnerLoop = 0; syInnerLoop < c_suzukiYoshidaOrder; ++syInnerLoop)
        {
            const real timeStep =
                    couplingTimeStep * c_suzukiYoshidaWeights[syInnerLoop] / c_suzukiYoshidaOrder;

            // Reverse loop - start from last thermostat in chain to update velocities,
            // because we need the new velocity to scale the next thermostat in the chain
            for (auto chainPosition = chainLength - 1; chainPosition >= 0; --chainPosition)
            {
                const real kineticEnergy2 =
                        ((chainPosition == 0) ? 2 * currentKineticEnergy
                                              : gmx::square(xiVelocities[chainPosition - 1])
                                                        / invXiMass[chainPosition - 1]);
                const real numDof         = ((chainPosition == 0) ? numDofSystem : 1);
                const real xiAcceleration = invXiMass[chainPosition]
                                            * (kineticEnergy2 - numDof * c_boltz * referenceTemperature);

                // We scale based on the next thermostat in chain.
                // Last thermostat in chain doesn't get scaled.
                const real localScalingFactor =
                        (chainPosition < chainLength - 1)
                                ? exp(-0.25 * timeStep * xiVelocities[chainPosition + 1])
                                : 1.0;
                xiVelocities[chainPosition] = localScalingFactor
                                              * (xiVelocities[chainPosition] * localScalingFactor
                                                 + 0.5 * timeStep * xiAcceleration);
            }

            // Calculate the new system scaling factor
            const real systemScalingFactor = std::exp(-timeStep * xiVelocities[0]);
            velocityScalingFactor *= systemScalingFactor;
            currentKineticEnergy *= systemScalingFactor * systemScalingFactor;

            // Forward loop - start from the system thermostat
            for (auto chainPosition = 0; chainPosition < chainLength; ++chainPosition)
            {
                // Update thermostat positions
                xi[chainPosition] += timeStep * xiVelocities[chainPosition];

                // Kinetic energy of system or previous chain member
                const real kineticEnergy2 =
                        ((chainPosition == 0) ? 2 * currentKineticEnergy
                                              : gmx::square(xiVelocities[chainPosition - 1])
                                                        / invXiMass[chainPosition - 1]);
                // DOF of system or previous chain member
                const real numDof         = ((chainPosition == 0) ? numDofSystem : 1);
                const real xiAcceleration = invXiMass[chainPosition]
                                            * (kineticEnergy2 - numDof * c_boltz * referenceTemperature);

                // We scale based on the next thermostat in chain.
                // Last thermostat in chain doesn't get scaled.
                const real localScalingFactor =
                        (chainPosition < chainLength - 1)
                                ? exp(-0.25 * timeStep * xiVelocities[chainPosition + 1])
                                : 1.0;
                xiVelocities[chainPosition] = localScalingFactor
                                              * (xiVelocities[chainPosition] * localScalingFactor
                                                 + 0.5 * timeStep * xiAcceleration);
            }
        }
    }
    return velocityScalingFactor;
}

/*!
 * \brief Calculate the current kinetic energy
 *
 * \param tcstat  The group's kinetic energy structure
 * \return real   The current kinetic energy
 */
inline real NoseHooverChainsElement::currentKineticEnergy(const t_grp_tcstat& tcstat)
{
    if (nhcUsage_ == NhcUsage::System)
    {
        if (useFullStepKE_ == UseFullStepKE::Yes)
        {
            return trace(tcstat.ekinf) * tcstat.ekinscalef_nhc;
        }
        else
        {
            return trace(tcstat.ekinh) * tcstat.ekinscaleh_nhc;
        }
    }
    else if (nhcUsage_ == NhcUsage::Barostat)
    {
        GMX_RELEASE_ASSERT(useFullStepKE_ == UseFullStepKE::Yes,
                           "Barostat NHC only works with full step KE.");
        return mttkData_->kineticEnergy();
    }
    else
    {
        gmx_fatal(FARGS, "Unknown NhcUsage.");
    }
}

void NoseHooverChainsElement::propagateNhc()
{
    auto* ekind = energyData_->ekindata();

    for (int temperatureGroup = 0; (temperatureGroup < noseHooverChainData_->numTemperatureGroups());
         temperatureGroup++)
    {
        auto nhcCoords     = noseHooverChainData_->coordinateView(temperatureGroup);
        auto scalingFactor = applyNhc(currentKineticEnergy(ekind->tcstat[temperatureGroup]),
                                      nhcCoords.positions,
                                      nhcCoords.velocities,
                                      nhcCoords.invMass,
                                      noseHooverChainData_->numDegreesOfFreedom(temperatureGroup),
                                      noseHooverChainData_->referenceTemperature(temperatureGroup),
                                      propagationTimeStep_);
        noseHooverChainData_->returnCoordinateView(std::move(nhcCoords), propagationTimeStep_);

        if (nhcUsage_ == NhcUsage::System)
        {
            // Scale system velocities by scalingFactor
            lambdaStartVelocities_[temperatureGroup] = scalingFactor;
            // Scale kinetic energy by scalingFactor^2
            ekind->tcstat[temperatureGroup].ekinscaleh_nhc *= scalingFactor * scalingFactor;
            ekind->tcstat[temperatureGroup].ekinscalef_nhc *= scalingFactor * scalingFactor;
        }
        else if (nhcUsage_ == NhcUsage::Barostat)
        {
            // Scale eta velocities by scalingFactor
            mttkData_->scale(scalingFactor);
        }
    }

    if (nhcUsage_ == NhcUsage::System && noseHooverChainData_->isAtFullCouplingTimeStep())
    {
        // We've set the scaling factors for the full time step, so scale
        // kinetic energy accordingly before it gets printed
        energyData_->updateKineticEnergy();
    }
}

NoseHooverChainsElement::NoseHooverChainsElement(int                   nstcouple,
                                                 int                   offset,
                                                 NhcUsage              nhcUsage,
                                                 UseFullStepKE         useFullStepKE,
                                                 double                propagationTimeStep,
                                                 ScheduleOnInitStep    scheduleOnInitStep,
                                                 Step                  initStep,
                                                 EnergyData*           energyData,
                                                 NoseHooverChainsData* noseHooverChainData,
                                                 MttkData*             mttkData) :
    nsttcouple_(nstcouple),
    offset_(offset),
    propagationTimeStep_(propagationTimeStep),
    nhcUsage_(nhcUsage),
    useFullStepKE_(useFullStepKE),
    scheduleOnInitStep_(scheduleOnInitStep),
    initialStep_(initStep),
    energyData_(energyData),
    noseHooverChainData_(noseHooverChainData),
    mttkData_(mttkData)
{
}

void NoseHooverChainsElement::elementSetup()
{
    GMX_RELEASE_ASSERT(
            !(nhcUsage_ == NhcUsage::System && !propagatorCallback_),
            "Nose-Hoover chain element was not connected to a propagator.\n"
            "Connection to a propagator element is needed to scale the velocities.\n"
            "Use connectWithPropagator(...) before building the ModularSimulatorAlgorithm "
            "object.");
}

void NoseHooverChainsElement::scheduleTask(Step step, Time /*unused*/, const RegisterRunFunction& registerRunFunction)
{
    if (step == initialStep_ && scheduleOnInitStep_ == ScheduleOnInitStep::No)
    {
        return;
    }
    if (do_per_step(step + nsttcouple_ + offset_, nsttcouple_))
    {
        // do T-coupling this step
        registerRunFunction([this]() { propagateNhc(); });

        if (propagatorCallback_)
        {
            // Let propagator know that we want to do T-coupling
            propagatorCallback_(step);
        }
    }
}

void NoseHooverChainsElement::connectWithPropagator(const PropagatorConnection& connectionData,
                                                    const PropagatorTag&        propagatorTag)
{
    if (connectionData.tag == propagatorTag)
    {
        GMX_RELEASE_ASSERT(connectionData.hasStartVelocityScaling(),
                           "Trotter NHC needs start velocity scaling.");
        connectionData.setNumVelocityScalingVariables(noseHooverChainData_->numTemperatureGroups(),
                                                      ScaleVelocities::PreStepOnly);
        lambdaStartVelocities_ = connectionData.getViewOnStartVelocityScaling();
        propagatorCallback_    = connectionData.getVelocityScalingCallback();
    }
}

//! \cond
// Doxygen gets confused by the overload
ISimulatorElement* NoseHooverChainsElement::getElementPointerImpl(
        LegacySimulatorData*                    legacySimulatorData,
        ModularSimulatorAlgorithmBuilderHelper* builderHelper,
        StatePropagatorData gmx_unused* statePropagatorData,
        EnergyData*                     energyData,
        FreeEnergyPerturbationData gmx_unused* freeEnergyPerturbationData,
        GlobalCommunicationHelper gmx_unused* globalCommunicationHelper,
        NhcUsage                              nhcUsage,
        Offset                                offset,
        UseFullStepKE                         useFullStepKE,
        ScheduleOnInitStep                    scheduleOnInitStep)
{
    GMX_RELEASE_ASSERT(nhcUsage == NhcUsage::Barostat, "System NHC element needs a propagator tag.");
    return getElementPointerImpl(legacySimulatorData,
                                 builderHelper,
                                 statePropagatorData,
                                 energyData,
                                 freeEnergyPerturbationData,
                                 globalCommunicationHelper,
                                 nhcUsage,
                                 offset,
                                 useFullStepKE,
                                 scheduleOnInitStep,
                                 PropagatorTag(""));
}

ISimulatorElement* NoseHooverChainsElement::getElementPointerImpl(
        LegacySimulatorData*                    legacySimulatorData,
        ModularSimulatorAlgorithmBuilderHelper* builderHelper,
        StatePropagatorData gmx_unused* statePropagatorData,
        EnergyData*                     energyData,
        FreeEnergyPerturbationData gmx_unused* freeEnergyPerturbationData,
        GlobalCommunicationHelper gmx_unused* globalCommunicationHelper,
        NhcUsage                              nhcUsage,
        Offset                                offset,
        UseFullStepKE                         useFullStepKE,
        ScheduleOnInitStep                    scheduleOnInitStep,
        const PropagatorTag&                  propagatorTag)
{
    if (!builderHelper->simulationData<NoseHooverChainsData>(NoseHooverChainsData::dataID(nhcUsage)))
    {
        NoseHooverChainsData::build(nhcUsage, legacySimulatorData, builderHelper, energyData);
    }
    auto* nhcData = builderHelper
                            ->simulationData<NoseHooverChainsData>(NoseHooverChainsData::dataID(nhcUsage))
                            .value();

    // MTTK data is only needed when connecting to a barostat
    MttkData* mttkData = nullptr;
    if (nhcUsage == NhcUsage::Barostat)
    {
        if (!builderHelper->simulationData<MttkData>(MttkData::dataID()))
        {
            MttkData::build(legacySimulatorData, builderHelper, statePropagatorData, energyData);
        }
        mttkData = builderHelper->simulationData<MttkData>(MttkData::dataID()).value();
    }

    // Element is now owned by the caller of this method, who will handle lifetime (see ModularSimulatorAlgorithm)
    auto* element = builderHelper->storeElement(std::make_unique<NoseHooverChainsElement>(
            legacySimulatorData->inputrec->nsttcouple,
            offset,
            nhcUsage,
            useFullStepKE,
            legacySimulatorData->inputrec->delta_t * legacySimulatorData->inputrec->nsttcouple / 2,
            scheduleOnInitStep,
            legacySimulatorData->inputrec->init_step,
            energyData,
            nhcData,
            mttkData));
    if (nhcUsage == NhcUsage::System)
    {
        auto* thermostat = static_cast<NoseHooverChainsElement*>(element);
        // Capturing pointer is safe because lifetime is handled by caller
        builderHelper->registerTemperaturePressureControl(
                [thermostat, propagatorTag](const PropagatorConnection& connection) {
                    thermostat->connectWithPropagator(connection, propagatorTag);
                });
    }
    else
    {
        GMX_RELEASE_ASSERT(propagatorTag == PropagatorTag(""),
                           "Propagator tag is unused for Barostat NHC element.");
    }
    return element;
}
//! \endcond

} // namespace gmx
