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
 * \brief Declares classes related to MTTK pressure coupling
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 *
 * This header is only used within the modular simulator module
 */

#ifndef GMX_MODULARSIMULATOR_MTTK_H
#define GMX_MODULARSIMULATOR_MTTK_H

#include <queue>

#include "modularsimulatorinterfaces.h"
#include "statepropagatordata.h"

namespace gmx
{
class EnergyData;
enum class ScheduleOnInitStep;

/*! \internal
 * \brief Class holding the extra dof and parameters used by the MTTK algorithm
 *
 * As the Trotter update is split in several sub-steps (i.e. is updated
 * by several element instances), the MTTK degrees of freedom must be
 * stored centrally rather than by the single elements.
 *
 * This class manages these extra degrees of freedom. It controls access, keeps
 * track of the current time stamp of the dofs, calculates the energy
 * related to the dof at the requested times, and writes the data needed
 * for restarts to checkpoint. As this is not implementing the
 * ISimulatorElement interface, it is not part of the simulator loop, but
 * relies on callbacks to perform its duties.
 */
class MttkData final : public ICheckpointHelperClient
{
public:
    //! Constructor
    MttkData(real                       referenceTemperature,
             real                       referencePressure,
             real                       couplingTimeStep,
             real                       couplingTime,
             real                       initialVolume,
             const tensor               compressibility,
             const StatePropagatorData* statePropagatorData);

    //! Explicit copy constructor (interface has a standard destructor)
    MttkData(const MttkData& other);

    //! The current kinetic energy of the MTTK degree of freedom
    [[nodiscard]] real kineticEnergy() const;
    //! Scale the MTTK dof velocity
    void scale(real scalingFactor);

    //! The current MTTK dof velocity
    [[nodiscard]] real etaVelocity() const;
    //! Set a new MTTK velocity
    void setEtaVelocity(real etaVelocity, real etaVelocityTimeIncrement);

    //! Get the inverse mass of the MTTK degree of freedom
    [[nodiscard]] real invEtaMass() const;
    //! Get the reference pressure
    real referencePressure() const;
    //! Pointer to the box velocities
    [[nodiscard]] rvec* boxVelocities();

    //! ICheckpointHelperClient write checkpoint implementation
    void saveCheckpointState(std::optional<WriteCheckpointData> checkpointData, const t_commrec* cr) override;
    //! ICheckpointHelperClient read checkpoint implementation
    void restoreCheckpointState(std::optional<ReadCheckpointData> checkpointData, const t_commrec* cr) override;
    //! ICheckpointHelperClient key implementation
    const std::string& clientID() override;

    //! Build object and store in builder helper object
    static void build(LegacySimulatorData*                    legacySimulatorData,
                      ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                      StatePropagatorData*                    statePropagatorData,
                      EnergyData*                             energyData);

    //! Identifier used to store objects
    static std::string dataID();

private:
    //! Return the current value of the MTTK dof contribution to the conserved energy
    double temperatureCouplingIntegral(Time time) const;
    //! Calculate the current value of the MTTK conserved energy if it is needed
    void calculateIntegralIfNeeded();
    //! Update the reference temperature
    void updateReferenceTemperature(real temperature, ReferenceTemperatureChangeAlgorithm algorithm);

    //! The coupling time step
    const real couplingTimeStep_;
    //! The velocity of the MTTK dof
    real etaVelocity_;
    //! The inverse mass of the MTTK dof
    real invMass_;
    //! The current time stamp of the MTTK dof velocity
    Time etaVelocityTime_;
    //! The current value of the MTTK dof contribution to the conserved energy
    double temperatureCouplingIntegral_;
    //! The current time stamp of the conserved energy contribution
    Time integralTime_;
    //! The reference pressure
    const real referencePressure_;
    //! The current box velocities (used for reporting only)
    tensor boxVelocity_;
    //! The reference temperature the mass is based on
    real referenceTemperature_;

    // TODO: Clarify relationship to data objects and find a more robust alternative to raw pointers (#3583)
    //! Pointer to the micro state data (access to the current box)
    const StatePropagatorData* statePropagatorData_;

    //! CheckpointHelper identifier
    const std::string identifier_ = "MttkData";
    //! Helper function to read from / write to CheckpointData
    template<CheckpointDataOperation operation>
    void doCheckpointData(CheckpointData<operation>* checkpointData);
};

/*! \internal
 * \brief Object holding the callbacks and scaling views for the connection
 *        of MTTKElement objects to propagators
 *
 * As the Trotter update is split in several sub-steps (i.e. is updated
 * by several element instances), the connection to propagators must be
 * stored centrally rather than by the single elements.
 */
class MttkPropagatorConnection
{
public:
    //! Build object and store in builder helper object
    static void build(ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                      const PropagatorTag&                    propagatorTagPrePosition,
                      const PropagatorTag&                    propagatorTagPostPosition,
                      int                                     positionOffset,
                      const PropagatorTag&                    propagatorTagPreVelocity1,
                      const PropagatorTag&                    propagatorTagPostVelocity1,
                      int                                     velocityOffset1,
                      const PropagatorTag&                    propagatorTagPreVelocity2,
                      const PropagatorTag&                    propagatorTagPostVelocity2,
                      int                                     velocityOffset2);

    //! Call the propagator callbacks
    void propagatorCallback(Step step) const;

    //! Set position scaling
    void setPositionScaling(real preStepScaling, real postStepScaling);
    //! Set velocity scaling
    void setVelocityScaling(real preStepScaling, real postStepScaling);

    //! Identifier used to store objects
    static std::string dataID();

private:
    //! Connect to pre-step velocity scaling
    void connectWithPropagatorVelocityPreStepScaling(const PropagatorConnection& connectionData,
                                                     const PropagatorTag&        propagatorTag,
                                                     int                         offset);
    //! Connect to post-step velocity scaling
    void connectWithPropagatorVelocityPostStepScaling(const PropagatorConnection& connectionData,
                                                      const PropagatorTag&        propagatorTag,
                                                      int                         offset);
    //! Connect to pre-step position scaling
    void connectWithPropagatorPositionPreStepScaling(const PropagatorConnection& connectionData,
                                                     const PropagatorTag&        propagatorTag,
                                                     int                         offset);
    //! Connect to post-step position scaling
    void connectWithPropagatorPositionPostStepScaling(const PropagatorConnection& connectionData,
                                                      const PropagatorTag&        propagatorTag,
                                                      int                         offset);

    //! View on the scaling factors of the position propagators (before step)
    std::vector<ArrayRef<real>> startPositionScalingFactors_;
    //! View on the scaling factors of the position propagators (after step)
    std::vector<ArrayRef<real>> endPositionScalingFactors_;
    //! View on the scaling factors of the velocity propagators (before step)
    std::vector<ArrayRef<real>> startVelocityScalingFactors_;
    //! View on the scaling factors of the velocity propagators (after step)
    std::vector<ArrayRef<real>> endVelocityScalingFactors_;
    //! Callbacks to let propagators know that we will update scaling factor, plus their offset
    std::vector<std::tuple<PropagatorCallback, int>> propagatorCallbacks_;
};

/*! \internal
 * \brief Element propagating the MTTK degree of freedom
 *
 * This roughly follows Tuckerman et al. 2006 (DOI: 10.1088/0305-4470/39/19/S18)
 *
 * This currently only implements the isotropic version of MTTK and does
 * not work with constraints. It also adopts some conventions chosen by
 * the legacy GROMACS implementation, such as using the number of degrees
 * of freedom and the kinetic energy scaling of the first temperature group
 * to calculate the current pressure. (It does use the full system kinetic
 * energy, however!)
 */
class MttkElement final : public ISimulatorElement
{
public:
    //! Constructor
    MttkElement(int                        nstcouple,
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
                real                       numDegreesOfFreedom);

    /*! \brief Register run function for step / time
     *
     * \param step                 The step number
     * \param time                 The time
     * \param registerRunFunction  Function allowing to register a run function
     */
    void scheduleTask(Step step, Time time, const RegisterRunFunction& registerRunFunction) override;

    //! No element setup needed
    void elementSetup() override {}
    //! No element teardown needed
    void elementTeardown() override {}

    /*! \brief Factory method implementation
     *
     * \param legacySimulatorData  Pointer allowing access to simulator level data
     * \param builderHelper  ModularSimulatorAlgorithmBuilder helper object
     * \param statePropagatorData  Pointer to the \c StatePropagatorData object
     * \param energyData  Pointer to the \c EnergyData object
     * \param freeEnergyPerturbationData  Pointer to the \c FreeEnergyPerturbationData object
     * \param globalCommunicationHelper  Pointer to the \c GlobalCommunicationHelper object
     * \param offset  The step offset at which the thermostat is applied
     * \param scheduleOnInitStep  Whether the element is scheduled on the initial step
     * \param propagatorTagPrePosition  Tag of the propagators to scale positions before propagation
     * \param propagatorTagPostPosition  Tag of the propagators to scale positions after propagation
     * \param positionOffset  The step offset at which the position scaling is applied
     * \param propagatorTagPreVelocity1  Tag of the propagators to scale velocities before first propagation
     * \param propagatorTagPostVelocity1  Tag of the propagators to scale velocities after first propagation
     * \param velocityOffset1  The step offset at which the first velocity scaling is applied
     * \param propagatorTagPreVelocity2  Tag of the propagators to scale velocities before second propagation
     * \param propagatorTagPostVelocity2  Tag of the propagators to scale velocities after second propagation
     * \param velocityOffset2  The step offset at which the second velocity scaling is applied
     *
     * \return  Pointer to the element to be added. Element needs to have been stored using \c storeElement
     */
    static ISimulatorElement* getElementPointerImpl(LegacySimulatorData* legacySimulatorData,
                                                    ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                                    StatePropagatorData*        statePropagatorData,
                                                    EnergyData*                 energyData,
                                                    FreeEnergyPerturbationData* freeEnergyPerturbationData,
                                                    GlobalCommunicationHelper* globalCommunicationHelper,
                                                    Offset                     offset,
                                                    ScheduleOnInitStep         scheduleOnInitStep,
                                                    const PropagatorTag& propagatorTagPrePosition,
                                                    const PropagatorTag& propagatorTagPostPosition,
                                                    Offset               positionOffset,
                                                    const PropagatorTag& propagatorTagPreVelocity1,
                                                    const PropagatorTag& propagatorTagPostVelocity1,
                                                    Offset               velocityOffset1,
                                                    const PropagatorTag& propagatorTagPreVelocity2,
                                                    const PropagatorTag& propagatorTagPostVelocity2,
                                                    Offset               velocityOffset2);

private:
    //! Propagation of the MTTK dof velocity
    void propagateEtaVelocity(Step step);

    //! The periodic boundary condition type
    const PbcType pbcType_;
    //! The number of walls
    const int numWalls_;
    //! The number of degrees of freedom in the first temperature group
    const real numDegreesOfFreedom_;

    //! The period at which the thermostat is applied
    const int nstcouple_;
    //! If != 0, offset the step at which the thermostat is applied
    const int offset_;
    //! The propagation time step - by how much we propagate the MTTK dof
    const real propagationTimeStep_;
    //! The simulation time step - by how much the propagators are advancing the positions
    const real simulationTimeStep_;
    //! Whether we're scheduling on the first step
    const ScheduleOnInitStep scheduleOnInitStep_;
    //! The initial step number
    const Step initialStep_;

    // TODO: Clarify relationship to data objects and find a more robust alternative to raw pointers (#3583)
    //! Pointer to the micro state
    const StatePropagatorData* statePropagatorData_;
    //! Pointer to the energy data (for ekindata)
    EnergyData* energyData_;
    //! Pointer to the MTTK data
    MttkData* mttkData_;
    //! Pointer to the propagator connection object
    MttkPropagatorConnection* mttkPropagatorConnection_;
};

/*! \internal
 * \brief This element scales the box based on the MTTK dof
 */
class MttkBoxScaling final : public ISimulatorElement
{
public:
    //! Constructor
    MttkBoxScaling(real simulationTimeStep, StatePropagatorData* statePropagatorData, MttkData* mttkData);

    /*! \brief Register run function for step / time
     *
     * \param step                 The step number
     * \param time                 The time
     * \param registerRunFunction  Function allowing to register a run function
     */
    void scheduleTask(Step step, Time time, const RegisterRunFunction& registerRunFunction) override;

    //! Sanity check at setup time
    void elementSetup() override {}
    //! No element teardown needed
    void elementTeardown() override {}

    /*! \brief Factory method implementation
     *
     * \param legacySimulatorData  Pointer allowing access to simulator level data
     * \param builderHelper  ModularSimulatorAlgorithmBuilder helper object
     * \param statePropagatorData  Pointer to the \c StatePropagatorData object
     * \param energyData  Pointer to the \c EnergyData object
     * \param freeEnergyPerturbationData  Pointer to the \c FreeEnergyPerturbationData object
     * \param globalCommunicationHelper  Pointer to the \c GlobalCommunicationHelper object
     *
     * \return  Pointer to the element to be added. Element needs to have been stored using \c storeElement
     */
    static ISimulatorElement* getElementPointerImpl(LegacySimulatorData* legacySimulatorData,
                                                    ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                                    StatePropagatorData*        statePropagatorData,
                                                    EnergyData*                 energyData,
                                                    FreeEnergyPerturbationData* freeEnergyPerturbationData,
                                                    GlobalCommunicationHelper* globalCommunicationHelper);

private:
    //! Scale the box
    void scaleBox();
    //! The simulation time step
    const real simulationTimeStep_;

    // TODO: Clarify relationship to data objects and find a more robust alternative to raw pointers (#3583)
    //! Pointer to the micro state
    StatePropagatorData* statePropagatorData_;
    //! Pointer to the MTTK data (nullptr if this is not connected to barostat)
    MttkData* mttkData_;
};

} // namespace gmx

#endif // GMX_MODULARSIMULATOR_MTTK_H
