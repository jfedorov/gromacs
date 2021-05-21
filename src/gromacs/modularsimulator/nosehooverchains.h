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
 * \brief Declares classes related to Nose-Hoover chains for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 *
 * This header is only used within the modular simulator module
 */

#ifndef GMX_MODULARSIMULATOR_NOSEHOOVERCHAINS_H
#define GMX_MODULARSIMULATOR_NOSEHOOVERCHAINS_H

#include <queue>

#include "modularsimulatorinterfaces.h"

struct t_grp_tcstat;

namespace gmx
{
class EnergyData;
class FreeEnergyPerturbationData;
class GlobalCommunicationHelper;
class LegacySimulatorData;
class ModularSimulatorAlgorithmBuilderHelper;
class MttkData;
struct NhcCoordinateView;
class StatePropagatorData;
enum class UseFullStepKE;

//! Whether the element does schedule on the initial step
enum class ScheduleOnInitStep
{
    Yes,  //!< Schedule on first step
    No,   //!< Do not schedule on first step
    Count //!< Number of enum entries
};

//! The usages of Nose-Hoover chains
enum class NhcUsage
{
    System,   //!< Couple system to temperature bath
    Barostat, //!< Couple barostat to temperature bath
    Count     //!< Number of enum entries
};

/*! \internal
 * \brief Class holding data used by the Nose-Hoover chains
 *
 * As the Trotter update is split in several sub-steps (i.e. is updated
 * by several element instances), the NHC degrees of freedom must be
 * stored centrally rather than by the single elements.
 *
 * This class manages these extra degrees of freedom. It controls access
 * (making sure that only one element has write access at a time), keeps
 * track of the current time stamp of the dofs, calculates the energy
 * related to the dof at the requested times, and writes the data needed
 * for restarts to checkpoint. As this is not implementing the
 * ISimulatorElement interface, it is not part of the simulator loop, but
 * relies on callbacks to perform it's duties.
 */
class NoseHooverChainsData final : public ICheckpointHelperClient
{
public:
    //! Constructor
    NoseHooverChainsData(int                  numTemperatureGroups,
                         real                 couplingTimeStep,
                         int                  chainLength,
                         ArrayRef<const real> referenceTemperature,
                         ArrayRef<const real> couplingTime,
                         ArrayRef<const real> numDegreesOfFreedom,
                         EnergyData*          energyData,
                         NhcUsage             nhcUsage);

    /*! \brief Get view on the NHC coordinates
     *
     * \param temperatureGroup  The temperature group of the requested coordinates
     * \return  A view on the current coordinates
     */
    NhcCoordinateView coordinateView(int temperatureGroup);
    /*! \brief Return a view on the NHC coordinates
     *
     * This call represents an agreement that the caller will not continue to use
     * a previously requested coordinate view. The caller also informs this object
     * of the time increment it performed on the view.
     *
     * \param nhcCoordinateView  The coordinate view
     * \param timeIncrement  By how much the coordinates have been propagated
     */
    void returnCoordinateView(NhcCoordinateView nhcCoordinateView, real timeIncrement);

    //! The number of temperature groups
    int numTemperatureGroups() const;
    //! Coupling temperature for temperature group
    real referenceTemperature(int temperatureGroup) const;
    //! Number of degrees for temperature group
    real numDegreesOfFreedom(int temperatureGroup) const;
    //! Whether the NHC dofs are at a full coupling time step
    bool isAtFullCouplingTimeStep() const;

    //! ICheckpointHelperClient write checkpoint implementation
    void saveCheckpointState(std::optional<WriteCheckpointData> checkpointData, const t_commrec* cr) override;
    //! ICheckpointHelperClient read checkpoint implementation
    void restoreCheckpointState(std::optional<ReadCheckpointData> checkpointData, const t_commrec* cr) override;
    //! ICheckpointHelperClient key implementation
    const std::string& clientID() override;

    //! Build object and store in builder helper object
    static void build(NhcUsage                                nhcUsage,
                      LegacySimulatorData*                    legacySimulatorData,
                      ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                      EnergyData*                             energyData);

    //! Identifier used to store objects
    static std::string dataID(NhcUsage nhcUsage);

private:
    //! Calculate the current value of the NHC integral for a temperature group
    void calculateIntegral(int temperatureGroup);
    //! Return the value of the coupling integral at a specific time
    double temperatureCouplingIntegral(Time time) const;
    //! Whether a temperature group dof is at a full coupling time step
    bool isAtFullCouplingTimeStep(int temperatureGroup) const;

    //! CheckpointHelper identifier
    const std::string identifier_;
    //! Helper function to read from / write to CheckpointData
    template<CheckpointDataOperation operation>
    void doCheckpointData(CheckpointData<operation>* checkpointData);

    //! The thermostat degree of freedom
    std::vector<std::vector<real>> xi_;
    //! Velocity of the thermostat dof
    std::vector<std::vector<real>> xiVelocities_;
    //! Work exerted by thermostat per group
    std::vector<double> temperatureCouplingIntegral_;
    //! Inverse mass of the thermostat dof
    std::vector<std::vector<real>> invXiMass_;
    //! Whether the xi view is currently in use
    std::vector<bool> coordinateViewInUse_;
    //! The current time of xi and xiVelocities
    std::vector<real> coordinateTime_;
    //! The current time of the temperature integral
    std::vector<real> integralTime_;

    //! The coupling time step
    const real couplingTimeStep_;
    //! The length of the Nose-Hoover chains
    const int chainLength_;
    //! The number of temperature groups
    const int numTemperatureGroups_;
    //! Coupling temperature per group
    ArrayRef<const real> referenceTemperature_;
    //! Coupling time per group
    ArrayRef<const real> couplingTime_;
    //! Number of degrees of freedom per group
    ArrayRef<const real> numDegreesOfFreedom_;
};

/*! \internal
 * \brief Element propagating the Nose-Hoover chains
 *
 * This propagates the Nose-Hoover chain degrees of freedom, and
 * transmits the scaling factor to a connected propagator.
 */
class NoseHooverChainsElement final : public ISimulatorElement
{
public:
    //! Constructor
    NoseHooverChainsElement(int                   nstcouple,
                            int                   offset,
                            NhcUsage              nhcUsage,
                            UseFullStepKE         useFullStepKE,
                            double                propagationTimeStep,
                            ScheduleOnInitStep    scheduleOnInitStep,
                            Step                  initStep,
                            EnergyData*           energyData,
                            NoseHooverChainsData* noseHooverChainData,
                            MttkData*             mttkData);

    /*! \brief Register run function for step / time
     *
     * \param step                 The step number
     * \param time                 The time
     * \param registerRunFunction  Function allowing to register a run function
     */
    void scheduleTask(Step step, Time time, const RegisterRunFunction& registerRunFunction) override;

    //! Sanity check at setup time
    void elementSetup() override;
    //! No element teardown needed
    void elementTeardown() override {}

    //! Connect this to propagator
    void connectWithPropagator(const PropagatorConnection& connectionData,
                               const PropagatorTag&        propagatorTag);

    /*! \brief Factory method implementation (no propagator connection)
     *
     * \param legacySimulatorData  Pointer allowing access to simulator level data
     * \param builderHelper  ModularSimulatorAlgorithmBuilder helper object
     * \param statePropagatorData  Pointer to the \c StatePropagatorData object
     * \param energyData  Pointer to the \c EnergyData object
     * \param freeEnergyPerturbationData  Pointer to the \c FreeEnergyPerturbationData object
     * \param globalCommunicationHelper  Pointer to the \c GlobalCommunicationHelper object
     * \param nhcUsage  What the NHC is connected to - system or barostat
     * \param offset  The step offset at which the thermostat is applied
     * \param useFullStepKE  Whether full step or half step KE is used
     * \param scheduleOnInitStep  Whether the element is scheduled on the initial step
     *
     * \return  Pointer to the element to be added. Element needs to have been stored using \c storeElement
     */
    static ISimulatorElement* getElementPointerImpl(LegacySimulatorData* legacySimulatorData,
                                                    ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                                    StatePropagatorData*        statePropagatorData,
                                                    EnergyData*                 energyData,
                                                    FreeEnergyPerturbationData* freeEnergyPerturbationData,
                                                    GlobalCommunicationHelper* globalCommunicationHelper,
                                                    NhcUsage                   nhcUsage,
                                                    Offset                     offset,
                                                    UseFullStepKE              useFullStepKE,
                                                    ScheduleOnInitStep         scheduleOnInitStep);

    /*! \brief Factory method implementation (including propagator connection)
     *
     * \param legacySimulatorData  Pointer allowing access to simulator level data
     * \param builderHelper  ModularSimulatorAlgorithmBuilder helper object
     * \param statePropagatorData  Pointer to the \c StatePropagatorData object
     * \param energyData  Pointer to the \c EnergyData object
     * \param freeEnergyPerturbationData  Pointer to the \c FreeEnergyPerturbationData object
     * \param globalCommunicationHelper  Pointer to the \c GlobalCommunicationHelper object
     * \param nhcUsage  What the NHC is connected to - system or barostat
     * \param offset  The step offset at which the thermostat is applied
     * \param useFullStepKE  Whether full step or half step KE is used
     * \param scheduleOnInitStep  Whether the element is scheduled on the initial step
     * \param propagatorTag  Tag of the propagator to connect to
     *
     * \return  Pointer to the element to be added. Element needs to have been stored using \c storeElement
     */
    static ISimulatorElement* getElementPointerImpl(LegacySimulatorData* legacySimulatorData,
                                                    ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                                    StatePropagatorData*        statePropagatorData,
                                                    EnergyData*                 energyData,
                                                    FreeEnergyPerturbationData* freeEnergyPerturbationData,
                                                    GlobalCommunicationHelper* globalCommunicationHelper,
                                                    NhcUsage                   nhcUsage,
                                                    Offset                     offset,
                                                    UseFullStepKE              useFullStepKE,
                                                    ScheduleOnInitStep         scheduleOnInitStep,
                                                    const PropagatorTag&       propagatorTag);

private:
    //! Propagate the NHC degrees of freedom
    void propagateNhc();
    //! Helper function returning the appropriate kinetic energy
    real currentKineticEnergy(const t_grp_tcstat& tcstat);

    //! View on the scaling factor of the propagator (pre-step velocities)
    ArrayRef<real> lambdaStartVelocities_;
    //! Callback to let propagator know that we will update lambda
    PropagatorCallback propagatorCallback_;

    //! The frequency at which the thermostat is applied
    const int nsttcouple_;
    //! If != 0, offset the step at which the thermostat is applied
    const int offset_;
    //! The propagation time step - by how much we propagate the NHC dof
    const double propagationTimeStep_;
    //! Whether this NHC is acting on the system or a barostat
    const NhcUsage nhcUsage_;
    //! Whether we're using full step kinetic energy
    const UseFullStepKE useFullStepKE_;
    //! Whether we're scheduling on the first step
    const ScheduleOnInitStep scheduleOnInitStep_;
    //! The initial step number
    const Step initialStep_;

    // TODO: Clarify relationship to data objects and find a more robust alternative to raw pointers (#3583)
    //! Pointer to the energy data (for ekindata)
    EnergyData* energyData_;
    //! Pointer to the NHC data
    NoseHooverChainsData* noseHooverChainData_;
    //! Pointer to the MTTK data (nullptr if this is not connected to barostat)
    MttkData* mttkData_;
};


} // namespace gmx

#endif // GMX_MODULARSIMULATOR_NOSEHOOVERCHAINS_H
