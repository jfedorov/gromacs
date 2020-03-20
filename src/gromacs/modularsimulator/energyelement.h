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
/*! \libinternal \file
 * \brief Declares the energy element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#ifndef GMX_MODULARSIMULATOR_ENERGYELEMENT_H
#define GMX_MODULARSIMULATOR_ENERGYELEMENT_H

#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/energyoutput.h"
#include "gromacs/mdtypes/state.h"

#include "modularsimulatorinterfaces.h"

struct gmx_ekindata_t;
struct gmx_enerdata_t;
struct gmx_mtop_t;
struct ObservablesHistory;
struct t_fcdata;
struct t_inputrec;
struct SimulationGroups;

namespace gmx
{
enum class StartingBehavior;
class CheckpointHelperBuilder;
class Constraints;
class FreeEnergyPerturbationElement;
class MDAtoms;
class ParrinelloRahmanBarostat;
class StatePropagatorData;
class TopologyHolder;
class TrajectoryElementBuilder;
class VRescaleThermostat;
struct MdModulesNotifier;

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Element managing energies
 *
 * The EnergyElement owns the EnergyObject, and is hence responsible
 * for saving energy data and writing it to trajectory. It also owns
 * the tensors for the different virials and the pressure as well as
 * the total dipole vector.
 *
 * It subscribes to the trajectory signaller, the energy signaller,
 * and the logging signaller to know when an energy calculation is
 * needed and when a non-recording step is enough. The simulator
 * builder is responsible to place the element in a location at
 * which a valid energy state is available. The EnergyElement is
 * also a subscriber to the trajectory writer element, as it is
 * responsible to write energy data to trajectory.
 *
 * The EnergyElement offers an interface to add virial contributions,
 * but also allows access to the raw pointers to tensor data, the
 * dipole vector, and the legacy energy data structures.
 */
class EnergyElement final :
    public ISimulatorElement,
    public ITrajectoryWriterClient,
    public ITrajectorySignallerClient,
    public IEnergySignallerClient,
    public ICheckpointHelperClient
{
public:
    /*! \brief Register run function for step / time
     *
     * This needs to be called when the energies are at a full time step.
     * Positioning this element is the responsibility of the programmer.
     *
     * This is also the place at which the current state becomes the previous
     * state.
     *
     * @param step                 The step number
     * @param time                 The time
     * @param registerRunFunction  Function allowing to register a run function
     */
    void scheduleTask(Step step, Time time, const RegisterRunFunctionPtr& registerRunFunction) override;

    //! No element setup needed
    void elementSetup() override {}

    /*! \brief Final output
     *
     * Prints the averages to log.
     */
    void elementTeardown() override;

    /*! \brief Add contribution to force virial
     *
     * This automatically resets the tensor if the step is higher
     * than the current step, starting the tensor calculation for
     * a new step at zero. Otherwise, it adds the new contribution
     * to the existing virial.
     */
    void addToForceVirial(const tensor virial, Step step);

    /*! \brief Add contribution to constraint virial
     *
     * This automatically resets the tensor if the step is higher
     * than the current step, starting the tensor calculation for
     * a new step at zero. Otherwise, it adds the new contribution
     * to the existing virial.
     */
    void addToConstraintVirial(const tensor virial, Step step);

    /*! \brief Get pointer to force virial tensor
     *
     * Allows access to the raw pointer to the tensor.
     */
    rvec* forceVirial(Step step);

    /*! \brief Get pointer to constraint virial tensor
     *
     * Allows access to the raw pointer to the tensor.
     */
    rvec* constraintVirial(Step step);

    /*! \brief Get pointer to total virial tensor
     *
     * Allows access to the raw pointer to the tensor.
     */
    rvec* totalVirial(Step step);

    /*! \brief Get pointer to pressure tensor
     *
     * Allows access to the raw pointer to the tensor.
     */
    rvec* pressure(Step step);

    /*! \brief Get pointer to mu_tot
     *
     * Allows access to the raw pointer to the dipole vector.
     */
    real* muTot();

    /*! \brief Get pointer to energy structure
     *
     */
    gmx_enerdata_t* enerdata();

    /*! \brief Get pointer to kinetic energy structure
     *
     */
    gmx_ekindata_t* ekindata();

    /*! \brief Get pointer to needToSumEkinhOld
     *
     */
    bool* needToSumEkinhOld();

    /*! \brief Initialize energy history
     *
     * Kept as a static function to allow usage from legacy code
     * \todo Make member function once legacy use is not needed anymore
     */
    static void initializeEnergyHistory(StartingBehavior    startingBehavior,
                                        ObservablesHistory* observablesHistory,
                                        EnergyOutput*       energyOutput);

    //! Allow builder to do its job
    friend class EnergyElementBuilder;

private:
    //! Constructor
    EnergyElement(const t_inputrec*        inputrec,
                  const MDAtoms*           mdAtoms,
                  gmx_enerdata_t*          enerd,
                  gmx_ekindata_t*          ekind,
                  const Constraints*       constr,
                  FILE*                    fplog,
                  t_fcdata*                fcd,
                  const MdModulesNotifier& mdModulesNotifier,
                  bool                     isMasterRank,
                  ObservablesHistory*      observablesHistory,
                  StartingBehavior         startingBehavior);

    /*! \brief Setup (needs file pointer)
     *
     * ITrajectoryWriterClient implementation.
     *
     * Initializes the EnergyOutput object, and does some logging output.
     *
     * @param mdoutf  File pointer
     */
    void trajectoryWriterSetup(gmx_mdoutf* mdoutf) override;
    //! No trajectory writer teardown needed
    void trajectoryWriterTeardown(gmx_mdoutf gmx_unused* outf) override {}

    //! ITrajectoryWriterClient implementation.
    SignallerCallbackPtr registerTrajectorySignallerCallback(TrajectoryEvent event) override;
    //! ITrajectorySignallerClient implementation
    ITrajectoryWriterCallbackPtr registerTrajectoryWriterCallback(TrajectoryEvent event) override;
    //! IEnergySignallerClient implementation
    SignallerCallbackPtr registerEnergyCallback(EnergySignallerEvent event) override;

    /*! \brief Save data at energy steps
     *
     * @param time  The current time
     * @param isEnergyCalculationStep  Whether the current step is an energy calculation step
     * @param isFreeEnergyCalculationStep  Whether the current step is a free energy calculation step
     */
    void doStep(Time time, bool isEnergyCalculationStep, bool isFreeEnergyCalculationStep);

    /*! \brief Write to energy trajectory
     *
     * This is only called by master - writes energy to trajectory and to log.
     */
    void write(gmx_mdoutf* outf, Step step, Time time, bool writeTrajectory, bool writeLog);

    //! ICheckpointHelperClient implementation
    void writeCheckpoint(t_state* localState, t_state* globalState) override;

    /*
     * Data owned by EnergyElement
     */
    //! The energy output object
    std::unique_ptr<EnergyOutput> energyOutput_;

    //! Whether this is the master rank
    const bool isMasterRank_;
    //! The next communicated energy writing step
    Step energyWritingStep_;
    //! The next communicated energy calculation step
    Step energyCalculationStep_;
    //! The next communicated free energy calculation step
    Step freeEnergyCalculationStep_;

    //! The force virial tensor
    tensor forceVirial_;
    //! The constraint virial tensor
    tensor shakeVirial_;
    //! The total virial tensor
    tensor totalVirial_;
    //! The pressure tensor
    tensor pressure_;
    //! The total dipole moment
    rvec muTot_;

    //! The step number of the current force virial tensor
    Step forceVirialStep_;
    //! The step number of the current constraint virial tensor
    Step shakeVirialStep_;
    //! The step number of the current total virial tensor
    Step totalVirialStep_;
    //! The step number of the current pressure tensor
    Step pressureStep_;

    //! Whether ekinh_old needs to be summed up (set by compute globals)
    bool needToSumEkinhOld_;

    //! Describes how the simulation (re)starts
    const StartingBehavior startingBehavior_;

    //! Legacy state object used to communicate with energy output
    t_state dummyLegacyState_;

    /*
     * Pointers to Simulator data
     */
    //! Pointer to the state propagator data
    StatePropagatorData* statePropagatorData_;
    //! Pointer to the free energy perturbation element
    FreeEnergyPerturbationElement* freeEnergyPerturbationElement_;
    //! Pointer to the vrescale thermostat
    const VRescaleThermostat* vRescaleThermostat_;
    //! Pointer to the Parrinello-Rahman barostat
    const ParrinelloRahmanBarostat* parrinelloRahmanBarostat_;
    //! Contains user input mdp options.
    const t_inputrec* inputrec_;
    //! Full system topology.
    const gmx_mtop_t* top_global_;
    //! Atom parameters for this domain.
    const MDAtoms* mdAtoms_;
    //! Energy data structure
    gmx_enerdata_t* enerd_;
    //! Kinetic energy data
    gmx_ekindata_t* ekind_;
    //! Handles constraints.
    const Constraints* constr_;
    //! Handles logging.
    FILE* fplog_;
    //! Helper struct for force calculations.
    t_fcdata* fcd_;
    //! Notification to MD modules
    const MdModulesNotifier& mdModulesNotifier_;
    //! Global topology groups
    const SimulationGroups* groups_;
    //! History of simulation observables.
    ObservablesHistory* observablesHistory_;
};

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Builder for the energy element
 */
class EnergyElementBuilder
{
public:
    //! Constructor, forwarding arguments to EnergyElement constructor
    template<typename... Args>
    explicit EnergyElementBuilder(Args&&... args);

    //! Set pointer to StatePropagatorData valid throughout the simulation (required)
    void setStatePropagatorData(StatePropagatorData* statePropagatorData);
    //! Set pointer to FreeEnergyPerturbationElement valid throughout the simulation (optional)
    void setFreeEnergyPerturbationElement(FreeEnergyPerturbationElement* freeEnergyPerturbationElement);
    //! Set pointer to TopologyHolder (required)
    void setTopologyHolder(TopologyHolder* topologyHolder);

    //! Register element with EnergySignaller (required)
    void registerWithEnergySignaller(SignallerBuilder<EnergySignaller>* signallerBuilder);
    //! Register element with TrajectoryElement (required)
    void registerWithTrajectoryElement(TrajectoryElementBuilder* trajectoryElementBuilder);

    //! Register element with CheckpointHelper (required)
    void registerWithCheckpointHelper(CheckpointHelperBuilder* checkpointHelperBuilder);

    /*! \brief Set v-rescale thermostat
     *
     * This allows to set a pointer to the vrescale thermostat used to
     * print the thermostat integral.
     * TODO: This should be made obsolete by a more modular energy element
     */
    void setVRescaleThermostat(const VRescaleThermostat* vRescaleThermostat);

    /*! \brief Set Parrinello-Rahman barostat
     *
     * This allows to set a pointer to the Parrinello-Rahman barostat used to
     * print the box velocities.
     * TODO: This should be made obsolete by a more modular energy element
     */
    void setParrinelloRahmanBarostat(const ParrinelloRahmanBarostat* parrinelloRahmanBarostat);

    //! Get (non-owning) pointer before element is built
    EnergyElement* getPointer();

    //! Return EnergyElement
    std::unique_ptr<EnergyElement> build();

    //! Destructor, make sure we didn't connect an element which won't exist anymore
    ~EnergyElementBuilder();

private:
    //! The element to be built
    std::unique_ptr<EnergyElement> energyElement_ = nullptr;
    //! Whether we have registered the element with the energy signaller
    bool registeredWithEnergySignaller_ = false;
    //! Whether we have registered the element with the neighbor search signaller
    bool registeredWithTrajectoryElement_ = false;
    //! Whether we have registered the element with the checkpoint helper
    bool registeredWithCheckpointHelper_ = false;
};

template<typename... Args>
EnergyElementBuilder::EnergyElementBuilder(Args&&... args)
{
    // NOLINTNEXTLINE(modernize-make-unique): make_unique does not work with private constructor
    energyElement_ = std::unique_ptr<EnergyElement>(new EnergyElement(std::forward<Args>(args)...));
}

} // namespace gmx

#endif // GMX_MODULARSIMULATOR_ENERGYELEMENT_H
