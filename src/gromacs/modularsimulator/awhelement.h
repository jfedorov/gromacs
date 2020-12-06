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
 * \brief Declares the AWH element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 *
 * This header is only used within the modular simulator module
 */

#ifndef GMX_MODULARSIMULATOR_AWHELEMENT_H
#define GMX_MODULARSIMULATOR_AWHELEMENT_H

#include "modularsimulatorinterfaces.h"

namespace gmx
{
class Awh;
struct AwhHistory;
class EnergyData;
class FreeEnergyPerturbationData;
class GlobalCommunicationHelper;
class LegacySimulatorData;
class MDAtoms;
class ModularSimulatorAlgorithmBuilderHelper;
class StatePropagatorData;

class AwhElement : public ISimulatorElement, public ICheckpointHelperClient, public ILoggingSignallerClient
{
public:
    //! Constructor
    AwhElement(Awh* awh, FreeEnergyPerturbationData* freeEnergyPerturbationData, bool isMasterRank);
    //! Update annealing temperature
    void scheduleTask(Step step, Time time, const RegisterRunFunction& registerRunFunction) override;
    //! Set initial annealing temperature
    void elementSetup() override;
    //! No teardown needed
    void elementTeardown() override {}

    //! ICheckpointHelperClient write checkpoint implementation
    void saveCheckpointState(std::optional<WriteCheckpointData> checkpointData, const t_commrec* cr) override;
    //! ICheckpointHelperClient read checkpoint implementation
    void restoreCheckpointState(std::optional<ReadCheckpointData> checkpointData, const t_commrec* cr) override;
    //! ICheckpointHelperClient key implementation
    const std::string& clientID() override;

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

    //! Retrieve Awh instance from builder, build if non-existent
    static Awh* getAwhObject(LegacySimulatorData*                    legacySimulatorData,
                             ModularSimulatorAlgorithmBuilderHelper* builderHelper);

private:
    //! The history object for checkpointing
    std::shared_ptr<AwhHistory> awhHistory_;

    //! Callback to set a new lambda state
    SetFepState setFepState_;
    //! Callback to announce setting a new lambda state at scheduling time
    SignalFepStateSetting signalFepStateSetting_;

    // TODO: Clarify relationship to data objects and find a more robust alternative to raw pointers (#3583)
    //! The Awh object
    Awh* awh_;
    //! Pointer to the free energy perturbation data
    const FreeEnergyPerturbationData* freeEnergyPerturbationData_;

    //! Whether this is running on master rank
    const bool isMasterRank_;

    //! CheckpointHelper identifier
    const std::string identifier_ = "AwhElement";
    //! Whether this object was restored from checkpoint
    bool restoredFromCheckpoint_;

    //! ILoggingSignallerClient implementation
    std::optional<SignallerCallback> registerLoggingCallback() override;
    //! The next logging step
    Step nextLogWritingStep_;

    // Access to ISimulator data
    //! Handles logging.
    FILE* fplog_;
};

} // namespace gmx

#endif // GMX_MODULARSIMULATOR_AWHELEMENT_H
