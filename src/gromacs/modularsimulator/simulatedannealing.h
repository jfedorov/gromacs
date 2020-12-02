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
 * \brief Declares the simulated annealing element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 *
 * This header is only used within the modular simulator module
 */

#ifndef GMX_MODULARSIMULATOR_SIMULATEDANNEALING_H
#define GMX_MODULARSIMULATOR_SIMULATEDANNEALING_H

#include "modularsimulatorinterfaces.h"

struct gmx_mtop_t;

namespace gmx
{
class EnergyData;
class FreeEnergyPerturbationData;
class GlobalCommunicationHelper;
class LegacySimulatorData;
class ModularSimulatorAlgorithmBuilderHelper;
class StatePropagatorData;

/*! \internal
 * \brief Element changing temperatures for simulated annealing
 */
class SimulatedAnnealingElement final : public ISimulatorElement, public ILoggingSignallerClient
{
public:
    //! Constructor
    SimulatedAnnealingElement(FILE*                        fplog,
                              const t_inputrec*            inputrec,
                              ReferenceTemperatureCallback setReferenceTemperature,
                              bool                         isMasterRank,
                              const gmx_mtop_t*            globalTopology);
    //! Update annealing temperature
    void scheduleTask(Step step, Time time, const RegisterRunFunction& registerRunFunction) override;
    //! Set initial annealing temperature
    void elementSetup() override;
    //! No teardown needed
    void elementTeardown() override{};
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
    //! Set the new annealing temperatures
    void updateAnnealingTemperature(Time time);
    //! Callback to set new reference temperature (simulated tempering only)
    ReferenceTemperatureCallback setReferenceTemperature_;

    //! ILoggingSignallerClient implementation
    std::optional<SignallerCallback> registerLoggingCallback() override;
    //! The next logging step
    Step nextLogWritingStep_;

    //! Whether this runs on master
    const bool isMasterRank_;
    //! The number of temperature groups
    const int numTemperatureGroups_;
    //! Handles logging.
    FILE* fplog_;
    //! Contains user input mdp options.
    const t_inputrec* inputrec_;
    //! Full system topology.
    const gmx_mtop_t* globalTopology_;
};

} // namespace gmx

#endif // GMX_MODULARSIMULATOR_SIMULATEDANNEALING_H
