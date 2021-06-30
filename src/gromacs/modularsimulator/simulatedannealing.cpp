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
 * \brief Defines the simulated annealing element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "gromacs/mdlib/coupling.h"
#include "gromacs/mdlib/energyoutput.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/topology/topology.h"

#include "referencetemperaturemanager.h"
#include "simulatedannealing.h"
#include "simulatoralgorithm.h"

namespace gmx
{

SimulatedAnnealingElement::SimulatedAnnealingElement(FILE*                        fplog,
                                                     const t_inputrec&            inputrec,
                                                     ReferenceTemperatureCallback setReferenceTemperature,
                                                     bool                         isMasterRank,
                                                     const gmx_mtop_t&            globalTopology) :
    setReferenceTemperature_(std::move(setReferenceTemperature)),
    isMasterRank_(isMasterRank),
    numTemperatureGroups_(inputrec.opts.ngtc),
    fplog_(fplog),
    inputrec_(inputrec),
    globalTopology_(globalTopology)
{
}

void SimulatedAnnealingElement::updateAnnealingTemperature(Time time)
{
    std::vector<real> temperatures(numTemperatureGroups_);
    for (int temperatureGroup = 0; temperatureGroup < numTemperatureGroups_; ++temperatureGroup)
    {
        temperatures[temperatureGroup] =
                computeAnnealingTargetTemperature(inputrec_, temperatureGroup, time);
    }
    setReferenceTemperature_(temperatures, ReferenceTemperatureChangeAlgorithm::SimulatedAnnealing);
}

void SimulatedAnnealingElement::scheduleTask(Step step, Time time, const RegisterRunFunction& registerRunFunction)
{
    registerRunFunction([this, time]() { updateAnnealingTemperature(time); });
    const bool doLog = (isMasterRank_ && step == nextLogWritingStep_ && (fplog_ != nullptr));
    if (doLog)
    {
        registerRunFunction([this]() {
            EnergyOutput::printAnnealingTemperatures(fplog_, &globalTopology_.groups, &(inputrec_.opts));
        });
    }
}

void SimulatedAnnealingElement::elementSetup()
{
    updateAnnealingTemperature(inputrec_.init_t);
}

std::optional<SignallerCallback> SimulatedAnnealingElement::registerLoggingCallback()
{
    if (isMasterRank_)
    {
        return [this](Step step, Time /*unused*/) { nextLogWritingStep_ = step; };
    }
    else
    {
        return std::nullopt;
    }
}

ISimulatorElement* SimulatedAnnealingElement::getElementPointerImpl(
        LegacySimulatorData*                    legacySimulatorData,
        ModularSimulatorAlgorithmBuilderHelper* builderHelper,
        StatePropagatorData gmx_unused* statePropagatorData,
        EnergyData gmx_unused*     energyData,
        FreeEnergyPerturbationData gmx_unused* freeEnergyPerturbationData,
        GlobalCommunicationHelper gmx_unused* globalCommunicationHelper)
{
    auto setReferenceTemperature = builderHelper->changeReferenceTemperatureCallback();
    return builderHelper->storeElement(
            std::make_unique<SimulatedAnnealingElement>(legacySimulatorData->fplog,
                                                        *legacySimulatorData->inputrec,
                                                        setReferenceTemperature,
                                                        MASTER(legacySimulatorData->cr),
                                                        legacySimulatorData->top_global));
}

} // namespace gmx
