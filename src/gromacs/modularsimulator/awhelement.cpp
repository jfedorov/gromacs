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
 * \brief Defines the AWH element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gromacs/applied_forces/awh/awh.h"
#include "gromacs/commandline/filenm.h"
#include "gromacs/mdrun/isimulator.h"
#include "gromacs/mdrunutility/handlerestart.h"
#include "gromacs/mdtypes/awh_history.h"
#include "gromacs/mdtypes/inputrec.h"

#include "awhelement.h"
#include "forceelement.h"
#include "simulatoralgorithm.h"

namespace gmx
{

Awh* AwhElement::getAwhObject(LegacySimulatorData*                    legacySimulatorData,
                              ModularSimulatorAlgorithmBuilderHelper* builderHelper)
{
    if (!builderHelper->simulationData<Awh>("AWH"))
    {
        std::unique_ptr<Awh> awh = nullptr;
        if (legacySimulatorData->inputrec->bDoAwh)
        {
            if (ForceElement::doShellFC(legacySimulatorData))
            {
                GMX_THROW(InvalidInputError("AWH biasing does not support shell particles."));
            }
            awh = std::make_unique<Awh>(
                    legacySimulatorData->fplog,
                    *legacySimulatorData->inputrec,
                    legacySimulatorData->cr,
                    legacySimulatorData->ms,
                    *legacySimulatorData->inputrec->awhParams,
                    opt2fn("-awh", legacySimulatorData->nfile, legacySimulatorData->fnm),
                    legacySimulatorData->pull_work,
                    legacySimulatorData->inputrec->fepvals->n_lambda,
                    legacySimulatorData->inputrec->fepvals->init_fep_state);
        }
        builderHelper->storeSimulationData("AWH", std::move(awh));
    }
    return builderHelper->simulationData<Awh>("AWH").value();
}

void AwhElement::scheduleTask(Step step, Time /*unused*/, const RegisterRunFunction& registerRunFunction)
{
    const auto doLog = (isMasterRank_ && step == nextLogWritingStep_ && awh_->hasFepLambdaDimension());
    if (doLog)
    {
        registerRunFunction([this]() {
            printLambdaStateToLog(fplog_, freeEnergyPerturbationData_->constLambdaView(), false);
        });
    }
    if (awh_->needForeignEnergyDifferences(step))
    {
        registerRunFunction([this, step]() { setFepState_(awh_->fepLambdaState(), step + 1); });
        // We'll compute a new lambda state and want it applied for next step
        signalFepStateSetting_(step + 1);
    }
}

void AwhElement::elementSetup()
{
    if (!restoredFromCheckpoint_ && isMasterRank_)
    {
        awhHistory_ = awh_->initHistoryFromState();
    }
}

namespace
{
/*!
 * \brief Enum describing the contents all AWH history classes implemented in this file
 *        write to modular checkpoint
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

template<CheckpointDataOperation operation>
void doCheckpoint(CheckpointData<operation>* checkpointData, AwhHistory* awhHistory)
{
    checkpointVersion(checkpointData, "AwhElement version", c_currentVersion);
    awhHistory->doCheckpoint(checkpointData->subCheckpointData("AWH history"));
}
} // namespace

void AwhElement::saveCheckpointState(std::optional<WriteCheckpointData> checkpointData, const t_commrec* cr)
{
    if (MASTER(cr))
    {
        /* Note: In legacy simulator, this call needs to happen at a specific time to
         * avoid double-counting force evaluation when restarting from checkpoint.
         * As modular simulator stores checkpoint between steps, this problem doesn't
         * arise here. */
        awh_->updateHistory(awhHistory_.get());
        doCheckpoint(&checkpointData.value(), awhHistory_.get());
    }
}

void AwhElement::restoreCheckpointState(std::optional<ReadCheckpointData> checkpointData,
                                        const t_commrec*                  cr)
{
    if (MASTER(cr))
    {
        awhHistory_ = std::make_shared<AwhHistory>();
        doCheckpoint(&checkpointData.value(), awhHistory_.get());
    }
    awh_->restoreStateFromHistory(awhHistory_.get());
    restoredFromCheckpoint_ = true;
}

const std::string& AwhElement::clientID()
{
    return identifier_;
}

std::optional<SignallerCallback> AwhElement::registerLoggingCallback()
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

AwhElement::AwhElement(Awh* awh, FreeEnergyPerturbationData* freeEnergyPerturbationData, bool isMasterRank) :
    awhHistory_(nullptr),
    awh_(awh),
    freeEnergyPerturbationData_(freeEnergyPerturbationData),
    isMasterRank_(isMasterRank),
    restoredFromCheckpoint_(false)
{
    if (freeEnergyPerturbationData)
    {
        std::tie(signalFepStateSetting_, setFepState_) = freeEnergyPerturbationData->fepStateCallbacks();
    }
}

ISimulatorElement*
AwhElement::getElementPointerImpl(LegacySimulatorData*                    legacySimulatorData,
                                  ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                  StatePropagatorData* gmx_unused statePropagatorData,
                                  EnergyData* gmx_unused      energyData,
                                  FreeEnergyPerturbationData* freeEnergyPerturbationData,
                                  GlobalCommunicationHelper* gmx_unused globalCommunicationHelper)
{
    return builderHelper->storeElement(std::make_unique<AwhElement>(
            AwhElement::getAwhObject(legacySimulatorData, builderHelper),
            freeEnergyPerturbationData,
            MASTER(legacySimulatorData->cr)));
}

} // namespace gmx
