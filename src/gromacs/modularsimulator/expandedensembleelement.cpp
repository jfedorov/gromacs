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
 * \brief Defines the expanded ensemble element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "expandedensembleelement.h"

#include "gromacs/domdec/distribute.h"
#include "gromacs/mdlib/expanded.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/mdtypes/checkpointdata.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/df_history.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/utility/fatalerror.h"

#include "energydata.h"
#include "simulatoralgorithm.h"

namespace gmx
{

void ExpandedEnsembleElement::apply(Step step, bool doLambdaStep, bool doLog)
{
    if (doLambdaStep)
    {
        const int newFepState =
                expandedEnsembleUpdateLambdaState(fplog_,
                                                  inputrec_,
                                                  energyData_->enerdata(),
                                                  freeEnergyPerturbationData_->currentFEPState(),
                                                  dfhist_.get(),
                                                  step);
        // If we're doing simulated tempering, we need to adjust the temperatures when changing lambda state
        if (doSimulatedTempering_ && (newFepState != freeEnergyPerturbationData_->currentFEPState()))
        {
            std::vector<real> newTemperatures(inputrec_->opts.ngtc,
                                              inputrec_->simtempvals->temperatures[newFepState]);

            auto referenceTemperatures =
                    constArrayRefFromArray(inputrec_->opts.ref_t, inputrec_->opts.ngtc);
            if (std::any_of(referenceTemperatures.begin(),
                            referenceTemperatures.end(),
                            [](const real& temperature) { return temperature <= 0; }))
            {
                for (int temperatureGroup = 0; temperatureGroup < inputrec_->opts.ngtc; ++temperatureGroup)
                {
                    if (referenceTemperatures[temperatureGroup] <= 0)
                    {
                        newTemperatures[temperatureGroup] = 0;
                    }
                }
            }
            setReferenceTemperature_(newTemperatures,
                                     ReferenceTemperatureChangeAlgorithm::SimulatedTempering);
        }
        // Set new state at next step
        setFepState_(newFepState, step + 1);
    }
    if (doLog)
    {
        /* only needed if doing expanded ensemble */
        PrintFreeEnergyInfoToFile(fplog_,
                                  inputrec_->fepvals,
                                  inputrec_->expandedvals,
                                  inputrec_->bSimTemp ? inputrec_->simtempvals : nullptr,
                                  dfhist_.get(),
                                  freeEnergyPerturbationData_->currentFEPState(),
                                  inputrec_->nstlog,
                                  step);
    }
}

void ExpandedEnsembleElement::elementSetup()
{
    // Check nstexpanded here, because the grompp check was broken (#2714)
    if (inputrec_->expandedvals->nstexpanded % inputrec_->nstcalcenergy != 0)
    {
        gmx_fatal(FARGS,
                  "With expanded ensemble, nstexpanded should be a multiple of nstcalcenergy");
    }
    init_expanded_ensemble(restoredFromCheckpoint_, inputrec_, dfhist_.get());
}

void ExpandedEnsembleElement::scheduleTask(Step step, Time /*unused*/, const RegisterRunFunction& registerRunFunction)
{
    const bool isFirstStep  = (step == initialStep_);
    const bool doLambdaStep = (do_per_step(step, frequency_) && !isFirstStep);
    const bool doLog        = (isMasterRank_ && step == nextLogWritingStep_ && (fplog_ != nullptr));

    if (doLambdaStep || doLog)
    {
        registerRunFunction([this, step, doLambdaStep, doLog]() { apply(step, doLambdaStep, doLog); });
    }
    if (doLambdaStep)
    {
        // We'll compute a new lambda state and want it applied for next step
        signalFepStateSetting_(step + 1);
    }
}

namespace
{
/*!
 * \brief Enum describing the contents FreeEnergyPerturbationData::Element writes to modular checkpoint
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
void ExpandedEnsembleElement::doCheckpointData(CheckpointData<operation>* checkpointData)
{
    checkpointVersion(checkpointData, "ExpandedEnsembleElement version", c_currentVersion);

    dfhist_->doCheckpoint<operation>(checkpointData->subCheckpointData("dfhist"),
                                     inputrec_->expandedvals->elamstats);
}

void ExpandedEnsembleElement::saveCheckpointState(std::optional<WriteCheckpointData> checkpointData,
                                                  const t_commrec*                   cr)
{
    if (MASTER(cr))
    {
        doCheckpointData<CheckpointDataOperation::Write>(&checkpointData.value());
    }
}

void ExpandedEnsembleElement::restoreCheckpointState(std::optional<ReadCheckpointData> checkpointData,
                                                     const t_commrec*                  cr)
{
    if (MASTER(cr))
    {
        doCheckpointData<CheckpointDataOperation::Read>(&checkpointData.value());
    }
    if (DOMAINDECOMP(cr))
    {
        dd_distribute_dfhist(cr->dd, dfhist_.get());
    }
    restoredFromCheckpoint_ = true;
}

const std::string& ExpandedEnsembleElement::clientID()
{
    return identifier_;
}

std::optional<SignallerCallback> ExpandedEnsembleElement::registerLoggingCallback()
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

ExpandedEnsembleElement::ExpandedEnsembleElement(bool              doSimulatedTempering,
                                                 bool              isMasterRank,
                                                 Step              initialStep,
                                                 int               frequency,
                                                 const EnergyData* energyData,
                                                 const FreeEnergyPerturbationData* freeEnergyPerturbationData,
                                                 FILE*                             fplog,
                                                 const t_inputrec*                 inputrec,
                                                 std::optional<ReferenceTemperatureCallback> setReferenceTemperature) :
    doSimulatedTempering_(doSimulatedTempering),
    isMasterRank_(isMasterRank),
    initialStep_(initialStep),
    frequency_(frequency),
    nextLogWritingStep_(-1),
    dfhist_(std::make_unique<df_history_t>()),
    restoredFromCheckpoint_(false),
    energyData_(energyData),
    freeEnergyPerturbationData_(freeEnergyPerturbationData),
    fplog_(fplog),
    inputrec_(inputrec)
{
    GMX_RELEASE_ASSERT(!doSimulatedTempering || setReferenceTemperature.has_value(),
                       "Simulated tempering needs a callback to set new reference temperatures.");
    if (doSimulatedTempering_)
    {
        setReferenceTemperature_ = setReferenceTemperature.value();
    }
    init_df_history(dfhist_.get(), inputrec_->fepvals->n_lambda);
    std::tie(signalFepStateSetting_, setFepState_) = freeEnergyPerturbationData->fepStateCallbacks();
}

ISimulatorElement* ExpandedEnsembleElement::getElementPointerImpl(
        LegacySimulatorData*                    legacySimulatorData,
        ModularSimulatorAlgorithmBuilderHelper* builderHelper,
        StatePropagatorData* gmx_unused statePropagatorData,
        EnergyData*                     energyData,
        FreeEnergyPerturbationData*     freeEnergyPerturbationData,
        GlobalCommunicationHelper* gmx_unused globalCommunicationHelper)
{
    const bool doSimulatedTempering = legacySimulatorData->inputrec->bSimTemp;
    auto       setReferenceTemperature =
            doSimulatedTempering
                    ? std::make_optional(builderHelper->changeReferenceTemperatureCallback())
                    : std::nullopt;
    return builderHelper->storeElement(std::make_unique<ExpandedEnsembleElement>(
            doSimulatedTempering,
            MASTER(legacySimulatorData->cr),
            legacySimulatorData->inputrec->init_step,
            legacySimulatorData->inputrec->expandedvals->nstexpanded,
            energyData,
            freeEnergyPerturbationData,
            legacySimulatorData->fplog,
            legacySimulatorData->inputrec,
            setReferenceTemperature));
}

} // namespace gmx