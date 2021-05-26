/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
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
 * \brief Defines the free energy perturbation element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "freeenergyperturbationdata.h"

#include "gromacs/domdec/domdec_network.h"
#include "gromacs/mdlib/freeenergyparameters.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdtypes/checkpointdata.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/trajectory/trajectoryframe.h"

#include "modularsimulator.h"
#include "simulatoralgorithm.h"

namespace gmx
{

FreeEnergyPerturbationData::FreeEnergyPerturbationData(FILE* fplog, const t_inputrec& inputrec, MDAtoms* mdAtoms) :
    element_(std::make_unique<Element>(this, inputrec.fepvals->delta_lambda)),
    lambda_(),
    currentFEPState_(0),
    fplog_(fplog),
    inputrec_(inputrec),
    mdAtoms_(mdAtoms)
{
    std::fill(lambda_.begin(), lambda_.end(), 0);
    // The legacy implementation only filled the lambda vector in state_global, which is only
    // available on master. We have the lambda vector available everywhere, so we pass a `true`
    // for isMaster on all ranks. See #3647.
    initialize_lambdas(fplog_,
                       inputrec_.efep,
                       inputrec_.bSimTemp,
                       *inputrec_.fepvals,
                       inputrec_.simtempvals->temperatures,
                       gmx::arrayRefFromArray(inputrec_.opts.ref_t, inputrec_.opts.ngtc),
                       true,
                       &currentFEPState_,
                       lambda_);
}

void FreeEnergyPerturbationData::Element::scheduleTask(Step                       step,
                                                       Time gmx_unused            time,
                                                       const RegisterRunFunction& registerRunFunction)
{
    if (lambdasChange_)
    {
        registerRunFunction([this, step]() { freeEnergyPerturbationData_->updateLambdas(step); });
    }
}

void FreeEnergyPerturbationData::updateLambdas(Step step)
{
    // at beginning of step (if lambdas change...)
    lambda_ = currentLambdas(step, *(inputrec_.fepvals), currentFEPState_);
    updateMDAtoms();
}

ArrayRef<real> FreeEnergyPerturbationData::lambdaView()
{
    return lambda_;
}

ArrayRef<const real> FreeEnergyPerturbationData::constLambdaView()
{
    return lambda_;
}

int FreeEnergyPerturbationData::currentFEPState() const
{
    return currentFEPState_;
}

void FreeEnergyPerturbationData::updateMDAtoms()
{
    update_mdatoms(mdAtoms_->mdatoms(), lambda_[FreeEnergyPerturbationCouplingType::Mass]);
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
void FreeEnergyPerturbationData::doCheckpointData(CheckpointData<operation>* checkpointData)
{
    checkpointVersion(checkpointData, "FreeEnergyPerturbationData version", c_currentVersion);

    checkpointData->scalar("current FEP state", &currentFEPState_);
    checkpointData->arrayRef("lambda vector", makeCheckpointArrayRef<operation>(lambda_));
}

void FreeEnergyPerturbationData::Element::saveCheckpointState(std::optional<WriteCheckpointData> checkpointData,
                                                              const t_commrec*                   cr)
{
    if (MASTER(cr))
    {
        freeEnergyPerturbationData_->doCheckpointData<CheckpointDataOperation::Write>(
                &checkpointData.value());
    }
}

void FreeEnergyPerturbationData::Element::restoreCheckpointState(std::optional<ReadCheckpointData> checkpointData,
                                                                 const t_commrec* cr)
{
    if (MASTER(cr))
    {
        freeEnergyPerturbationData_->doCheckpointData<CheckpointDataOperation::Read>(
                &checkpointData.value());
    }
    if (DOMAINDECOMP(cr))
    {
        dd_bcast(cr->dd, sizeof(int), &freeEnergyPerturbationData_->currentFEPState_);
        dd_bcast(cr->dd,
                 ssize(freeEnergyPerturbationData_->lambda_) * int(sizeof(real)),
                 freeEnergyPerturbationData_->lambda_.data());
    }
}

const std::string& FreeEnergyPerturbationData::Element::clientID()
{
    return FreeEnergyPerturbationData::checkpointID();
}

DomDecCallback FreeEnergyPerturbationData::Element::registerDomDecCallback()
{
    return [this]() { freeEnergyPerturbationData_->updateMDAtoms(); };
}

FreeEnergyPerturbationData::Element::Element(FreeEnergyPerturbationData* freeEnergyPerturbationElement,
                                             double                      deltaLambda) :
    freeEnergyPerturbationData_(freeEnergyPerturbationElement), lambdasChange_(deltaLambda != 0)
{
}

void FreeEnergyPerturbationData::Element::elementSetup()
{
    freeEnergyPerturbationData_->updateMDAtoms();
}

FreeEnergyPerturbationData::Element* FreeEnergyPerturbationData::element()
{
    return element_.get();
}

ISimulatorElement* FreeEnergyPerturbationData::Element::getElementPointerImpl(
        LegacySimulatorData gmx_unused*        legacySimulatorData,
        ModularSimulatorAlgorithmBuilderHelper gmx_unused* builderHelper,
        StatePropagatorData gmx_unused* statePropagatorData,
        EnergyData gmx_unused*      energyData,
        FreeEnergyPerturbationData* freeEnergyPerturbationData,
        GlobalCommunicationHelper gmx_unused* globalCommunicationHelper)
{
    return freeEnergyPerturbationData->element();
}

void FreeEnergyPerturbationData::readCheckpointToTrxFrame(t_trxframe* trxFrame,
                                                          std::optional<ReadCheckpointData> readCheckpointData)
{
    if (readCheckpointData)
    {
        FreeEnergyPerturbationData freeEnergyPerturbationData(nullptr, t_inputrec(), nullptr);
        freeEnergyPerturbationData.doCheckpointData(&readCheckpointData.value());
        trxFrame->lambda = freeEnergyPerturbationData.lambda_[FreeEnergyPerturbationCouplingType::Fep];
        trxFrame->fep_state = freeEnergyPerturbationData.currentFEPState_;
    }
    else
    {
        trxFrame->lambda    = 0;
        trxFrame->fep_state = 0;
    }
    trxFrame->bLambda = true;
}

const std::string& FreeEnergyPerturbationData::checkpointID()
{
    static const std::string identifier = "FreeEnergyPerturbationData";
    return identifier;
}

} // namespace gmx
