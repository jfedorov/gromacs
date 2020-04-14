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
/*! \internal \file
 * \brief Defines the free energy perturbation element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "freeenergyperturbationelement.h"

#include "gromacs/domdec/domdec_network.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdtypes/checkpointdata.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/state.h"

#include "builders.h"
#include "checkpointhelper.h"

namespace gmx
{

FreeEnergyPerturbationElement::FreeEnergyPerturbationElement(FILE*             fplog,
                                                             const t_inputrec* inputrec,
                                                             MDAtoms*          mdAtoms) :
    lambda_(),
    lambda0_(),
    currentFEPState_(0),
    lambdasChange_(inputrec->fepvals->delta_lambda != 0),
    fplog_(fplog),
    inputrec_(inputrec),
    mdAtoms_(mdAtoms)
{
    if (inputrec->efep == efepNO)
    {
        GMX_THROW(ElementNotNeededException(
                "FreeEnergyPerturbationElement is not needed without FEP."));
    }
    lambda_.fill(0);
    lambda0_.fill(0);
    // The legacy implementation only filled the lambda vector in state_global, which is only available on master. We have the lambda vector available everywhere, so we pass a `true` for isMaster on all ranks.
    initialize_lambdas(fplog_, *inputrec_, true, &currentFEPState_, lambda_, lambda0_.data());
    update_mdatoms(mdAtoms_->mdatoms(), lambda_[efptMASS]);
}

void FreeEnergyPerturbationElement::scheduleTask(Step step,
                                                 Time gmx_unused               time,
                                                 const RegisterRunFunctionPtr& registerRunFunction)
{
    if (lambdasChange_)
    {
        (*registerRunFunction)(
                std::make_unique<SimulatorRunFunction>([this, step]() { updateLambdas(step); }));
    }
}

void FreeEnergyPerturbationElement::updateLambdas(Step step)
{
    // at beginning of step (if lambdas change...)
    setCurrentLambdasLocal(step, inputrec_->fepvals, lambda0_.data(), lambda_, currentFEPState_);
    update_mdatoms(mdAtoms_->mdatoms(), lambda_[efptMASS]);
}

ArrayRef<real> FreeEnergyPerturbationElement::lambdaView()
{
    return lambda_;
}

ArrayRef<const real> FreeEnergyPerturbationElement::constLambdaView()
{
    return lambda_;
}

int FreeEnergyPerturbationElement::currentFEPState()
{
    return currentFEPState_;
}

template<CheckpointDataOperation operation>
void FreeEnergyPerturbationElement::doCheckpointData(CheckpointData* checkpointData, const t_commrec* cr)
{
    if (MASTER(cr))
    {
        checkpointData->scalar<operation>("current FEP state", &currentFEPState_);
        checkpointData->arrayRef<operation>("lambda vector", makeCheckpointArrayRef<operation>(lambda_));
    }
    if (operation == CheckpointDataOperation::Read)
    {
        if (DOMAINDECOMP(cr))
        {
            dd_bcast(cr->dd, sizeof(int), &currentFEPState_);
            dd_bcast(cr->dd, lambda_.size() * sizeof(real), lambda_.data());
        }
        update_mdatoms(mdAtoms_->mdatoms(), lambda_[efptMASS]);
    }
}

void FreeEnergyPerturbationElement::writeCheckpoint(CheckpointData checkpointData, const t_commrec* cr)
{
    doCheckpointData<CheckpointDataOperation::Write>(&checkpointData, cr);
}

void FreeEnergyPerturbationElement::readCheckpoint(CheckpointData checkpointData, const t_commrec* cr)
{
    doCheckpointData<CheckpointDataOperation::Read>(&checkpointData, cr);
}

void FreeEnergyPerturbationElementBuilder::connectWithBuilders(ElementAndSignallerBuilders* builders)
{
    registerWithCheckpointHelper(builders->checkpointHelper.get());
}

void FreeEnergyPerturbationElementBuilder::registerWithCheckpointHelper(CheckpointHelperBuilder* checkpointHelperBuilder)
{
    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Tried to register to CheckpointHelper without available "
                       "FreeEnergyPerturbationElement.");
    if (element_)
    {
        checkpointHelperBuilder->registerClient(compat::make_not_null(element_.get()), element_->identifier);
    }
    registeredWithCheckpointHelper_ = true;
}

FreeEnergyPerturbationElement* FreeEnergyPerturbationElementBuilder::getPointer()
{
    GMX_RELEASE_ASSERT(
            registrationPossible_,
            "Called getPointer() without available FreeEnergyPerturbationElement object.");
    return element_.get();
}

std::unique_ptr<FreeEnergyPerturbationElement> FreeEnergyPerturbationElementBuilder::build()
{
    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Called build() without available FreeEnergyPerturbationElement.");
    if (element_)
    {
        GMX_RELEASE_ASSERT(registeredWithCheckpointHelper_,
                           "Tried to build FreeEnergyPerturbationElement before registering with "
                           "CheckpointHelper.");
    }
    registrationPossible_ = false;
    return std::move(element_);
}

FreeEnergyPerturbationElementBuilder::~FreeEnergyPerturbationElementBuilder()
{
    // If the element was built, but not consumed, we risk having dangling pointers
    GMX_ASSERT(!element_, "FreeEnergyPerturbationElement was constructed, but not used.");
}

} // namespace gmx
