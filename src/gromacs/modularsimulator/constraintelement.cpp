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
 * \brief Defines the constraint element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "constraintelement.h"

#include "gromacs/math/vec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/utility/fatalerror.h"

#include "energyelement.h"
#include "freeenergyperturbationelement.h"
#include "signallers.h"
#include "statepropagatordata.h"
#include "trajectoryelement.h"

namespace gmx
{
template<ConstraintVariable variable>
ConstraintsElement<variable>::ConstraintsElement(Constraints*      constr,
                                                 bool              isMaster,
                                                 FILE*             fplog,
                                                 const t_inputrec* inputrec,
                                                 const t_mdatoms*  mdAtoms) :
    nextVirialCalculationStep_(-1),
    nextEnergyWritingStep_(-1),
    nextLogWritingStep_(-1),
    isMasterRank_(isMaster),
    statePropagatorData_(nullptr),
    energyElement_(nullptr),
    freeEnergyPerturbationElement_(nullptr),
    constr_(constr),
    fplog_(fplog),
    inputrec_(inputrec),
    mdAtoms_(mdAtoms)
{
    if (not constr_)
    {
        GMX_THROW(ElementNotNeededException("ConstraintsElement not needed if constr == nullptr."));
    }
}

template<ConstraintVariable variable>
void ConstraintsElement<variable>::elementSetup()
{
    if (!inputrec_->bContinuation
        && ((variable == ConstraintVariable::Positions && inputrec_->eI == eiMD)
            || (variable == ConstraintVariable::Velocities && inputrec_->eI == eiVV)))
    {
        const real lambdaBonded = freeEnergyPerturbationElement_
                                          ? freeEnergyPerturbationElement_->constLambdaView()[efptBONDED]
                                          : 0;
        // Constrain the initial coordinates and velocities
        do_constrain_first(fplog_, constr_, inputrec_, mdAtoms_, statePropagatorData_->localNumAtoms(),
                           statePropagatorData_->positionsView(), statePropagatorData_->velocitiesView(),
                           statePropagatorData_->box(), lambdaBonded);

        if (isMasterRank_)
        {
            if (inputrec_->eConstrAlg == econtLINCS)
            {
                fprintf(fplog_, "RMS relative constraint deviation after constraining: %.2e\n",
                        constr_->rmsd());
            }
        }
    }
}

template<ConstraintVariable variable>
void ConstraintsElement<variable>::scheduleTask(Step step,
                                                Time gmx_unused               time,
                                                const RegisterRunFunctionPtr& registerRunFunction)
{
    bool calculateVirial = (step == nextVirialCalculationStep_);
    bool writeLog        = (step == nextLogWritingStep_);
    bool writeEnergy     = (step == nextEnergyWritingStep_);

    // register constraining
    (*registerRunFunction)(std::make_unique<SimulatorRunFunction>(
            [this, step, calculateVirial, writeLog, writeEnergy]() {
                apply(step, calculateVirial, writeLog, writeEnergy);
            }));
}

template<ConstraintVariable variable>
void ConstraintsElement<variable>::apply(Step step, bool calculateVirial, bool writeLog, bool writeEnergy)
{
    tensor vir_con;

    ArrayRefWithPadding<RVec> x;
    ArrayRefWithPadding<RVec> xprime;
    ArrayRef<RVec>            min_proj;
    ArrayRefWithPadding<RVec> v;

    const real lambdaBonded = freeEnergyPerturbationElement_
                                      ? freeEnergyPerturbationElement_->constLambdaView()[efptBONDED]
                                      : 0;
    real dvdlambda = 0;

    switch (variable)
    {
        case ConstraintVariable::Positions:
            x      = statePropagatorData_->previousPositionsView();
            xprime = statePropagatorData_->positionsView();
            v      = statePropagatorData_->velocitiesView();
            break;
        case ConstraintVariable::Velocities:
            x        = statePropagatorData_->positionsView();
            xprime   = statePropagatorData_->velocitiesView();
            min_proj = statePropagatorData_->velocitiesView().unpaddedArrayRef();
            break;
        default: gmx_fatal(FARGS, "Constraint algorithm not implemented for modular simulator.");
    }

    constr_->apply(writeLog, writeEnergy, step, 1, 1.0, x, xprime, min_proj, statePropagatorData_->box(),
                   lambdaBonded, &dvdlambda, v, calculateVirial ? &vir_con : nullptr, variable);

    if (calculateVirial)
    {
        if (inputrec_->eI == eiVV)
        {
            // For some reason, the shake virial in VV is reset twice a step.
            // Energy element will only do this once per step.
            // TODO: Investigate this
            clear_mat(energyElement_->constraintVirial(step));
        }
        energyElement_->addToConstraintVirial(vir_con, step);
    }

    /* The factor of 2 correction is necessary because half of the constraint
     * force is removed in the VV step. This factor is either exact or a very
     * good approximation, statistically insignificant in any real free energy
     * calculation. Any possible error is not a simulation propagation error,
     * but a potential reporting error in the data that goes to dh/dlambda.
     * Cf. Issue #1255
     */
    const real c_dvdlConstraintCorrectionFactor = EI_VV(inputrec_->eI) ? 2.0 : 1.0;
    energyElement_->enerdata()->term[F_DVDL_CONSTR] += c_dvdlConstraintCorrectionFactor * dvdlambda;
}

template<ConstraintVariable variable>
SignallerCallbackPtr ConstraintsElement<variable>::registerEnergyCallback(EnergySignallerEvent event)
{
    if (event == EnergySignallerEvent::VirialCalculationStep)
    {
        return std::make_unique<SignallerCallback>(
                [this](Step step, Time /*unused*/) { nextVirialCalculationStep_ = step; });
    }
    return nullptr;
}

template<ConstraintVariable variable>
SignallerCallbackPtr ConstraintsElement<variable>::registerTrajectorySignallerCallback(TrajectoryEvent event)
{
    if (event == TrajectoryEvent::EnergyWritingStep)
    {
        return std::make_unique<SignallerCallback>(
                [this](Step step, Time /*unused*/) { nextEnergyWritingStep_ = step; });
    }
    return nullptr;
}

template<ConstraintVariable variable>
SignallerCallbackPtr ConstraintsElement<variable>::registerLoggingCallback()
{
    return std::make_unique<SignallerCallback>(
            [this](Step step, Time /*unused*/) { nextLogWritingStep_ = step; });
}

ConstraintsElementBuilder::ConstraintsElementBuilder(Constraints*      constr,
                                                     bool              isMaster,
                                                     FILE*             fplog,
                                                     const t_inputrec* inputrec,
                                                     const t_mdatoms*  mdAtoms)
{
    try
    {
        // NOLINTNEXTLINE(modernize-make-unique): make_unique does not work with private constructor
        positionConstraints_ = std::unique_ptr<ConstraintsElement<ConstraintVariable::Positions>>(
                new ConstraintsElement<ConstraintVariable::Positions>(constr, isMaster, fplog,
                                                                      inputrec, mdAtoms));
    }
    catch (ElementNotNeededException&)
    {
        positionConstraints_ = nullptr;
    }
    if (inputrec->eI == eiVV)
    {
        try
        {
            // NOLINTNEXTLINE(modernize-make-unique): make_unique does not work with private constructor
            velocityConstraints_ = std::unique_ptr<ConstraintsElement<ConstraintVariable::Velocities>>(
                    new ConstraintsElement<ConstraintVariable::Velocities>(constr, isMaster, fplog,
                                                                           inputrec, mdAtoms));
        }
        catch (ElementNotNeededException&)
        {
            velocityConstraints_ = nullptr;
        }
    }
    // Element being nullptr is a valid state, nullptr (element is not built)
    // needs to be handled by the caller of build().
    registrationPossible_ = true;
}

void ConstraintsElementBuilder::setStatePropagatorData(StatePropagatorData* statePropagatorData)
{
    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Tried to set StatePropagatorData after ConstraintsElement was built.");
    if (positionConstraints_)
    {
        positionConstraints_->statePropagatorData_ = statePropagatorData;
    }
    if (velocityConstraints_)
    {
        velocityConstraints_->statePropagatorData_ = statePropagatorData;
    }
}

void ConstraintsElementBuilder::setEnergyElement(EnergyElement* energyElement)
{
    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Tried to set EnergyElement after ConstraintsElement was built.");
    if (positionConstraints_)
    {
        positionConstraints_->energyElement_ = energyElement;
    }
    if (velocityConstraints_)
    {
        velocityConstraints_->energyElement_ = energyElement;
    }
}

void ConstraintsElementBuilder::setFreeEnergyPerturbationElement(FreeEnergyPerturbationElement* freeEnergyPerturbationElement)
{
    GMX_RELEASE_ASSERT(
            registrationPossible_,
            "Tried to set FreeEnergyPerturbationElement after ConstraintsElement was built.");
    if (positionConstraints_)
    {
        positionConstraints_->freeEnergyPerturbationElement_ = freeEnergyPerturbationElement;
    }
    if (velocityConstraints_)
    {
        velocityConstraints_->freeEnergyPerturbationElement_ = freeEnergyPerturbationElement;
    }
}

void ConstraintsElementBuilder::registerWithEnergySignaller(SignallerBuilder<EnergySignaller>* signallerBuilder)
{
    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Tried to set EnergySignaller after ConstraintsElement was built.");
    if (positionConstraints_)
    {
        signallerBuilder->registerSignallerClient(compat::make_not_null(positionConstraints_.get()));
    }
    if (velocityConstraints_)
    {
        signallerBuilder->registerSignallerClient(compat::make_not_null(velocityConstraints_.get()));
    }
    registeredWithEnergySignaller_ = true;
}

void ConstraintsElementBuilder::registerWithTrajectorySignaller(TrajectoryElementBuilder* signallerBuilder)
{
    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Tried to set TrajectorySignaller after ConstraintsElement was built.");
    if (positionConstraints_)
    {
        signallerBuilder->registerSignallerClient(compat::make_not_null(positionConstraints_.get()));
    }
    if (velocityConstraints_)
    {
        signallerBuilder->registerSignallerClient(compat::make_not_null(velocityConstraints_.get()));
    }
    registeredWithTrajectorySignaller_ = true;
}

void ConstraintsElementBuilder::registerWithLoggingSignaller(SignallerBuilder<LoggingSignaller>* signallerBuilder)
{
    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Tried to set LoggingSignaller after ConstraintsElement was built.");
    if (positionConstraints_)
    {
        signallerBuilder->registerSignallerClient(compat::make_not_null(positionConstraints_.get()));
    }
    if (velocityConstraints_)
    {
        signallerBuilder->registerSignallerClient(compat::make_not_null(velocityConstraints_.get()));
    }
    registeredWithLoggingSignaller_ = true;
}

template<>
std::unique_ptr<ConstraintsElement<ConstraintVariable::Positions>> ConstraintsElementBuilder::build()
{
    if (positionConstraints_)
    {
        GMX_RELEASE_ASSERT(positionConstraints_->statePropagatorData_,
                           "Tried to build ConstraintsElement before setting StatePropagatorData.");
        GMX_RELEASE_ASSERT(positionConstraints_->energyElement_,
                           "Tried to build ConstraintsElement before setting EnergyElement.");
        GMX_RELEASE_ASSERT(
                registeredWithEnergySignaller_,
                "Tried to build ConstraintsElement before registering with EnergySignaller.");
        GMX_RELEASE_ASSERT(
                registeredWithTrajectorySignaller_,
                "Tried to build ConstraintsElement before registering with TrajectorySignaller.");
        GMX_RELEASE_ASSERT(
                registeredWithLoggingSignaller_,
                "Tried to build ConstraintsElement before registering with LoggingSignaller.");
    }
    // Not accepting any registrations anymore
    registrationPossible_ = false;
    return std::move(positionConstraints_);
}

template<>
std::unique_ptr<ConstraintsElement<ConstraintVariable::Velocities>> ConstraintsElementBuilder::build()
{
    if (velocityConstraints_)
    {
        GMX_RELEASE_ASSERT(velocityConstraints_->statePropagatorData_,
                           "Tried to build ConstraintsElement before setting StatePropagatorData.");
        GMX_RELEASE_ASSERT(velocityConstraints_->energyElement_,
                           "Tried to build ConstraintsElement before setting EnergyElement.");
        GMX_RELEASE_ASSERT(
                registeredWithEnergySignaller_,
                "Tried to build ConstraintsElement before registering with EnergySignaller.");
        GMX_RELEASE_ASSERT(
                registeredWithTrajectorySignaller_,
                "Tried to build ConstraintsElement before registering with TrajectorySignaller.");
        GMX_RELEASE_ASSERT(
                registeredWithLoggingSignaller_,
                "Tried to build ConstraintsElement before registering with LoggingSignaller.");
    }
    // Not accepting any registrations anymore
    registrationPossible_ = false;
    return std::move(velocityConstraints_);
}

ConstraintsElementBuilder::~ConstraintsElementBuilder()
{
    // If elements were built, but not consumed, we risk dangling pointers
    GMX_RELEASE_ASSERT(!positionConstraints_,
                       "Position constraint element was constructed, but not used.");
    GMX_RELEASE_ASSERT(!velocityConstraints_,
                       "Velocity constraint element was constructed, but not used.");
}

//! Explicit template initialization
//! @{
template class ConstraintsElement<ConstraintVariable::Positions>;
template class ConstraintsElement<ConstraintVariable::Velocities>;
//! @}

} // namespace gmx
