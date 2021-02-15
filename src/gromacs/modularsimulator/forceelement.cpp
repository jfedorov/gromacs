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
 * \brief Defines the force element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include <gromacs/listed_forces/disre.h>
#include <gromacs/listed_forces/orires.h>
#include "gmxpre.h"

#include "forceelement.h"

#include "gromacs/domdec/mdsetup.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdrun/shellfc.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/forcebuffers.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/mdrunoptions.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/mtop_util.h"

#include "awhelement.h"
#include "energydata.h"
#include "freeenergyperturbationdata.h"
#include "modularsimulator.h"
#include "simulatoralgorithm.h"
#include "statepropagatordata.h"

struct gmx_edsam;
struct gmx_enfrot;
struct gmx_multisim_t;
class history_t;

namespace gmx
{
bool ForceElement::doShellFC(LegacySimulatorData* legacySimulatorData)
{
    const int numFlexibleConstraints =
            legacySimulatorData->constr ? legacySimulatorData->constr->numFlexibleConstraints() : 0;
    const int numShellParticles = gmx_mtop_particletype_count(*legacySimulatorData->top_global)[eptShell];
    return (numFlexibleConstraints != 0 || numShellParticles != 0);
}

ForceElement::ForceElement(StatePropagatorData*        statePropagatorData,
                           EnergyData*                 energyData,
                           FreeEnergyPerturbationData* freeEnergyPerturbationData,
                           bool                        isVerbose,
                           bool                        isDynamicBox,
                           FILE*                       fplog,
                           const t_commrec*            cr,
                           const t_inputrec*           inputrec,
                           const MDAtoms*              mdAtoms,
                           t_nrnb*                     nrnb,
                           t_forcerec*                 fr,
                           gmx_wallcycle*              wcycle,
                           MdrunScheduleWorkload*      runScheduleWork,
                           VirtualSitesHandler*        vsite,
                           ImdSession*                 imdSession,
                           pull_t*                     pull_work,
                           Constraints*                constr,
                           const gmx_mtop_t*           globalTopology,
                           gmx_enfrot*                 enforcedRotation,
                           Awh*                        awh,
                           const gmx_multisim_t*       multisim,
                           std::optional<history_t*>   restrainingHistory) :
    shellfc_(init_shell_flexcon(fplog,
                                globalTopology,
                                constr ? constr->numFlexibleConstraints() : 0,
                                inputrec->nstcalcenergy,
                                DOMAINDECOMP(cr),
                                runScheduleWork->simulationWork.useGpuPme)),
    doShellFC_(shellfc_ != nullptr),
    nextNSStep_(-1),
    nextEnergyCalculationStep_(-1),
    nextVirialCalculationStep_(-1),
    nextFreeEnergyCalculationStep_(-1),
    statePropagatorData_(statePropagatorData),
    energyData_(energyData),
    freeEnergyPerturbationData_(freeEnergyPerturbationData),
    awh_(awh),
    localTopology_(nullptr),
    isDynamicBox_(isDynamicBox),
    isVerbose_(isVerbose),
    nShellRelaxationSteps_(0),
    haveNmrDistanceRestraints_(fr->fcdata->disres->nres > 0),
    haveNmrOrientationRestraints_(fr->fcdata->orires->nr > 0),
    ddBalanceRegionHandler_(cr),
    lambda_(),
    fplog_(fplog),
    cr_(cr),
    multisim_(multisim),
    inputrec_(inputrec),
    mdAtoms_(mdAtoms),
    nrnb_(nrnb),
    wcycle_(wcycle),
    fr_(fr),
    vsite_(vsite),
    imdSession_(imdSession),
    pull_work_(pull_work),
    runScheduleWork_(runScheduleWork),
    constr_(constr),
    enforcedRotation_(enforcedRotation),
    restrainingHistory_(restrainingHistory.value_or(nullptr))
{
    lambda_.fill(0);

    if (doShellFC_ && !DOMAINDECOMP(cr))
    {
        // This was done in mdAlgorithmsSetupAtomData(), but shellfc
        // won't be available outside this element.
        make_local_shells(cr, mdAtoms->mdatoms(), shellfc_);
    }

    // This is checked earlier in init_disres and init_orires, but
    // adding this here since this file would need to be adapted if this changes
    GMX_RELEASE_ASSERT(!PAR(cr) || !(haveNmrDistanceRestraints_ || haveNmrOrientationRestraints_),
                       "NMR restraints are not implemented with MPI");
    GMX_RELEASE_ASSERT(!(MASTER(cr) && restrainingHistory_ == nullptr),
                       "Restraining history is expected to be non-null on master rank.");
}

void ForceElement::scheduleTask(Step step, Time time, const RegisterRunFunction& registerRunFunction)
{
    unsigned int flags =
            (GMX_FORCE_STATECHANGED | GMX_FORCE_ALLFORCES | (isDynamicBox_ ? GMX_FORCE_DYNAMICBOX : 0)
             | (nextVirialCalculationStep_ == step ? GMX_FORCE_VIRIAL : 0)
             | (nextEnergyCalculationStep_ == step ? GMX_FORCE_ENERGY : 0)
             | (nextFreeEnergyCalculationStep_ == step ? GMX_FORCE_DHDL : 0)
             | (!doShellFC_ && nextNSStep_ == step ? GMX_FORCE_NS : 0));

    if (doShellFC_)
    {
        registerRunFunction([this, step, time, flags]() { run<true>(step, time, flags); });
    }
    else
    {
        registerRunFunction([this, step, time, flags]() { run<false>(step, time, flags); });
    }
}

void ForceElement::elementSetup()
{
    GMX_ASSERT(localTopology_, "Setup called before local topology was set.");
}

template<bool executeShellFC>
void ForceElement::run(Step step, Time time, unsigned int flags)
{
    if (vsite_ != nullptr)
    {
        statePropagatorData_->ensureVirtualSitesAreValid(VSiteOperation::Positions);
    }

    if (!DOMAINDECOMP(cr_) && (flags & GMX_FORCE_NS) && inputrecDynamicBox(inputrec_))
    {
        // TODO: Correcting the box is done in DomDecHelper (if using DD) or here (non-DD simulations).
        //       Think about unifying this responsibility, could this be done in one place?
        auto box = statePropagatorData_->box();
        correct_box(fplog_, step, box);
    }

    /* The coordinates (x) are shifted (to get whole molecules)
     * in do_force.
     * This is parallelized as well, and does communication too.
     * Check comments in sim_util.c
     */
    auto  x      = statePropagatorData_->positionsView();
    auto& forces = statePropagatorData_->forcesView();
    auto  box    = statePropagatorData_->constBox();

    tensor force_vir = { { 0 } };
    // TODO: Make lambda const (needs some adjustments in lower force routines)
    ArrayRef<real> lambda =
            freeEnergyPerturbationData_ ? freeEnergyPerturbationData_->lambdaView() : lambda_;

    if (executeShellFC)
    {
        auto v = statePropagatorData_->velocitiesView();

        relax_shell_flexcon(fplog_,
                            cr_,
                            multisim_,
                            isVerbose_,
                            enforcedRotation_,
                            step,
                            inputrec_,
                            imdSession_,
                            pull_work_,
                            step == nextNSStep_,
                            static_cast<int>(flags),
                            localTopology_,
                            constr_,
                            energyData_->enerdata(),
                            statePropagatorData_->localNumAtoms(),
                            x,
                            v,
                            box,
                            lambda,
                            restrainingHistory_,
                            &forces,
                            force_vir,
                            mdAtoms_->mdatoms(),
                            nrnb_,
                            wcycle_,
                            shellfc_,
                            fr_,
                            runScheduleWork_,
                            time,
                            energyData_->muTot(),
                            vsite_,
                            ddBalanceRegionHandler_);
        nShellRelaxationSteps_++;
    }
    else
    {
        // Disabled functionality
        gmx_edsam* ed = nullptr;

        do_force(fplog_,
                 cr_,
                 multisim_,
                 inputrec_,
                 awh_,
                 enforcedRotation_,
                 imdSession_,
                 pull_work_,
                 step,
                 nrnb_,
                 wcycle_,
                 localTopology_,
                 box,
                 x,
                 restrainingHistory_,
                 &forces,
                 force_vir,
                 mdAtoms_->mdatoms(),
                 energyData_->enerdata(),
                 lambda,
                 fr_,
                 runScheduleWork_,
                 vsite_,
                 energyData_->muTot(),
                 time,
                 ed,
                 static_cast<int>(flags),
                 ddBalanceRegionHandler_);
    }
    energyData_->addToForceVirial(force_vir, step);
    if (haveNmrDistanceRestraints_)
    {
        GMX_ASSERT(restrainingHistory_ != nullptr,
                   "Distance restraining needs valid restraining history");
        update_disres_history(*fr_->fcdata->disres, restrainingHistory_);
    }
    if (haveNmrOrientationRestraints_)
    {
        GMX_ASSERT(restrainingHistory_ != nullptr,
                   "Orientation restraining needs valid restraining history");
        update_orires_history(*fr_->fcdata->orires, restrainingHistory_);
    }
}

void ForceElement::elementTeardown()
{
    if (doShellFC_)
    {
        done_shellfc(fplog_, shellfc_, nShellRelaxationSteps_);
    }
}

void ForceElement::setTopology(const gmx_localtop_t* top)
{
    localTopology_ = top;
}

std::optional<SignallerCallback> ForceElement::registerNSCallback()
{
    return [this](Step step, Time gmx_unused time) { this->nextNSStep_ = step; };
}

std::optional<SignallerCallback> ForceElement::registerEnergyCallback(EnergySignallerEvent event)
{
    if (event == EnergySignallerEvent::EnergyCalculationStep)
    {
        return [this](Step step, Time /*unused*/) { nextEnergyCalculationStep_ = step; };
    }
    if (event == EnergySignallerEvent::VirialCalculationStep)
    {
        return [this](Step step, Time /*unused*/) { nextVirialCalculationStep_ = step; };
    }
    if (event == EnergySignallerEvent::FreeEnergyCalculationStep)
    {
        return [this](Step step, Time /*unused*/) { nextFreeEnergyCalculationStep_ = step; };
    }
    return std::nullopt;
}

namespace
{
/*!
 * \brief Enum describing the contents EnergyData::Element writes to modular checkpoint
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
void ForceElement::doCheckpoint(CheckpointData<operation>* checkpointData)
{
    checkpointVersion(checkpointData, "ForceElement version", c_currentVersion);
    restrainingHistory_->doCheckpoint<operation>(
            checkpointData->subCheckpointData("restraining history"));
}

void ForceElement::saveCheckpointState(std::optional<WriteCheckpointData> checkpointData,
                                       const t_commrec*                   cr)
{
    // This is currently only checkpointing the restraining history,
    // which only exists on master, so no communication needed
    if (MASTER(cr))
    {
        doCheckpoint<CheckpointDataOperation::Write>(&checkpointData.value());
    }
}

void ForceElement::restoreCheckpointState(std::optional<ReadCheckpointData> checkpointData,
                                          const t_commrec*                  cr)
{
    // This is currently only checkpointing the restraining history,
    // which only exists on master, so no communication needed
    if (MASTER(cr))
    {
        if (!checkpointData.has_value())
        {
            // We're reading a modular checkpoint written before NMR restraints
            // were introduced in modular simulator. If the current simulation
            // doesn't use NMR restraints, we can proceed, otherwise we need to abort.
            if (!(haveNmrDistanceRestraints_ || haveNmrOrientationRestraints_))
            {
                return;
            }
            throw SimulationAlgorithmSetupError(
                    "The current modular checkpoint file does not contain NMR restraint "
                    "information. It was likely created with a prior version of GROMACS. "
                    "Try using the original GROMACS version or start a new simulation.");
        }
        doCheckpoint<CheckpointDataOperation::Read>(&checkpointData.value());
    }
}

const std::string& ForceElement::clientID()
{
    return identifier_;
}

ISimulatorElement*
ForceElement::getElementPointerImpl(LegacySimulatorData*                    legacySimulatorData,
                                    ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                    StatePropagatorData*                    statePropagatorData,
                                    EnergyData*                             energyData,
                                    FreeEnergyPerturbationData* freeEnergyPerturbationData,
                                    GlobalCommunicationHelper gmx_unused* globalCommunicationHelper)
{
    const bool isVerbose    = legacySimulatorData->mdrunOptions.verbose;
    const bool isDynamicBox = inputrecDynamicBox(legacySimulatorData->inputrec);
    return builderHelper->storeElement(std::make_unique<ForceElement>(
            statePropagatorData,
            energyData,
            freeEnergyPerturbationData,
            isVerbose,
            isDynamicBox,
            legacySimulatorData->fplog,
            legacySimulatorData->cr,
            legacySimulatorData->inputrec,
            legacySimulatorData->mdAtoms,
            legacySimulatorData->nrnb,
            legacySimulatorData->fr,
            legacySimulatorData->wcycle,
            legacySimulatorData->runScheduleWork,
            legacySimulatorData->vsite,
            legacySimulatorData->imdSession,
            legacySimulatorData->pull_work,
            legacySimulatorData->constr,
            legacySimulatorData->top_global,
            legacySimulatorData->enforcedRotation,
            AwhElement::getAwhObject(legacySimulatorData, builderHelper),
            legacySimulatorData->ms,
            MASTER(legacySimulatorData->cr) ? std::make_optional(&legacySimulatorData->state_global->hist)
                                            : std::nullopt));
}
} // namespace gmx
