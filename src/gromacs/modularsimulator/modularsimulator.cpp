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
 * \brief Defines the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "modularsimulator.h"

#include "gromacs/commandline/filenm.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/pme_load_balancing.h"
#include "gromacs/ewald/pme_pp.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/mdlib/checkpointhandler.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/resethandler.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdrun/replicaexchange.h"
#include "gromacs/mdrun/shellfc.h"
#include "gromacs/mdrunutility/handlerestart.h"
#include "gromacs/mdrunutility/printtime.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/mdrunoptions.h"
#include "gromacs/mdtypes/observableshistory.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"

#include "builders.h"
#include "compositesimulatorelement.h"
#include "computeglobalselement.h"
#include "constraintelement.h"
#include "forceelement.h"
#include "freeenergyperturbationelement.h"
#include "parrinellorahmanbarostat.h"
#include "signallers.h"
#include "trajectoryelement.h"
#include "vrescalethermostat.h"

namespace gmx
{
void ModularSimulator::run()
{
    GMX_LOG(mdlog.info).asParagraph().appendText("Using the modular simulator.");
    constructElementsAndSignallers();
    simulatorSetup();
    for (auto& signaller : signallerCallList_)
    {
        signaller->signallerSetup();
    }
    if (domDecHelper_)
    {
        domDecHelper_->setup();
    }

    for (auto& element : elementsOwnershipList_)
    {
        element->elementSetup();
    }
    if (pmeLoadBalanceHelper_)
    {
        // State must have been initialized so pmeLoadBalanceHelper_ gets a valid box
        pmeLoadBalanceHelper_->setup();
    }

    while (step_ <= signalHelper_->lastStep_)
    {
        populateTaskQueue();

        while (!taskQueue_.empty())
        {
            auto task = std::move(taskQueue_.front());
            taskQueue_.pop();
            // run function
            (*task)();
        }
    }

    for (auto& element : elementsOwnershipList_)
    {
        element->elementTeardown();
    }
    if (pmeLoadBalanceHelper_)
    {
        pmeLoadBalanceHelper_->teardown();
    }
    simulatorTeardown();
}

void ModularSimulator::simulatorSetup()
{
    if (!mdrunOptions.writeConfout)
    {
        // This is on by default, and the main known use case for
        // turning it off is for convenience in benchmarking, which is
        // something that should not show up in the general user
        // interface.
        GMX_LOG(mdlog.info)
                .asParagraph()
                .appendText(
                        "The -noconfout functionality is deprecated, and "
                        "may be removed in a future version.");
    }

    if (MASTER(cr))
    {
        char        sbuf[STEPSTRSIZE], sbuf2[STEPSTRSIZE];
        std::string timeString;
        fprintf(stderr, "starting mdrun '%s'\n", *(top_global->name));
        if (inputrec->nsteps >= 0)
        {
            timeString = formatString("%8.1f", static_cast<double>(inputrec->init_step + inputrec->nsteps)
                                                       * inputrec->delta_t);
        }
        else
        {
            timeString = "infinite";
        }
        if (inputrec->init_step > 0)
        {
            fprintf(stderr, "%s steps, %s ps (continuing from step %s, %8.1f ps).\n",
                    gmx_step_str(inputrec->init_step + inputrec->nsteps, sbuf), timeString.c_str(),
                    gmx_step_str(inputrec->init_step, sbuf2), inputrec->init_step * inputrec->delta_t);
        }
        else
        {
            fprintf(stderr, "%s steps, %s ps.\n", gmx_step_str(inputrec->nsteps, sbuf),
                    timeString.c_str());
        }
        fprintf(fplog, "\n");
    }

    walltime_accounting_start_time(walltime_accounting);
    wallcycle_start(wcycle, ewcRUN);
    print_start(fplog, cr, walltime_accounting, "mdrun");

    step_ = inputrec->init_step;
}

void ModularSimulator::preStep(Step step, Time gmx_unused time, bool isNeighborSearchingStep)
{
    if (stopHandler_->stoppingAfterCurrentStep(isNeighborSearchingStep) && step != signalHelper_->lastStep_)
    {
        /*
         * Stop handler wants to stop after the current step, which was
         * not known when building the current task queue. This happens
         * e.g. when a stop is signalled by OS. We therefore want to purge
         * the task queue now, and re-schedule this step as last step.
         */
        // clear task queue
        std::queue<SimulatorRunFunctionPtr>().swap(taskQueue_);
        // rewind step
        step_ = step;
        return;
    }

    resetHandler_->setSignal(walltime_accounting);
    // This is a hack to avoid having to rewrite StopHandler to be a NeighborSearchSignaller
    // and accept the step as input. Eventually, we want to do that, but currently this would
    // require introducing NeighborSearchSignaller in the legacy do_md or a lot of code
    // duplication.
    stophandlerIsNSStep_    = isNeighborSearchingStep;
    stophandlerCurrentStep_ = step;
    stopHandler_->setSignal();

    wallcycle_start(wcycle, ewcSTEP);
}

void ModularSimulator::postStep(Step step, Time gmx_unused time)
{
    // Output stuff
    if (MASTER(cr))
    {
        if (do_per_step(step, inputrec->nstlog))
        {
            if (fflush(fplog) != 0)
            {
                gmx_fatal(FARGS, "Cannot flush logfile - maybe you are out of disk space?");
            }
        }
    }
    const bool do_verbose = mdrunOptions.verbose
                            && (step % mdrunOptions.verboseStepPrintInterval == 0
                                || step == inputrec->init_step || step == signalHelper_->lastStep_);
    // Print the remaining wall clock time for the run
    if (MASTER(cr) && (do_verbose || gmx_got_usr_signal())
        && !(pmeLoadBalanceHelper_ && pmeLoadBalanceHelper_->pmePrinting()))
    {
        print_time(stderr, walltime_accounting, step, inputrec, cr);
    }

    double cycles = wallcycle_stop(wcycle, ewcSTEP);
    if (DOMAINDECOMP(cr) && wcycle)
    {
        dd_cycles_add(cr->dd, static_cast<float>(cycles), ddCyclStep);
    }

    resetHandler_->resetCounters(
            step, step - inputrec->init_step, mdlog, fplog, cr, fr->nbv.get(), nrnb, fr->pmedata,
            pmeLoadBalanceHelper_ ? pmeLoadBalanceHelper_->loadBalancingObject() : nullptr, wcycle,
            walltime_accounting);
}

void ModularSimulator::simulatorTeardown()
{

    // Stop measuring walltime
    walltime_accounting_end_time(walltime_accounting);

    if (!thisRankHasDuty(cr, DUTY_PME))
    {
        /* Tell the PME only node to finish */
        gmx_pme_send_finish(cr);
    }

    walltime_accounting_set_nsteps_done(walltime_accounting, step_ - inputrec->init_step);
}

void ModularSimulator::populateTaskQueue()
{
    auto registerRunFunction = std::make_unique<RegisterRunFunction>(
            [this](SimulatorRunFunctionPtr ptr) { taskQueue_.push(std::move(ptr)); });

    Time startTime = inputrec->init_t;
    Time timeStep  = inputrec->delta_t;
    Time time      = startTime + step_ * timeStep;

    // Run an initial call to the signallers
    for (auto& signaller : signallerCallList_)
    {
        signaller->signal(step_, time);
    }

    if (checkpointHelper_)
    {
        checkpointHelper_->run(step_, time);
    }

    if (pmeLoadBalanceHelper_)
    {
        pmeLoadBalanceHelper_->run(step_, time);
    }
    if (domDecHelper_)
    {
        domDecHelper_->run(step_, time);
    }

    do
    {
        // local variables for lambda capturing
        const int  step     = step_;
        const bool isNSStep = step == signalHelper_->nextNSStep_;

        // register pre-step
        (*registerRunFunction)(std::make_unique<SimulatorRunFunction>(
                [this, step, time, isNSStep]() { preStep(step, time, isNSStep); }));
        // register elements for step
        for (auto& element : elementCallList_)
        {
            element->scheduleTask(step_, time, registerRunFunction);
        }
        // register post-step
        (*registerRunFunction)(
                std::make_unique<SimulatorRunFunction>([this, step, time]() { postStep(step, time); }));

        // prepare next step
        step_++;
        time = startTime + step_ * timeStep;
        for (auto& signaller : signallerCallList_)
        {
            signaller->signal(step_, time);
        }
    } while (step_ != signalHelper_->nextNSStep_ && step_ <= signalHelper_->lastStep_);
}

void ModularSimulator::constructElementsAndSignallers()
{
    /* When restarting from a checkpoint, it can be appropriate to
     * initialize ekind from quantities in the checkpoint. Otherwise,
     * compute_globals must initialize ekind before the simulation
     * starts/restarts. However, only the master rank knows what was
     * found in the checkpoint file, so we have to communicate in
     * order to coordinate the restart.
     *
     * TODO (modular) This should become obsolete when checkpoint reading
     *      happens within the modular simulator framework: The energy
     *      element should read its data from the checkpoint file pointer,
     *      and signal to the compute globals element if it needs anything
     *      reduced.
     *
     * TODO (legacy) Consider removing this communication if/when checkpoint
     *      reading directly follows .tpr reading, because all ranks can
     *      agree on hasReadEkinState at that time.
     */
    hasReadEkinState_ = MASTER(cr) ? state_global->ekinstate.hasReadEkinState : false;
    if (PAR(cr))
    {
        gmx_bcast(sizeof(hasReadEkinState_), &hasReadEkinState_, cr->mpi_comm_mygroup);
    }
    if (hasReadEkinState_)
    {
        restore_ekinstate_from_state(cr, ekind, &state_global->ekinstate);
    }

    // Multisim is currently disabled
    multiSimNeedsSynchronizedState_ = false;

    // Build stop handler (needed by last step signaller builder)
    stopHandler_ = stopHandlerBuilder->getStopHandlerMD(
            compat::not_null<SimulationSignal*>(&signals_[eglsSTOPCOND]),
            multiSimNeedsSynchronizedState_, MASTER(cr), inputrec->nstlist, mdrunOptions.reproducible,
            nstglobalcomm_, mdrunOptions.maximumHoursToRun, inputrec->nstlist == 0, fplog,
            stophandlerCurrentStep_, stophandlerIsNSStep_, walltime_accounting);

    // Get fully interconnected builders
    auto builders = ElementAndSignallerBuilders::getInstance(this);
    /*
     * Register the simulator itself to the neighbor search / last step signaller
     */
    builders.neighborSearchSignaller->registerSignallerClient(compat::make_not_null(signalHelper_.get()));
    builders.lastStepSignaller->registerSignallerClient(compat::make_not_null(signalHelper_.get()));

    /*
     * Build integrator - this takes care of force calculation, propagation,
     * constraining, and of the place the statePropagatorData and the energy element
     * have a full timestep state.
     */
    auto integrator = buildIntegrator(&builders);

    /*
     * Build infrastructure elements
     */
    domDecHelper_         = builders.domDecHelper->build();
    pmeLoadBalanceHelper_ = builders.pmeLoadBalanceHelper->build();
    topologyHolder_       = builders.topologyHolder->build();
    checkpointHelper_     = builders.checkpointHelper->build();

    const bool simulationsShareResetCounters = false;
    resetHandler_                            = std::make_unique<ResetHandler>(
            compat::make_not_null<SimulationSignal*>(&signals_[eglsRESETCOUNTERS]),
            simulationsShareResetCounters, inputrec->nsteps, MASTER(cr),
            mdrunOptions.timingOptions.resetHalfway, mdrunOptions.maximumHoursToRun, mdlog, wcycle,
            walltime_accounting);

    /*
     * Build signaller list
     *
     * Note that as signallers depend on each others, the order of calling the signallers
     * matters. It is the responsibility of this builder to ensure that the order is
     * maintained.
     *
     * As the trajectory element is both a signaller and an element, it is added to the
     * signaller _call_ list, but owned by the element list.
     */
    auto trajectoryElement = builders.trajectoryElement->build();

    addToCallListAndMove(builders.neighborSearchSignaller->build(), signallerCallList_,
                         signallersOwnershipList_);
    addToCallListAndMove(builders.lastStepSignaller->build(), signallerCallList_, signallersOwnershipList_);
    addToCallListAndMove(builders.loggingSignaller->build(), signallerCallList_, signallersOwnershipList_);
    addToCallList(trajectoryElement, signallerCallList_);
    addToCallListAndMove(builders.energySignaller->build(), signallerCallList_, signallersOwnershipList_);

    /*
     * Build the element list
     *
     * This is the actual sequence of (non-infrastructure) elements to be run.
     * For NVE, the trajectory element is used outside of the integrator
     * (composite) element, as well as the checkpoint helper. The checkpoint
     * helper should be on top of the loop, and is only part of the simulator
     * call list to be able to react to the last step being signalled.
     */
    addToCallList(checkpointHelper_, elementCallList_);
    addToCallListAndMove(builders.freeEnergyPerturbationElement->build(), elementCallList_,
                         elementsOwnershipList_);
    addToCallListAndMove(std::move(integrator), elementCallList_, elementsOwnershipList_);
    addToCallListAndMove(std::move(trajectoryElement), elementCallList_, elementsOwnershipList_);
    // for vv, we need to setup statePropagatorData after the compute
    // globals so that we reset the right velocities
    // TODO: Avoid this by getting rid of the need of resetting velocities in vv
    elementsOwnershipList_.emplace_back(builders.statePropagatorData->build());
}

std::unique_ptr<ISimulatorElement> ModularSimulator::buildIntegrator(ElementAndSignallerBuilders* builders)
{
    // list of elements owned by the simulator composite object
    std::vector<std::unique_ptr<ISimulatorElement>> elementsOwnershipList;
    // call list of the simulator composite object
    std::vector<compat::not_null<ISimulatorElement*>> elementCallList;

    // Note that the state propagator data needs to be added to the ownership list later, so we will only add the (non-owning) pointer to the call list here

    if (inputrec->eI == eiMD)
    {
        addToCallListAndMove(builders->forceElement->build(), elementCallList, elementsOwnershipList);
        addToCallList(builders->statePropagatorData->getPointer(),
                      elementCallList); // we have a full microstate at time t here!
        addToCallListAndMove(builders->vRescaleThermostat->build(-1, false), elementCallList,
                             elementsOwnershipList);
        addToCallListAndMove(builders->leapFrogPropagator->build(), elementCallList, elementsOwnershipList);
        addToCallListAndMove(builders->constraintsElement->build<ConstraintVariable::Positions>(),
                             elementCallList, elementsOwnershipList);
        addToCallListAndMove(builders->computeGlobalsElement->build<ComputeGlobalsAlgorithm::LeapFrog>(),
                             elementCallList, elementsOwnershipList);
        addToCallListAndMove(builders->energyElement->build(), elementCallList,
                             elementsOwnershipList); // we have the energies at time t here!
        addToCallListAndMove(builders->parrinelloRahmanBarostat->build(-1), elementCallList,
                             elementsOwnershipList);
    }
    else if (inputrec->eI == eiVV)
    {
        // compute globals is called twice in VV, so we build it here, and add it once only to the call list, and move it only for the second call
        auto computeGlobalsElement =
                builders->computeGlobalsElement->build<ComputeGlobalsAlgorithm::VelocityVerlet>();

        addToCallListAndMove(builders->forceElement->build(), elementCallList, elementsOwnershipList);
        addToCallListAndMove(builders->velocityPropagator->build(), elementCallList, elementsOwnershipList);
        addToCallListAndMove(builders->constraintsElement->build<ConstraintVariable::Velocities>(),
                             elementCallList, elementsOwnershipList);
        addToCallList(computeGlobalsElement.get(), elementCallList);
        addToCallList(builders->statePropagatorData->getPointer(),
                      elementCallList); // we have a full microstate at time t here!
        addToCallListAndMove(builders->vRescaleThermostat->build(0, true), elementCallList,
                             elementsOwnershipList);
        addToCallListAndMove(builders->velocityVerletPropagator->build(), elementCallList,
                             elementsOwnershipList);
        addToCallListAndMove(builders->constraintsElement->build<ConstraintVariable::Positions>(),
                             elementCallList, elementsOwnershipList);
        addToCallListAndMove(std::move(computeGlobalsElement), elementCallList, elementsOwnershipList);
        addToCallListAndMove(builders->energyElement->build(), elementCallList,
                             elementsOwnershipList); // we have the energies at time t here!
        addToCallListAndMove(builders->parrinelloRahmanBarostat->build(-1), elementCallList,
                             elementsOwnershipList);
    }
    else
    {
        gmx_fatal(FARGS, "Integrator not implemented for the modular simulator.");
    }

    auto integrator = std::make_unique<CompositeSimulatorElement>(std::move(elementCallList),
                                                                  std::move(elementsOwnershipList));
    // std::move *should* not be needed with c++-14, but clang-3.6 still requires it
    return std::move(integrator);
}

bool ModularSimulator::isInputCompatible(bool                             exitOnFailure,
                                         const t_inputrec*                inputrec,
                                         bool                             doRerun,
                                         const gmx_mtop_t&                globalTopology,
                                         const gmx_multisim_t*            ms,
                                         const ReplicaExchangeParameters& replExParams,
                                         const t_fcdata*                  fcd,
                                         bool                             doEssentialDynamics,
                                         bool                             doMembed)
{
    auto conditionalAssert = [exitOnFailure](bool condition, const char* message) {
        if (exitOnFailure)
        {
            GMX_RELEASE_ASSERT(condition, message);
        }
        return condition;
    };

    bool isInputCompatible = true;

    // GMX_USE_MODULAR_SIMULATOR allows to use modular simulator also for non-standard uses,
    // such as the leap-frog integrator
    const auto modularSimulatorExplicitlyTurnedOn = (getenv("GMX_USE_MODULAR_SIMULATOR") != nullptr);
    // GMX_USE_MODULAR_SIMULATOR allows to use disable modular simulator for all uses,
    // including the velocity-verlet integrator used by default
    const auto modularSimulatorExplicitlyTurnedOff = (getenv("GMX_DISABLE_MODULAR_SIMULATOR") != nullptr);

    GMX_RELEASE_ASSERT(
            !(modularSimulatorExplicitlyTurnedOn && modularSimulatorExplicitlyTurnedOff),
            "Cannot have both GMX_USE_MODULAR_SIMULATOR=ON and GMX_DISABLE_MODULAR_SIMULATOR=ON. "
            "Unset one of the two environment variables to explicitly chose which simulator to "
            "use, "
            "or unset both to recover default behavior.");

    GMX_RELEASE_ASSERT(
            !(modularSimulatorExplicitlyTurnedOff && inputrec->eI == eiVV
              && inputrec->epc == epcPARRINELLORAHMAN),
            "Cannot use a Parrinello-Rahman barostat with md-vv and "
            "GMX_DISABLE_MODULAR_SIMULATOR=ON, "
            "as the Parrinello-Rahman barostat is not implemented in the legacy simulator. Unset "
            "GMX_DISABLE_MODULAR_SIMULATOR or use a different pressure control algorithm.");

    isInputCompatible =
            isInputCompatible
            && conditionalAssert(
                       inputrec->eI == eiMD || inputrec->eI == eiVV,
                       "Only integrators md and md-vv are supported by the modular simulator.");
    isInputCompatible = isInputCompatible
                        && conditionalAssert(inputrec->eI != eiMD || modularSimulatorExplicitlyTurnedOn,
                                             "Set GMX_USE_MODULAR_SIMULATOR=ON to use the modular "
                                             "simulator with integrator md.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(!doRerun, "Rerun is not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(
                       inputrec->etc == etcNO || inputrec->etc == etcVRESCALE,
                       "Only v-rescale thermostat is supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(
                       inputrec->epc == epcNO || inputrec->epc == epcPARRINELLORAHMAN,
                       "Only Parrinello-Rahman barostat is supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(
                       !(inputrecNptTrotter(inputrec) || inputrecNphTrotter(inputrec)
                         || inputrecNvtTrotter(inputrec)),
                       "Legacy Trotter decomposition is not supported by the modular simulator.");
    isInputCompatible = isInputCompatible
                        && conditionalAssert(inputrec->efep == efepNO || inputrec->efep == efepYES
                                                     || inputrec->efep == efepSLOWGROWTH,
                                             "Expanded ensemble free energy calculation is not "
                                             "supported by the modular simulator.");
    isInputCompatible = isInputCompatible
                        && conditionalAssert(!inputrec->bPull,
                                             "Pulling is not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(inputrec->opts.ngacc == 1 && inputrec->opts.acc[0][XX] == 0.0
                                         && inputrec->opts.acc[0][YY] == 0.0
                                         && inputrec->opts.acc[0][ZZ] == 0.0 && inputrec->cos_accel == 0.0,
                                 "Acceleration is not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(inputrec->opts.ngfrz == 1 && inputrec->opts.nFreeze[0][XX] == 0
                                         && inputrec->opts.nFreeze[0][YY] == 0
                                         && inputrec->opts.nFreeze[0][ZZ] == 0,
                                 "Freeze groups are not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(
                       inputrec->deform[XX][XX] == 0.0 && inputrec->deform[XX][YY] == 0.0
                               && inputrec->deform[XX][ZZ] == 0.0 && inputrec->deform[YY][XX] == 0.0
                               && inputrec->deform[YY][YY] == 0.0 && inputrec->deform[YY][ZZ] == 0.0
                               && inputrec->deform[ZZ][XX] == 0.0 && inputrec->deform[ZZ][YY] == 0.0
                               && inputrec->deform[ZZ][ZZ] == 0.0,
                       "Deformation is not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(gmx_mtop_interaction_count(globalTopology, IF_VSITE) == 0,
                                 "Virtual sites are not supported by the modular simulator.");
    isInputCompatible = isInputCompatible
                        && conditionalAssert(!inputrec->bDoAwh,
                                             "AWH is not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(gmx_mtop_ftype_count(globalTopology, F_DISRES) == 0,
                                 "Distance restraints are not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(
                       gmx_mtop_ftype_count(globalTopology, F_ORIRES) == 0,
                       "Orientation restraints are not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(ms == nullptr,
                                 "Multi-sim are not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(replExParams.exchangeInterval == 0,
                                 "Replica exchange is not supported by the modular simulator.");

    int numEnsembleRestraintSystems;
    if (fcd)
    {
        numEnsembleRestraintSystems = fcd->disres.nsystems;
    }
    else
    {
        auto distantRestraintEnsembleEnvVar = getenv("GMX_DISRE_ENSEMBLE_SIZE");
        numEnsembleRestraintSystems =
                (ms != nullptr && distantRestraintEnsembleEnvVar != nullptr)
                        ? static_cast<int>(strtol(distantRestraintEnsembleEnvVar, nullptr, 10))
                        : 0;
    }
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(numEnsembleRestraintSystems <= 1,
                                 "Ensemble restraints are not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(!doSimulatedAnnealing(inputrec),
                                 "Simulated annealing is not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(!inputrec->bSimTemp,
                                 "Simulated tempering is not supported by the modular simulator.");
    isInputCompatible = isInputCompatible
                        && conditionalAssert(!inputrec->bExpanded,
                                             "Expanded ensemble simulations are not supported by "
                                             "the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(!doEssentialDynamics,
                                 "Essential dynamics is not supported by the modular simulator.");
    isInputCompatible = isInputCompatible
                        && conditionalAssert(inputrec->eSwapCoords == eswapNO,
                                             "Ion / water position swapping is not supported by "
                                             "the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(!inputrec->bIMD,
                                 "Interactive MD is not supported by the modular simulator.");
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(!doMembed,
                                 "Membrane embedding is not supported by the modular simulator.");
    // TODO: Change this to the boolean passed when we merge the user interface change for the GPU update.
    isInputCompatible =
            isInputCompatible
            && conditionalAssert(
                       getenv("GMX_FORCE_UPDATE_DEFAULT_GPU") == nullptr,
                       "Integration on the GPU is not supported by the modular simulator.");
    // Modular simulator is centered around NS updates
    // TODO: think how to handle nstlist == 0
    isInputCompatible = isInputCompatible
                        && conditionalAssert(inputrec->nstlist != 0,
                                             "Simulations without neighbor list update are not "
                                             "supported by the modular simulator.");
    isInputCompatible = isInputCompatible
                        && conditionalAssert(!GMX_FAHCORE,
                                             "GMX_FAHCORE not supported by the modular simulator.");

    return isInputCompatible;
}

void ModularSimulator::checkInputForDisabledFunctionality()
{
    isInputCompatible(true, inputrec, doRerun, *top_global, ms, replExParams, fcd,
                      opt2bSet("-ei", nfile, fnm), membed != nullptr);
    if (observablesHistory->edsamHistory)
    {
        gmx_fatal(FARGS,
                  "The checkpoint is from a run with essential dynamics sampling, "
                  "but the current run did not specify the -ei option. "
                  "Either specify the -ei option to mdrun, or do not use this checkpoint file.");
    }
}

SignallerCallbackPtr ModularSimulator::SignalHelper::registerLastStepCallback()
{
    return std::make_unique<SignallerCallback>(
            [this](Step step, Time gmx_unused time) { this->lastStep_ = step; });
}

SignallerCallbackPtr ModularSimulator::SignalHelper::registerNSCallback()
{
    return std::make_unique<SignallerCallback>(
            [this](Step step, Time gmx_unused time) { this->nextNSStep_ = step; });
}
} // namespace gmx
