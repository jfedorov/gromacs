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
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/checkpointhandler.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/energyoutput.h"
#include "gromacs/mdlib/forcerec.h"
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
#include "gromacs/mdtypes/state.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"

#include "compositesimulatorelement.h"
#include "computeglobalselement.h"
#include "constraintelement.h"
#include "energyelement.h"
#include "forceelement.h"
#include "freeenergyperturbationelement.h"
#include "parrinellorahmanbarostat.h"
#include "propagator.h"
#include "signallers.h"
#include "statepropagatordata.h"
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
    bool hasReadEkinState = MASTER(cr) ? state_global->ekinstate.hasReadEkinState : false;
    if (PAR(cr))
    {
        gmx_bcast(sizeof(hasReadEkinState), &hasReadEkinState, cr->mpi_comm_mygroup);
    }
    if (hasReadEkinState)
    {
        restore_ekinstate_from_state(cr, ekind, &state_global->ekinstate);
    }

    // Build the topology holder
    TopologyHolderBuilder topologyHolderBuilder(*top_global, cr, inputrec, fr, mdAtoms, constr, vsite);

    /*
     * Create simulator builders
     */
    SignallerBuilder<NeighborSearchSignaller> neighborSearchSignallerBuilder;
    SignallerBuilder<LastStepSignaller>       lastStepSignallerBuilder;
    SignallerBuilder<LoggingSignaller>        loggingSignallerBuilder;
    SignallerBuilder<EnergySignaller>         energySignallerBuilder;
    TrajectoryElementBuilder                  trajectoryElementBuilder;

    // Multisim is currently disabled
    const bool simulationsShareState = false;

    // Builder for the checkpoint helper
    CheckpointHelperBuilder checkpointHelperBuilder(inputrec->init_step, top_global->natoms, fplog,
                                                    cr, observablesHistory, walltime_accounting,
                                                    state_global, mdrunOptions.writeConfout);
    checkpointHelperBuilder.setCheckpointHandler(std::make_unique<CheckpointHandler>(
            compat::make_not_null<SimulationSignal*>(&signals_[eglsCHKPT]), simulationsShareState,
            inputrec->nstlist == 0, MASTER(cr), mdrunOptions.writeConfout,
            mdrunOptions.checkpointOptions.period));

    // State propagator data builder
    StatePropagatorDataBuilder statePropagatorDataBuilder(
            top_global->natoms, fplog, cr, state_global, inputrec->nstxout, inputrec->nstvout,
            inputrec->nstfout, inputrec->nstxout_compressed, fr->nbv->useGpu(), fr->bMolPBC,
            mdrunOptions.writeConfout, opt2fn("-c", nfile, fnm), inputrec, mdAtoms->mdatoms());

    // Energy element builder
    EnergyElementBuilder energyElementBuilder(inputrec, mdAtoms, enerd, ekind, constr, fplog, fcd,
                                              mdModulesNotifier, MASTER(cr), observablesHistory,
                                              startingBehavior);

    // Free energy perturbation element builder
    FreeEnergyPerturbationElementBuilder freeEnergyPerturbationElementBuilder(fplog, inputrec, mdAtoms);

    // Domain decomposition helper builder
    DomDecHelperBuilder domDecHelperBuilder(
            mdrunOptions.verbose, mdrunOptions.verboseStepPrintInterval, nstglobalcomm_, fplog, cr,
            mdlog, constr, inputrec, mdAtoms, nrnb, wcycle, fr, vsite, imdSession, pull_work);

    // PME load balance helper builder
    PmeLoadBalanceHelperBuilder pmeLoadBalanceHelperBuilder(mdrunOptions, fplog, cr, mdlog,
                                                            inputrec, wcycle, fr);

    /*
     * Connect simulator builders
     */
    // State propagator data
    statePropagatorDataBuilder.setFreeEnergyPerturbationElement(
            freeEnergyPerturbationElementBuilder.getPointer());
    statePropagatorDataBuilder.registerWithLastStepSignaller(&lastStepSignallerBuilder);
    statePropagatorDataBuilder.registerWithTrajectoryElement(&trajectoryElementBuilder);
    statePropagatorDataBuilder.setTopologyHolder(topologyHolderBuilder.getPointer());
    statePropagatorDataBuilder.registerWithCheckpointHelper(&checkpointHelperBuilder);

    // Energy element
    energyElementBuilder.setStatePropagatorData(statePropagatorDataBuilder.getPointer());
    energyElementBuilder.setFreeEnergyPerturbationElement(freeEnergyPerturbationElementBuilder.getPointer());
    energyElementBuilder.registerWithEnergySignaller(&energySignallerBuilder);
    energyElementBuilder.registerWithTrajectoryElement(&trajectoryElementBuilder);
    energyElementBuilder.registerWithCheckpointHelper(&checkpointHelperBuilder);
    energyElementBuilder.setTopologyHolder(topologyHolderBuilder.getPointer());

    // FEP element
    freeEnergyPerturbationElementBuilder.registerWithCheckpointHelper(&checkpointHelperBuilder);

    // Checkpoint helper
    checkpointHelperBuilder.registerWithLastStepSignaller(&lastStepSignallerBuilder);

    // DD helper
    domDecHelperBuilder.setStatePropagatorData(statePropagatorDataBuilder.getPointer());
    domDecHelperBuilder.registerWithNeighborSearchSignaller(&neighborSearchSignallerBuilder);
    domDecHelperBuilder.setTopologyHolder(topologyHolderBuilder.getPointer());

    // PME load balance helper
    pmeLoadBalanceHelperBuilder.setStatePropagatorData(statePropagatorDataBuilder.getPointer());
    pmeLoadBalanceHelperBuilder.registerWithNeighborSearchSignaller(&neighborSearchSignallerBuilder);

    /*
     * Build stop handler
     */
    stopHandler_ = stopHandlerBuilder->getStopHandlerMD(
            compat::not_null<SimulationSignal*>(&signals_[eglsSTOPCOND]), simulationsShareState,
            MASTER(cr), inputrec->nstlist, mdrunOptions.reproducible, nstglobalcomm_,
            mdrunOptions.maximumHoursToRun, inputrec->nstlist == 0, fplog, stophandlerCurrentStep_,
            stophandlerIsNSStep_, walltime_accounting);

    /*
     * Register the simulator itself to the neighbor search / last step signaller
     */
    neighborSearchSignallerBuilder.registerSignallerClient(compat::make_not_null(signalHelper_.get()));
    lastStepSignallerBuilder.registerSignallerClient(compat::make_not_null(signalHelper_.get()));

    /*
     * Build integrator - this takes care of force calculation, propagation,
     * constraining, and of the place the statePropagatorData and the energy element
     * have a full timestep state.
     */
    auto integrator = buildIntegrator(
            &neighborSearchSignallerBuilder, &energySignallerBuilder, &loggingSignallerBuilder,
            &trajectoryElementBuilder, &checkpointHelperBuilder, &domDecHelperBuilder,
            compat::make_not_null(statePropagatorDataBuilder.getPointer()), &energyElementBuilder,
            freeEnergyPerturbationElementBuilder.getPointer(), &topologyHolderBuilder, hasReadEkinState);

    /*
     * Build infrastructure elements
     */
    domDecHelper_         = domDecHelperBuilder.build();
    pmeLoadBalanceHelper_ = pmeLoadBalanceHelperBuilder.build();
    topologyHolder_       = topologyHolderBuilder.build();

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
     */
    auto energySignaller = energySignallerBuilder.build(
            inputrec->nstcalcenergy, inputrec->fepvals->nstdhdl, inputrec->nstpcouple);
    trajectoryElementBuilder.registerSignallerClient(compat::make_not_null(energySignaller.get()));
    loggingSignallerBuilder.registerSignallerClient(compat::make_not_null(energySignaller.get()));
    auto trajectoryElement = trajectoryElementBuilder.build(
            fplog, nfile, fnm, mdrunOptions, cr, outputProvider, mdModulesNotifier, inputrec,
            top_global, oenv, wcycle, startingBehavior, simulationsShareState);
    loggingSignallerBuilder.registerSignallerClient(compat::make_not_null(trajectoryElement.get()));

    checkpointHelperBuilder.setTrajectoryElement(trajectoryElement.get());
    checkpointHelper_ = checkpointHelperBuilder.build();

    lastStepSignallerBuilder.registerSignallerClient(compat::make_not_null(trajectoryElement.get()));
    auto loggingSignaller =
            loggingSignallerBuilder.build(inputrec->nstlog, inputrec->init_step, inputrec->init_t);
    lastStepSignallerBuilder.registerSignallerClient(compat::make_not_null(loggingSignaller.get()));
    auto lastStepSignaller =
            lastStepSignallerBuilder.build(inputrec->nsteps, inputrec->init_step, stopHandler_.get());
    neighborSearchSignallerBuilder.registerSignallerClient(compat::make_not_null(lastStepSignaller.get()));
    auto neighborSearchSignaller = neighborSearchSignallerBuilder.build(
            inputrec->nstlist, inputrec->init_step, inputrec->init_t);

    addToCallListAndMove(std::move(neighborSearchSignaller), signallerCallList_, signallersOwnershipList_);
    addToCallListAndMove(std::move(lastStepSignaller), signallerCallList_, signallersOwnershipList_);
    addToCallListAndMove(std::move(loggingSignaller), signallerCallList_, signallersOwnershipList_);
    addToCallList(trajectoryElement, signallerCallList_);
    addToCallListAndMove(std::move(energySignaller), signallerCallList_, signallersOwnershipList_);

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
    addToCallListAndMove(freeEnergyPerturbationElementBuilder.build(), elementCallList_,
                         elementsOwnershipList_);
    addToCallListAndMove(std::move(integrator), elementCallList_, elementsOwnershipList_);
    addToCallListAndMove(std::move(trajectoryElement), elementCallList_, elementsOwnershipList_);
    // for vv, we need to setup statePropagatorData after the compute
    // globals so that we reset the right velocities
    // TODO: Avoid this by getting rid of the need of resetting velocities in vv
    elementsOwnershipList_.emplace_back(statePropagatorDataBuilder.build());
}

std::unique_ptr<ISimulatorElement> ModularSimulator::buildIntegrator(
        SignallerBuilder<NeighborSearchSignaller>* neighborSearchSignallerBuilder,
        SignallerBuilder<EnergySignaller>*         energySignallerBuilder,
        SignallerBuilder<LoggingSignaller>*        loggingSignallerBuilder,
        TrajectoryElementBuilder*                  trajectoryElementBuilder,
        CheckpointHelperBuilder*                   checkpointHelperBuilder,
        DomDecHelperBuilder*                       domDecHelperBuilder,
        compat::not_null<StatePropagatorData*>     statePropagatorDataPtr,
        EnergyElementBuilder*                      energyElementBuilder,
        FreeEnergyPerturbationElement*             freeEnergyPerturbationElementPtr,
        TopologyHolderBuilder*                     topologyHolderBuilder,
        bool                                       hasReadEkinState)
{
    /*
     * Build module builders
     */
    ConstraintsElementBuilder constraintsElementBuilder(constr, MASTER(cr), fplog, inputrec,
                                                        mdAtoms->mdatoms());

    const bool          isVerbose    = mdrunOptions.verbose;
    const bool          isDynamicBox = inputrecDynamicBox(inputrec);
    ForceElementBuilder forceElementBuilder(isVerbose, isDynamicBox, fplog, cr, inputrec, mdAtoms,
                                            nrnb, fr, fcd, wcycle, runScheduleWork, vsite, imdSession,
                                            pull_work, constr, top_global, enforcedRotation);

    ComputeGlobalsElementBuilder computeGlobalsElementBuilder(
            inputrec->eI, &signals_, nstglobalcomm_, fplog, mdlog, cr, inputrec, mdAtoms, nrnb,
            wcycle, fr, top_global, constr, hasReadEkinState);

    ParrinelloRahmanBarostatBuilder parrinelloRahmanBarostatBuilder(
            inputrec->nstpcouple, inputrec->delta_t * inputrec->nstpcouple, inputrec->init_step,
            fplog, inputrec, mdAtoms, state_global, cr, inputrec->bContinuation);

    VRescaleThermostatBuilder vRescaleThermostatBuilder(
            inputrec->nsttcouple, inputrec->ld_seed, inputrec->opts.ngtc,
            inputrec->delta_t * inputrec->nsttcouple, inputrec->opts.ref_t, inputrec->opts.tau_t,
            inputrec->opts.nrdf, state_global, cr, inputrec->bContinuation, inputrec->etc);

    /*
     * Connect builders
     */
    // Constraint element
    constraintsElementBuilder.setStatePropagatorData(statePropagatorDataPtr);
    constraintsElementBuilder.setEnergyElement(energyElementBuilder->getPointer());
    constraintsElementBuilder.setFreeEnergyPerturbationElement(freeEnergyPerturbationElementPtr);
    constraintsElementBuilder.registerWithEnergySignaller(energySignallerBuilder);
    constraintsElementBuilder.registerWithTrajectorySignaller(trajectoryElementBuilder);
    constraintsElementBuilder.registerWithLoggingSignaller(loggingSignallerBuilder);

    // Force element
    forceElementBuilder.setStatePropagatorData(statePropagatorDataPtr);
    forceElementBuilder.setEnergyElement(energyElementBuilder->getPointer());
    forceElementBuilder.setFreeEnergyPerturbationElement(freeEnergyPerturbationElementPtr);
    forceElementBuilder.registerWithNeighborSearchSignaller(neighborSearchSignallerBuilder);
    forceElementBuilder.registerWithEnergySignaller(energySignallerBuilder);
    forceElementBuilder.registerWithTopologyHolder(topologyHolderBuilder);

    // Compute-globals element
    computeGlobalsElementBuilder.setStatePropagatorData(statePropagatorDataPtr);
    computeGlobalsElementBuilder.setEnergyElement(energyElementBuilder->getPointer());
    computeGlobalsElementBuilder.setFreeEnergyPerturbationElement(freeEnergyPerturbationElementPtr);
    computeGlobalsElementBuilder.registerWithTopologyHolder(topologyHolderBuilder);
    computeGlobalsElementBuilder.registerWithEnergySignaller(energySignallerBuilder);
    computeGlobalsElementBuilder.registerWithTrajectorySignaller(trajectoryElementBuilder);

    domDecHelperBuilder->setComputeGlobalsElementBuilder(&computeGlobalsElementBuilder);

    // Parrinello-Rahman barostat
    parrinelloRahmanBarostatBuilder.setStatePropagatorData(statePropagatorDataPtr);
    parrinelloRahmanBarostatBuilder.setEnergyElementBuilder(energyElementBuilder);
    parrinelloRahmanBarostatBuilder.registerWithCheckpointHelper(checkpointHelperBuilder);

    // v-rescale thermostat
    vRescaleThermostatBuilder.setEnergyElementBuilder(energyElementBuilder);
    vRescaleThermostatBuilder.registerWithCheckpointHelper(checkpointHelperBuilder);

    // list of elements owned by the simulator composite object
    std::vector<std::unique_ptr<ISimulatorElement>> elementsOwnershipList;
    // call list of the simulator composite object
    std::vector<compat::not_null<ISimulatorElement*>> elementCallList;

    if (inputrec->eI == eiMD)
    {
        PropagatorBuilder<IntegrationStep::LeapFrog> propagatorBuilder(inputrec->delta_t, mdAtoms, wcycle);
        propagatorBuilder.setStatePropagatorData(statePropagatorDataPtr);
        parrinelloRahmanBarostatBuilder.setPropagatorBuilder(&propagatorBuilder);
        vRescaleThermostatBuilder.setPropagatorBuilder(&propagatorBuilder);

        addToCallListAndMove(forceElementBuilder.build(), elementCallList, elementsOwnershipList);
        addToCallList(statePropagatorDataPtr, elementCallList); // we have a full microstate at time t here!

        addToCallListAndMove(vRescaleThermostatBuilder.build(-1, false), elementCallList,
                             elementsOwnershipList);

        addToCallListAndMove(propagatorBuilder.build(), elementCallList, elementsOwnershipList);

        addToCallListAndMove(constraintsElementBuilder.build<ConstraintVariable::Positions>(),
                             elementCallList, elementsOwnershipList);

        addToCallListAndMove(computeGlobalsElementBuilder.build<ComputeGlobalsAlgorithm::LeapFrog>(),
                             elementCallList, elementsOwnershipList);

        addToCallListAndMove(energyElementBuilder->build(), elementCallList,
                             elementsOwnershipList); // we have the energies at time t here!

        addToCallListAndMove(parrinelloRahmanBarostatBuilder.build(-1), elementCallList,
                             elementsOwnershipList);
    }
    else if (inputrec->eI == eiVV)
    {
        PropagatorBuilder<IntegrationStep::VelocitiesOnly> velocityPropagatorBuilder(
                inputrec->delta_t * 0.5, mdAtoms, wcycle);
        velocityPropagatorBuilder.setStatePropagatorData(statePropagatorDataPtr);
        parrinelloRahmanBarostatBuilder.setPropagatorBuilder(&velocityPropagatorBuilder);

        PropagatorBuilder<IntegrationStep::VelocityVerletPositionsAndVelocities> velocityAndPositionPropagatorBuilder(
                inputrec->delta_t, mdAtoms, wcycle);
        velocityAndPositionPropagatorBuilder.setStatePropagatorData(statePropagatorDataPtr);
        vRescaleThermostatBuilder.setPropagatorBuilder(&velocityAndPositionPropagatorBuilder);

        addToCallListAndMove(forceElementBuilder.build(), elementCallList, elementsOwnershipList);

        addToCallListAndMove(velocityPropagatorBuilder.build(), elementCallList, elementsOwnershipList);

        addToCallListAndMove(constraintsElementBuilder.build<ConstraintVariable::Velocities>(),
                             elementCallList, elementsOwnershipList);

        auto computeGlobalsElement =
                computeGlobalsElementBuilder.build<ComputeGlobalsAlgorithm::VelocityVerlet>();
        addToCallList(computeGlobalsElement.get(), elementCallList);
        addToCallList(statePropagatorDataPtr, elementCallList); // we have a full microstate at time t here!

        addToCallListAndMove(vRescaleThermostatBuilder.build(0, true), elementCallList,
                             elementsOwnershipList);

        addToCallListAndMove(velocityAndPositionPropagatorBuilder.build(), elementCallList,
                             elementsOwnershipList);

        addToCallListAndMove(constraintsElementBuilder.build<ConstraintVariable::Positions>(),
                             elementCallList, elementsOwnershipList);

        addToCallListAndMove(std::move(computeGlobalsElement), elementCallList, elementsOwnershipList);
        addToCallListAndMove(energyElementBuilder->build(), elementCallList,
                             elementsOwnershipList); // we have the energies at time t here!

        addToCallListAndMove(parrinelloRahmanBarostatBuilder.build(-1), elementCallList,
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
