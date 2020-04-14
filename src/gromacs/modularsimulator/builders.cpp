/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020, by the GROMACS development team, led by
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
 * \brief Declares the ElementAndSignallerBuilders struct.
 *
 * This helper struct holds all element and signaller builders for the
 * modular simulator. It is defined in its own header to reduce
 * include dependencies. Its implementation is located in
 * src/gromacs/modularsimulator/modularsimulator.cpp.
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "builders.h"

#include "gromacs/commandline/filenm.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/mdrunoptions.h"
#include "gromacs/nbnxm/nbnxm.h"

#include "computeglobalselement.h"
#include "constraintelement.h"
#include "energyelement.h"
#include "forceelement.h"
#include "freeenergyperturbationelement.h"
#include "modularsimulator.h"
#include "parrinellorahmanbarostat.h"
#include "signallers.h"
#include "statepropagatordata.h"
#include "trajectoryelement.h"
#include "vrescalethermostat.h"

namespace gmx
{

ElementAndSignallerBuilders::ElementAndSignallerBuilders(ModularSimulator* simulator)
{
    // Signalers
    neighborSearchSignaller = std::make_unique<SignallerBuilder<NeighborSearchSignaller>>(
            simulator->inputrec->nstlist, simulator->inputrec->init_step, simulator->inputrec->init_t);
    lastStepSignaller = std::make_unique<SignallerBuilder<LastStepSignaller>>(
            simulator->inputrec->nsteps, simulator->inputrec->init_step, simulator->stopHandler_.get());
    loggingSignaller = std::make_unique<SignallerBuilder<LoggingSignaller>>(
            simulator->inputrec->nstlog, simulator->inputrec->init_step, simulator->inputrec->init_t);
    energySignaller = std::make_unique<SignallerBuilder<EnergySignaller>>(
            simulator->inputrec->nstcalcenergy, simulator->inputrec->fepvals->nstdhdl,
            simulator->inputrec->nstpcouple);
    trajectoryElement = std::make_unique<TrajectoryElementBuilder>(
            simulator->fplog, simulator->nfile, simulator->fnm, simulator->mdrunOptions,
            simulator->cr, simulator->outputProvider, simulator->mdModulesNotifier,
            simulator->inputrec, simulator->top_global, simulator->oenv, simulator->wcycle,
            simulator->startingBehavior, simulator->multiSimNeedsSynchronizedState_);

    // Data elements
    statePropagatorData = std::make_unique<StatePropagatorDataBuilder>(
            simulator->top_global->natoms, simulator->fplog, simulator->cr, simulator->state_global,
            simulator->inputrec->nstxout, simulator->inputrec->nstvout, simulator->inputrec->nstfout,
            simulator->inputrec->nstxout_compressed, simulator->fr->nbv->useGpu(), simulator->fr->bMolPBC,
            simulator->mdrunOptions.writeConfout, opt2fn("-c", simulator->nfile, simulator->fnm),
            simulator->inputrec, simulator->mdAtoms->mdatoms());
    energyElement = std::make_unique<EnergyElementBuilder>(
            simulator->inputrec, simulator->mdAtoms, simulator->enerd, simulator->ekind,
            simulator->constr, simulator->fplog, simulator->fcd, simulator->mdModulesNotifier,
            MASTER(simulator->cr), simulator->startingBehavior);
    freeEnergyPerturbationElement = std::make_unique<FreeEnergyPerturbationElementBuilder>(
            simulator->fplog, simulator->inputrec, simulator->mdAtoms);

    // Infrastructure elements
    topologyHolder = std::make_unique<TopologyHolderBuilder>(
            *simulator->top_global, simulator->cr, simulator->inputrec, simulator->fr,
            simulator->mdAtoms, simulator->constr, simulator->vsite);
    checkpointHelper = std::make_unique<CheckpointHelperBuilder>(
            simulator->inputrec->init_step, simulator->top_global->natoms,
            std::move(simulator->modularSimulatorCheckpointTree), simulator->startingBehavior,
            simulator->fplog, simulator->cr, simulator->observablesHistory, simulator->walltime_accounting,
            simulator->state_global, simulator->mdrunOptions.writeConfout);
    checkpointHelper->setCheckpointHandler(std::make_unique<CheckpointHandler>(
            compat::make_not_null<SimulationSignal*>(&simulator->signals_[eglsCHKPT]),
            simulator->multiSimNeedsSynchronizedState_, simulator->inputrec->nstlist == 0,
            MASTER(simulator->cr), simulator->mdrunOptions.writeConfout,
            simulator->mdrunOptions.checkpointOptions.period));
    domDecHelper = std::make_unique<DomDecHelperBuilder>(
            simulator->mdrunOptions.verbose, simulator->mdrunOptions.verboseStepPrintInterval,
            simulator->nstglobalcomm_, simulator->fplog, simulator->cr, simulator->mdlog,
            simulator->constr, simulator->inputrec, simulator->mdAtoms, simulator->nrnb, simulator->wcycle,
            simulator->fr, simulator->vsite, simulator->imdSession, simulator->pull_work);
    pmeLoadBalanceHelper = std::make_unique<PmeLoadBalanceHelperBuilder>(
            simulator->mdrunOptions, simulator->fplog, simulator->cr, simulator->mdlog,
            simulator->inputrec, simulator->wcycle, simulator->fr);

    // Integrator elements
    constraintsElement = std::make_unique<ConstraintsElementBuilder>(
            simulator->constr, MASTER(simulator->cr), simulator->fplog, simulator->inputrec,
            simulator->mdAtoms->mdatoms());
    forceElement = std::make_unique<ForceElementBuilder>(
            simulator->mdrunOptions.verbose, inputrecDynamicBox(simulator->inputrec),
            simulator->fplog, simulator->cr, simulator->inputrec, simulator->mdAtoms, simulator->nrnb,
            simulator->fr, simulator->fcd, simulator->wcycle, simulator->runScheduleWork,
            simulator->vsite, simulator->imdSession, simulator->pull_work, simulator->constr,
            simulator->top_global, simulator->enforcedRotation);
    computeGlobalsElement = std::make_unique<ComputeGlobalsElementBuilder>(
            simulator->inputrec->eI, &simulator->signals_, simulator->nstglobalcomm_, simulator->fplog,
            simulator->mdlog, simulator->cr, simulator->inputrec, simulator->mdAtoms, simulator->nrnb,
            simulator->wcycle, simulator->fr, simulator->top_global, simulator->constr);
    parrinelloRahmanBarostat = std::make_unique<ParrinelloRahmanBarostatBuilder>(
            simulator->inputrec->nstpcouple, simulator->inputrec->delta_t * simulator->inputrec->nstpcouple,
            simulator->inputrec->init_step, simulator->fplog, simulator->inputrec, simulator->mdAtoms);
    vRescaleThermostat = std::make_unique<VRescaleThermostatBuilder>(
            simulator->inputrec->nsttcouple, simulator->inputrec->ld_seed, simulator->inputrec->opts.ngtc,
            simulator->inputrec->delta_t * simulator->inputrec->nsttcouple,
            simulator->inputrec->opts.ref_t, simulator->inputrec->opts.tau_t,
            simulator->inputrec->opts.nrdf, simulator->inputrec->etc);
    // TODO: Can this if / else be moved into the builder if we move to a more complex (policy-based) builder?
    if (simulator->inputrec->eI == eiMD)
    {
        leapFrogPropagator = std::make_unique<PropagatorBuilder<IntegrationStep::LeapFrog>>(
                simulator->inputrec->delta_t, simulator->mdAtoms, simulator->wcycle);
    }
    else if (simulator->inputrec->eI == eiVV)
    {
        velocityVerletPropagator =
                std::make_unique<PropagatorBuilder<IntegrationStep::VelocityVerletPositionsAndVelocities>>(
                        simulator->inputrec->delta_t, simulator->mdAtoms, simulator->wcycle);
        velocityPropagator = std::make_unique<PropagatorBuilder<IntegrationStep::VelocitiesOnly>>(
                simulator->inputrec->delta_t * 0.5, simulator->mdAtoms, simulator->wcycle);
    }
}

void ElementAndSignallerBuilders::connectBuilders()
{
    neighborSearchSignaller->connectWithBuilders(this);
    lastStepSignaller->connectWithBuilders(this);
    loggingSignaller->connectWithBuilders(this);
    trajectoryElement->connectWithBuilders(this);
    energySignaller->connectWithBuilders(this);

    statePropagatorData->connectWithBuilders(this);
    energyElement->connectWithBuilders(this);
    freeEnergyPerturbationElement->connectWithBuilders(this);

    checkpointHelper->connectWithBuilders(this);
    domDecHelper->connectWithBuilders(this);
    pmeLoadBalanceHelper->connectWithBuilders(this);

    constraintsElement->connectWithBuilders(this);
    forceElement->connectWithBuilders(this);
    computeGlobalsElement->connectWithBuilders(this);
    parrinelloRahmanBarostat->connectWithBuilders(this);
    vRescaleThermostat->connectWithBuilders(this);

    // Propagators
    if (leapFrogPropagator)
    {
        leapFrogPropagator->connectWithBuilders(this);
    }
    if (velocityVerletPropagator)
    {
        velocityVerletPropagator->connectWithBuilders(this);
    }
    if (velocityPropagator)
    {
        velocityPropagator->connectWithBuilders(this);
    }
}

ElementAndSignallerBuilders ElementAndSignallerBuilders::getInstance(ModularSimulator* simulator)
{
    // First stage: Construct all builders
    ElementAndSignallerBuilders builders(simulator);
    // Second stage: Allow builders to inter-connect
    builders.connectBuilders();
    // Return builders ready to use
    return builders;
}

} // namespace gmx
