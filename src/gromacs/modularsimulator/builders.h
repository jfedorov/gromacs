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
/*! \libinternal \file
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
#ifndef GROMACS_MODULARSIMULATOR_BUILDERS_H
#define GROMACS_MODULARSIMULATOR_BUILDERS_H

#include "propagator.h"

namespace gmx
{
class CheckpointHelperBuilder;
class ComputeGlobalsElementBuilder;
class ConstraintsElementBuilder;
class DomDecHelperBuilder;
class EnergyElementBuilder;
class ForceElementBuilder;
class FreeEnergyPerturbationElementBuilder;
class ModularSimulator;
class ParrinelloRahmanBarostatBuilder;
class PmeLoadBalanceHelperBuilder;
class StatePropagatorDataBuilder;
class TopologyHolderBuilder;
class TrajectoryElementBuilder;
class VRescaleThermostatBuilder;

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Helper struct holding all element and signaller builders.
 */
struct ElementAndSignallerBuilders final
{
    //! Element and signaller builders
    //!{
    std::unique_ptr<SignallerBuilder<NeighborSearchSignaller>> neighborSearchSignaller;
    std::unique_ptr<SignallerBuilder<LastStepSignaller>>       lastStepSignaller;
    std::unique_ptr<SignallerBuilder<LoggingSignaller>>        loggingSignaller;
    std::unique_ptr<SignallerBuilder<EnergySignaller>>         energySignaller;
    std::unique_ptr<TrajectoryElementBuilder>                  trajectoryElement;

    std::unique_ptr<TopologyHolderBuilder>                topologyHolder;
    std::unique_ptr<CheckpointHelperBuilder>              checkpointHelper;
    std::unique_ptr<StatePropagatorDataBuilder>           statePropagatorData;
    std::unique_ptr<EnergyElementBuilder>                 energyElement;
    std::unique_ptr<FreeEnergyPerturbationElementBuilder> freeEnergyPerturbationElement;

    std::unique_ptr<DomDecHelperBuilder>         domDecHelper;
    std::unique_ptr<PmeLoadBalanceHelperBuilder> pmeLoadBalanceHelper;

    std::unique_ptr<ConstraintsElementBuilder>                    constraintsElement;
    std::unique_ptr<ForceElementBuilder>                          forceElement;
    std::unique_ptr<ComputeGlobalsElementBuilder>                 computeGlobalsElement;
    std::unique_ptr<ParrinelloRahmanBarostatBuilder>              parrinelloRahmanBarostat;
    std::unique_ptr<VRescaleThermostatBuilder>                    vRescaleThermostat;
    std::unique_ptr<PropagatorBuilder<IntegrationStep::LeapFrog>> leapFrogPropagator;
    std::unique_ptr<PropagatorBuilder<IntegrationStep::VelocityVerletPositionsAndVelocities>> velocityVerletPropagator;
    std::unique_ptr<PropagatorBuilder<IntegrationStep::VelocitiesOnly>> velocityPropagator;
    //!}

    //! Get a self-consistent instance of the builder struct
    static ElementAndSignallerBuilders getInstance(ModularSimulator* simulator);

private:
    //! Constructor
    explicit ElementAndSignallerBuilders(ModularSimulator* simulator);
    //! Connect builders
    void connectBuilders();
};

} // namespace gmx
#endif // GROMACS_MODULARSIMULATOR_BUILDERS_H
