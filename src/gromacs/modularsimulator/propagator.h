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
/*! \libinternal \file
 * \brief Declares the propagator element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#ifndef GMX_MODULARSIMULATOR_PROPAGATOR_H
#define GMX_MODULARSIMULATOR_PROPAGATOR_H

#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

#include "modularsimulatorinterfaces.h"

struct gmx_wallcycle;

namespace gmx
{
struct ElementAndSignallerBuilders;
class MDAtoms;
class StatePropagatorData;

//! \addtogroup module_modularsimulator
//! \{

/*! \brief The different integration types we know about
 *
 * PositionsOnly:
 *   Moves the position vector by the given time step
 * VelocitiesOnly:
 *   Moves the velocity vector by the given time step
 * LeapFrog:
 *   Is a manual fusion of the previous two propagators
 * VelocityVerletPositionsAndVelocities:
 *   Is a manual fusion of VelocitiesOnly and PositionsOnly,
 *   where VelocitiesOnly is only propagated by half the
 *   time step of the positions.
 */
enum class IntegrationStep
{
    PositionsOnly,
    VelocitiesOnly,
    LeapFrog,
    VelocityVerletPositionsAndVelocities,
    Count
};

//! Sets the number of different velocity scaling values
enum class NumVelocityScalingValues
{
    None,     //!< No velocity scaling (either this step or ever)
    Single,   //!< Single T-scaling value (either one group or all values =1)
    Multiple, //!< Multiple T-scaling values, need to use T-group indices
    Count
};

//! Sets the type of Parrinello-Rahman pressure scaling
enum class ParrinelloRahmanVelocityScaling
{
    No,       //!< Do not apply velocity scaling (not a PR-coupling run or step)
    Diagonal, //!< Apply velocity scaling using a diagonal matrix
    Full,     //!< Apply velocity scaling using a full matrix
    Count
};


template<IntegrationStep algorithm>
class PropagatorBuilder;

//! Generic callback to the propagator
typedef std::function<void(Step)> PropagatorCallback;
//! Pointer to generic callback to the propagator
typedef std::unique_ptr<PropagatorCallback> PropagatorCallbackPtr;

/*! \libinternal
 * \brief Propagator element
 *
 * The propagator element can, through templating, cover the different
 * propagation types used in NVE MD. The combination of templating, static
 * functions, and having only the inner-most operations in the static
 * functions allows to have performance comparable to fused update elements
 * while keeping easily re-orderable single instructions.
 *
 * @tparam algorithm  The integration type
 */
template<IntegrationStep algorithm>
class Propagator final : public ISimulatorElement
{
public:
    /*! \brief Register run function for step / time
     *
     * @param step                 The step number
     * @param time                 The time
     * @param registerRunFunction  Function allowing to register a run function
     */
    void scheduleTask(Step step, Time time, const RegisterRunFunctionPtr& registerRunFunction) override;

    //! No element setup needed
    void elementSetup() override {}
    //! No element teardown needed
    void elementTeardown() override {}

    //! Allow builder to do its job
    friend class PropagatorBuilder<algorithm>;

private:
    //! Constructor
    Propagator(double timestep, const MDAtoms* mdAtoms, gmx_wallcycle* wcycle);

    //! The actual propagation
    template<NumVelocityScalingValues numVelocityScalingValues, ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling>
    void run();

    //! The time step
    const real timestep_;

    //! Pointer to the micro state
    StatePropagatorData* statePropagatorData_;

    //! Whether we're doing single-value velocity scaling
    bool doSingleVelocityScaling;
    //! Wether we're doing group-wise velocity scaling
    bool doGroupVelocityScaling;
    //! The vector of velocity scaling values
    std::vector<real> velocityScaling_;
    //! The next velocity scaling step
    Step scalingStepVelocity_;

    //! The diagonal of the PR scaling matrix
    rvec diagPR;
    //! The full PR scaling matrix
    matrix matrixPR;
    //! The next PR scaling step
    Step scalingStepPR_;

    // Access to ISimulator data
    //! Atom parameters for this domain.
    const MDAtoms* mdAtoms_;
    //! Manages wall cycle accounting.
    gmx_wallcycle* wcycle_;
};

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Builder for the propagator element
 *
 * @tparam algorithm  The integration type
 */
template<IntegrationStep algorithm>
class PropagatorBuilder
{
public:
    //! Constructor, forwarding arguments to Propagator constructor
    template<typename... Args>
    explicit PropagatorBuilder(Args&&... args);

    //! Connect with other builders (required)
    void connectWithBuilders(ElementAndSignallerBuilders* builders);

    //! Set the number of velocity scaling variables
    void setNumVelocityScalingVariables(int numVelocityScalingVariables);
    //! Get view on the velocity scaling vector
    ArrayRef<real> viewOnVelocityScaling();
    //! Get velocity scaling callback
    PropagatorCallbackPtr velocityScalingCallback();

    //! Get view on the full PR scaling matrix
    ArrayRef<rvec> viewOnPRScalingMatrix();
    //! Get PR scaling callback
    PropagatorCallbackPtr prScalingCallback();

    //! Return Propagator
    std::unique_ptr<Propagator<algorithm>> build();

    //! Destructor, make sure we didn't connect an element which won't exist anymore
    ~PropagatorBuilder();

private:
    //! The element to be built
    std::unique_ptr<Propagator<algorithm>> propagator_ = nullptr;

    //! Set pointer to StatePropagatorData valid throughout the simulation (required)
    void setStatePropagatorData(StatePropagatorData* statePropagatorData);
};

template<IntegrationStep algorithm>
template<typename... Args>
PropagatorBuilder<algorithm>::PropagatorBuilder(Args&&... args)
{
    // NOLINTNEXTLINE(modernize-make-unique): make_unique does not work with private constructor
    propagator_ = std::unique_ptr<Propagator<algorithm>>(
            new Propagator<algorithm>(std::forward<Args>(args)...));
}

//! \}
} // namespace gmx

#endif // GMX_MODULARSIMULATOR_PROPAGATOR_H
