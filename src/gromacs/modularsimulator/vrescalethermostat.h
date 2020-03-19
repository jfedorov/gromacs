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
 * \brief Declares the v-rescale thermostat for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#ifndef GMX_MODULARSIMULATOR_VRESCALETHERMOSTAT_H
#define GMX_MODULARSIMULATOR_VRESCALETHERMOSTAT_H

#include "gromacs/utility/arrayref.h"

#include "energyelement.h"
#include "modularsimulatorinterfaces.h"
#include "propagator.h"

struct t_commrec;

namespace gmx
{
class CheckpointHelperBuilder;

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Element implementing the v-rescale thermostat
 *
 * This element takes a callback to the propagator and updates the velocity
 * scaling factor according to the v-rescale thermostat.
 */
class VRescaleThermostat final : public ISimulatorElement, public ICheckpointHelperClient
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

    //! Getter for the thermostatIntegral
    const std::vector<double>& thermostatIntegral() const;

    //! Allow builder to do its job
    friend class VRescaleThermostatBuilder;

private:
    //! Constructor
    VRescaleThermostat(int            nstcouple,
                       int64_t        seed,
                       int            numTemperatureGroups,
                       double         couplingTimeStep,
                       const real*    referenceTemperature,
                       const real*    couplingTime,
                       const real*    numDegreesOfFreedom,
                       const t_state* globalState,
                       t_commrec*     cr,
                       bool           isRestart,
                       int            inputThermostatType);

    //! The frequency at which the thermostat is applied
    const int nstcouple_;
    //! If != 0, offset the step at which the thermostat is applied
    int offset_;
    //! Whether we're using full step kinetic energy
    bool useFullStepKE_;
    //! The random seed
    const int64_t seed_;

    //! The number of temperature groups
    const int numTemperatureGroups_;
    //! The coupling time step - simulation time step x nstcouple_
    const double couplingTimeStep_;
    //! Coupling temperature per group
    const std::vector<real> referenceTemperature_;
    //! Coupling time per group
    const std::vector<real> couplingTime_;
    //! Number of degrees of freedom per group
    const std::vector<real> numDegreesOfFreedom_;
    //! Work exerted by thermostat
    std::vector<double> thermostatIntegral_;

    //! Pointer to the energy element (for ekindata)
    EnergyElement* energyElement_;

    //! View on the scaling factor of the propagator
    ArrayRef<real> lambda_;
    //! Callback to let propagator know that we updated lambda
    PropagatorCallbackPtr propagatorCallback_;

    //! Set new lambda value (at T-coupling steps)
    void setLambda(Step step);

    //! ICheckpointHelperClient implementation
    void writeCheckpoint(t_state* localState, t_state* globalState) override;
};

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Builder for the v-rescale thermostat element
 */
class VRescaleThermostatBuilder
{
public:
    //! Constructor, forwarding arguments to VRescaleThermostat constructor
    template<typename... Args>
    explicit VRescaleThermostatBuilder(Args&&... args);

    //! Set pointer to EnergyElement valid throughout the simulation (required)
    void setEnergyElement(EnergyElement* energyElement);

    //! Register element with CheckpointHelper (required)
    void registerWithCheckpointHelper(CheckpointHelperBuilder* checkpointHelperBuilder);

    //! Set propagator
    template<IntegrationStep integrationStep>
    void setPropagator(Propagator<integrationStep>* propagator);

    /*! \brief Return VRescaleThermostat
     *
     * @param offset         Offset the step at which the thermostat is applied
     * @param useFullStepKE  Whether we're using full step kinetic energy
     */
    std::unique_ptr<VRescaleThermostat> build(int offset, bool useFullStepKE);

    //! Destructor, make sure we didn't connect an element which won't exist anymore
    ~VRescaleThermostatBuilder();

private:
    //! The element to be built
    std::unique_ptr<VRescaleThermostat> vrThermostat_ = nullptr;
    //! Whether we have an element that we can move
    bool registrationPossible_ = false;
    //! Whether we have registered the element with the checkpoint helper
    bool registeredWithCheckpointHelper_ = false;
    //! Whether we have registered the element with a propagator
    bool registeredWithPropagator_ = false;
};

template<typename... Args>
VRescaleThermostatBuilder::VRescaleThermostatBuilder(Args&&... args)
{
    try
    {
        vrThermostat_ =
                // NOLINTNEXTLINE(modernize-make-unique): make_unique does not work with private constructor
                std::unique_ptr<VRescaleThermostat>(new VRescaleThermostat(std::forward<Args>(args)...));
    }
    catch (ElementNotNeededException&)
    {
        vrThermostat_ = nullptr;
    }
    // Element being nullptr is a valid state, nullptr (element is not built)
    // needs to be handled by the caller of build().
    registrationPossible_ = true;
}

template<IntegrationStep integrationStep>
void VRescaleThermostatBuilder::setPropagator(Propagator<integrationStep>* propagator)
{

    GMX_RELEASE_ASSERT(registrationPossible_,
                       "Tried to set propagator after v-rescale thermostat was built.");
    GMX_RELEASE_ASSERT(!registeredWithPropagator_,
                       "Tried to set propagator of v-rescale thermostat more than once.");
    if (vrThermostat_)
    {
        // TODO: With increased complexity of the propagator, this will need further development,
        //       e.g. using propagators templated for velocity propagation policies
        propagator->setNumVelocityScalingVariables(vrThermostat_->numTemperatureGroups_);
        vrThermostat_->lambda_             = propagator->viewOnVelocityScaling();
        vrThermostat_->propagatorCallback_ = propagator->velocityScalingCallback();
    }
    registeredWithPropagator_ = true;
}

} // namespace gmx

#endif // GMX_MODULARSIMULATOR_VRESCALETHERMOSTAT_H
