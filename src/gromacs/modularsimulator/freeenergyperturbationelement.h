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
 * \brief Declares the free energy perturbation element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#ifndef GMX_MODULARSIMULATOR_FREEENERGYPERTURBATIONELEMENT_H
#define GMX_MODULARSIMULATOR_FREEENERGYPERTURBATIONELEMENT_H

#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/real.h"

#include "modularsimulatorinterfaces.h"

struct t_inputrec;

namespace gmx
{
class CheckpointHelperBuilder;
struct ElementAndSignallerBuilders;
class MDAtoms;

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief The free energy perturbation element
 *
 * The lambda vector and the current FEP state are held by the
 * FreeEnergyPerturbationElement, offering access to its values via getter
 * functions. The FreeEnergyPerturbationElement does update the lambda
 * values during the simulation run if lambda is non-static. It does
 * implement the checkpointing client interface to save its current
 * state for restart.
 */
class FreeEnergyPerturbationElement final : public ISimulatorElement, public ICheckpointHelperClient
{
public:
    //! Get a view of the current lambda vector
    ArrayRef<real> lambdaView();
    //! Get a const view of the current lambda vector
    ArrayRef<const real> constLambdaView();
    //! Get the current FEP state
    int currentFEPState();

    //! Update lambda and mdatoms
    void scheduleTask(Step step, Time time, const RegisterRunFunctionPtr& registerRunFunction) override;

    //! No setup needed
    void elementSetup() override{};

    //! No teardown needed
    void elementTeardown() override{};

    //! Allow builder to do its job
    friend class FreeEnergyPerturbationElementBuilder;

private:
    //! Constructor
    FreeEnergyPerturbationElement(FILE* fplog, const t_inputrec* inputrec, MDAtoms* mdAtoms);

    //! ICheckpointHelperClient implementation
    void writeCheckpoint(t_state* localState, t_state* globalState) override;
    //! Update the lambda values
    void updateLambdas(Step step);

    //! The lambda vector
    std::array<real, efptNR> lambda_;
    //! The starting lambda vector
    std::array<double, efptNR> lambda0_;
    //! The current free energy state
    int currentFEPState_;

    //! Whether lambda values are non-static
    const bool lambdasChange_;

    //! Handles logging.
    FILE* fplog_;
    //! Contains user input mdp options.
    const t_inputrec* inputrec_;
    //! Atom parameters for this domain.
    MDAtoms* mdAtoms_;
};

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Builder for the free energy perturbation element
 */
class FreeEnergyPerturbationElementBuilder
{
public:
    //! Constructor, forwarding arguments to FreeEnergyPerturbationElement constructor
    template<typename... Args>
    explicit FreeEnergyPerturbationElementBuilder(Args&&... args);

    //! Connect with other builders (required)
    void connectWithBuilders(ElementAndSignallerBuilders* builders);

    //! Get (non-owning) pointer before element is built
    FreeEnergyPerturbationElement* getPointer();

    //! Return FreeEnergyPerturbationElement
    std::unique_ptr<FreeEnergyPerturbationElement> build();

    //! Destructor, make sure we didn't connect an element which won't exist anymore
    ~FreeEnergyPerturbationElementBuilder();

private:
    //! The element to be built
    std::unique_ptr<FreeEnergyPerturbationElement> element_ = nullptr;
    //! Whether we allow registration
    bool registrationPossible_ = false;
    //! Whether we have registered the element with the checkpoint helper
    bool registeredWithCheckpointHelper_ = false;

    //! Register element with CheckpointHelper (required)
    void registerWithCheckpointHelper(CheckpointHelperBuilder* checkpointHelperBuilder);
};

template<typename... Args>
FreeEnergyPerturbationElementBuilder::FreeEnergyPerturbationElementBuilder(Args&&... args)
{
    try
    {
        // NOLINTNEXTLINE(modernize-make-unique): make_unique does not work with private constructor
        element_ = std::unique_ptr<FreeEnergyPerturbationElement>(
                new FreeEnergyPerturbationElement(std::forward<Args>(args)...));
    }
    catch (ElementNotNeededException&)
    {
        element_ = nullptr;
    }
    // Element being nullptr is a valid state, nullptr (element is not built)
    // needs to be handled by the caller of build().
    registrationPossible_ = true;
}

} // namespace gmx

#endif // GMX_MODULARSIMULATOR_FREEENERGYPERTURBATIONELEMENT_H
