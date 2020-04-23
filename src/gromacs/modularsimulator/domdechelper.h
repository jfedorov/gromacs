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
 * \brief Declares the domain decomposition helper for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#ifndef GMX_MODULARSIMULATOR_DOMDECHELPER_H
#define GMX_MODULARSIMULATOR_DOMDECHELPER_H

#include "gromacs/utility/exceptions.h"

#include "modularsimulatorinterfaces.h"

struct gmx_localtop_t;
struct gmx_vsite_t;
struct gmx_wallcycle;
struct pull_t;
struct t_commrec;
struct t_forcerec;
struct t_inputrec;
struct t_nrnb;

namespace gmx
{
class ComputeGlobalsElementBuilder;
class Constraints;
struct ElementAndSignallerBuilders;
class ImdSession;
class MDAtoms;
class MDLogger;
class StatePropagatorData;
class TopologyHolder;

//! \addtogroup module_modularsimulator
//! \{

//! The function type allowing to request a check of the number of bonded interactions
typedef std::function<void()> CheckBondedInteractionsCallback;
//! Pointer to the function type allowing to request a check of the number of bonded interactions
typedef std::unique_ptr<CheckBondedInteractionsCallback> CheckBondedInteractionsCallbackPtr;

/*! \libinternal
 * \brief Infrastructure element responsible for domain decomposition
 *
 * This encapsulates the function call to domain decomposition, which is
 * important for performance but outside of the current scope of the modular
 * simulator project. This relies on legacy data structures for the state
 * and the topology.
 *
 * This element does not implement the ISimulatorElement interface, as
 * the Simulator is calling it explicitly between task queue population
 * steps. This allows elements to be aware of any changes before
 * deciding what functionality they need to run.
 */
class DomDecHelper final : public INeighborSearchSignallerClient
{
public:
    /*! \brief Run domain decomposition
     *
     * Does domain decomposition partitioning at neighbor searching steps
     *
     * @param step  The step number
     * @param time  The time
     */
    void run(Step step, Time time);

    /*! \brief The initial domain decomposition partitioning
     *
     */
    void setup();

    //! Allow builder to do its job
    friend class DomDecHelperBuilder;

private:
    //! Constructor
    DomDecHelper(bool            isVerbose,
                 int             verbosePrintInterval,
                 int             nstglobalcomm,
                 FILE*           fplog,
                 t_commrec*      cr,
                 const MDLogger& mdlog,
                 Constraints*    constr,
                 t_inputrec*     inputrec,
                 MDAtoms*        mdAtoms,
                 t_nrnb*         nrnb,
                 gmx_wallcycle*  wcycle,
                 t_forcerec*     fr,
                 gmx_vsite_t*    vsite,
                 ImdSession*     imdSession,
                 pull_t*         pull_work);

    //! INeighborSearchSignallerClient implementation
    SignallerCallbackPtr registerNSCallback() override;

    //! The next NS step
    Step nextNSStep_;
    //! Whether we're being verbose
    const bool isVerbose_;
    //! If we're being verbose, how often?
    const int verbosePrintInterval_;
    //! The global communication frequency
    const int nstglobalcomm_;

    //! Pointer to the micro state
    StatePropagatorData* statePropagatorData_;
    //! Pointer to the topology
    TopologyHolder* topologyHolder_;
    //! Pointer to the ComputeGlobalsHelper object - to ask for # of bonded interaction checking
    CheckBondedInteractionsCallbackPtr checkBondedInteractionsCallback_;

    // Access to ISimulator data
    //! Handles logging.
    FILE* fplog_;
    //! Handles communication.
    t_commrec* cr_;
    //! Handles logging.
    const MDLogger& mdlog_;
    //! Handles constraints.
    Constraints* constr_;
    //! Contains user input mdp options.
    t_inputrec* inputrec_;
    //! Atom parameters for this domain.
    MDAtoms* mdAtoms_;
    //! Manages flop accounting.
    t_nrnb* nrnb_;
    //! Manages wall cycle accounting.
    gmx_wallcycle* wcycle_;
    //! Parameters for force calculations.
    t_forcerec* fr_;
    //! Handles virtual sites.
    gmx_vsite_t* vsite_;
    //! The Interactive Molecular Dynamics session.
    ImdSession* imdSession_;
    //! The pull work object.
    pull_t* pull_work_;
};

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Builder for the domain decomposition helper
 */
class DomDecHelperBuilder
{
public:
    //! Constructor, forwarding arguments to DomDecHelper constructor
    template<typename... Args>
    explicit DomDecHelperBuilder(Args&&... args);

    //! Connect with other builders (required)
    void connectWithBuilders(ElementAndSignallerBuilders* builders);

    //! Return DomDecHelper
    std::unique_ptr<DomDecHelper> build();

    //! Destructor, make sure we didn't connect an element which won't exist anymore
    ~DomDecHelperBuilder();

private:
    //! The element to be built
    std::unique_ptr<DomDecHelper> domDecHelper_ = nullptr;
    //! Whether we allow registration
    bool registrationPossible_ = false;
    //! Whether we have registered the element with the neighbor search signaller
    bool registeredWithNeighborSearchSignaller_ = false;

    //! Set pointer to StatePropagatorData valid throughout the simulation (required)
    void setStatePropagatorData(StatePropagatorData* statePropagatorData);
    //! Set pointer to TopologyHolder valid throughout the simulation (required)
    void setTopologyHolder(TopologyHolder* topologyHolder);
    //! Allow DomDecHelperBuilder to get CheckNumberOfBondedInteractionsCallback (required)
    void setComputeGlobalsElementBuilder(ComputeGlobalsElementBuilder* computeGlobalsElementBuilder);
    //! Register element with NeighborSearchSignaller (required)
    void registerWithNeighborSearchSignaller(SignallerBuilder<NeighborSearchSignaller>* signallerBuilder);
};

template<typename... Args>
DomDecHelperBuilder::DomDecHelperBuilder(Args&&... args)
{
    try
    {
        domDecHelper_ =
                // NOLINTNEXTLINE(modernize-make-unique): make_unique does not work with private constructor
                std::unique_ptr<DomDecHelper>(new DomDecHelper(std::forward<Args>(args)...));
    }
    catch (ElementNotNeededException&)
    {
        domDecHelper_ = nullptr;
    }
    // Element being nullptr is a valid state, nullptr (element is not built)
    // needs to be handled by the caller of build().
    registrationPossible_ = true;
}

//! \}
} // namespace gmx

#endif // GMX_MODULARSIMULATOR_DOMDECHELPER_H
