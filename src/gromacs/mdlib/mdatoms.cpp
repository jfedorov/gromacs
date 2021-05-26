/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2012,2013,2014,2015,2016 by the GROMACS development team.
 * Copyright (c) 2017,2018,2019,2020,2021, by the GROMACS development team, led by
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
#include "gmxpre.h"

#include "mdatoms.h"

#include <cmath>

#include <memory>
#include <vector>

#include "gromacs/ewald/pme.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/paddedvector.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/topology/mtop_lookup.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/allocator.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/smalloc.h"

#define ALMOST_ZERO 1e-30

namespace gmx
{

class MDAtoms::Impl
{
public:
    //! Memory for chargeA that can be set up for efficient GPU transfer.
    PaddedHostVector<real> chargeA;
    //! Memory for chargeB that can be set up for efficient GPU transfer.
    PaddedHostVector<real> chargeB;
    //! Total mass in state A
    double tmassA = 0.0;
    //! Total mass in state B
    double tmassB = 0.0;
    //! Total mass
    double tmass = 0.0;
    //! Number of atoms in arrays
    int nr = 0;
    //! Number of energy groups
    int nenergrp = 0;
    //! Do we have multiple center of mass motion removal groups
    bool bVCMgrps = false;
    //! Do we have any virtual sites?
    bool haveVsites = false;
    //! Do we have atoms that are frozen along 1 or 2 (not 3) dimensions?
    bool havePartiallyFrozenAtoms = false;
    //! Number of perturbed atoms
    int nPerturbed = 0;
    //! Number of atoms for which the mass is perturbed
    int nMassPerturbed = 0;
    //! Number of atoms for which the charge is perturbed
    int nChargePerturbed = 0;
    //! Number of atoms for which the type is perturbed
    int nTypePerturbed = 0;
    //! Do we have orientation restraints
    bool bOrires = false;
    //! Atomic mass in A state
    std::vector<real> massA;
    //! Atomic mass in B state
    std::vector<real> massB;
    //! Atomic mass in present state
    std::vector<real> massT;
    //! Inverse atomic mass per atom, 0 for vsites and shells
    PaddedVector<real, Allocator<real, AlignedAllocationPolicy>> invmass;
    //! Inverse atomic mass per atom and dimension, 0 for vsites, shells and frozen dimensions
    std::vector<RVec> invMassPerDim;
    //! Dispersion constant C6 in A state
    std::vector<real> sqrt_c6A;
    //! Dispersion constant C6 in A state
    std::vector<real> sqrt_c6B;
    //! Van der Waals radius sigma in the A state
    std::vector<real> sigmaA;
    //! Van der Waals radius sigma in the B state
    std::vector<real> sigmaB;
    //! Van der Waals radius sigma^3 in the A state
    std::vector<real> sigma3A;
    //! Van der Waals radius sigma^3 in the B state
    std::vector<real> sigma3B;
    //! Is this atom perturbed
    std::vector<bool> bPerturbed;
    //! Type of atom in the A state
    std::vector<int> typeA;
    //! Type of atom in the B state
    std::vector<int> typeB;
    //! Particle type
    std::vector<ParticleType> ptype;
    //! Group index for temperature coupling
    std::vector<unsigned short> cTC;
    //! Group index for energy matrix
    std::vector<unsigned short> cENER;
    //! Group index for acceleration
    std::vector<unsigned short> cACC;
    //! Group index for freezing
    std::vector<unsigned short> cFREEZE;
    //! Group index for center of mass motion removal
    std::vector<unsigned short> cVCM;
    //! Group index for user 1
    std::vector<unsigned short> cU1;
    //! Group index for user 2
    std::vector<unsigned short> cU2;
    //! Group index for orientation restraints
    std::vector<unsigned short> cORF;
    //! Number of atoms on this processor.
    int homenr = 0;
    //! The lambda value used to create the contents of the struct
    real lambda = 0.0;
};

MDAtoms::MDAtoms() : impl_(std::make_unique<MDAtoms::Impl>()) {}

MDAtoms::~MDAtoms() {}

void MDAtoms::resizeChargeA(int newSize)
{
    impl_->chargeA.resizeWithPadding(newSize);
}

void MDAtoms::resizeChargeB(const int newSize)
{
    impl_->chargeB.resizeWithPadding(newSize);
}

int MDAtoms::size() const
{
    return impl_->nr;
}

int MDAtoms::homenr() const
{
    return impl_->homenr;
}

int MDAtoms::nenergrp() const
{
    return impl_->nenergrp;
}

int MDAtoms::nChargePerturbed() const
{
    return impl_->nChargePerturbed;
}

int MDAtoms::nTypePerturbed() const
{
    return impl_->nTypePerturbed;
}

double MDAtoms::tmass() const
{
    return impl_->tmass;
}

real MDAtoms::lambda() const
{
    return impl_->lambda;
}

bool MDAtoms::havePerturbedCharges() const
{
    return impl_->nChargePerturbed != 0;
}

bool MDAtoms::havePerturbedMasses() const
{
    return impl_->nMassPerturbed != 0;
}

bool MDAtoms::havePerturbedTypes() const
{
    return impl_->nTypePerturbed != 0;
}

bool MDAtoms::havePerturbed() const
{
    return impl_->nPerturbed != 0;
}

int MDAtoms::numPerturbed() const
{
    return impl_->nPerturbed;
}

bool MDAtoms::haveVsites() const
{
    return impl_->haveVsites;
}

bool MDAtoms::havePartiallyFrozenAtoms() const
{
    return impl_->havePartiallyFrozenAtoms;
}

ArrayRef<const real> MDAtoms::massA() const
{
    return impl_->massA;
}

ArrayRef<const real> MDAtoms::massB() const
{
    return impl_->massB;
}

ArrayRef<const real> MDAtoms::massT() const
{
    return impl_->massT;
}

ArrayRefWithPadding<const real> MDAtoms::invmass() const
{
    return impl_->invmass;
}

ArrayRef<const RVec> MDAtoms::invMassPerDim() const
{
    return impl_->invMassPerDim;
}

ArrayRef<const real> MDAtoms::chargeA() const
{
    return impl_->chargeA.constArrayRefWithPadding().paddedConstArrayRef();
}

ArrayRef<const real> MDAtoms::chargeB() const
{
    return impl_->chargeB.constArrayRefWithPadding().paddedConstArrayRef();
}

ArrayRef<const real> MDAtoms::sqrt_c6A() const
{
    return impl_->sqrt_c6A;
}

ArrayRef<const real> MDAtoms::sqrt_c6B() const
{
    return impl_->sqrt_c6B;
}

ArrayRef<const real> MDAtoms::sigmaA() const
{
    return impl_->sigmaA;
}

ArrayRef<const real> MDAtoms::sigmaB() const
{
    return impl_->sigmaB;
}

ArrayRef<const real> MDAtoms::sigma3A() const
{
    return impl_->sigma3A;
}

ArrayRef<const real> MDAtoms::sigma3B() const
{

    return impl_->sigma3B;
}

const std::vector<bool>& MDAtoms::bPerturbed() const
{
    return impl_->bPerturbed;
}

ArrayRef<const int> MDAtoms::typeA() const
{
    return impl_->typeA;
}

ArrayRef<const int> MDAtoms::typeB() const
{
    return impl_->typeB;
}

ArrayRef<const ParticleType> MDAtoms::ptype() const
{
    return impl_->ptype;
}

ArrayRef<const unsigned short> MDAtoms::cTC() const
{
    return impl_->cTC;
}

ArrayRef<const unsigned short> MDAtoms::cENER() const
{
    return impl_->cENER;
}

ArrayRef<const unsigned short> MDAtoms::cACC() const
{
    return impl_->cACC;
}

ArrayRef<const unsigned short> MDAtoms::cFREEZE() const
{
    return impl_->cFREEZE;
}

ArrayRef<const unsigned short> MDAtoms::cVCM() const
{
    return impl_->cVCM;
}

ArrayRef<const unsigned short> MDAtoms::cU1() const
{
    return impl_->cU1;
}

ArrayRef<const unsigned short> MDAtoms::cU2() const
{
    return impl_->cU2;
}

ArrayRef<const unsigned short> MDAtoms::cORF() const
{
    return impl_->cORF;
}


std::unique_ptr<MDAtoms> makeMDAtoms(FILE* fp, const gmx_mtop_t& mtop, const t_inputrec& ir, const bool rankHasPmeGpuTask)
{
    std::unique_ptr<MDAtoms> mdAtoms(new MDAtoms);
    // GPU transfers may want to use a suitable pinning mode.
    if (rankHasPmeGpuTask)
    {
        changePinningPolicy(&mdAtoms->impl_->chargeA, pme_get_pinning_policy());
        changePinningPolicy(&mdAtoms->impl_->chargeB, pme_get_pinning_policy());
    }
    mdAtoms->impl_->nenergrp = mtop.groups.groups[SimulationAtomGroupType::EnergyOutput].size();
    for (int i = 0; i < mtop.natoms && !mdAtoms->impl_->bVCMgrps; i++)
    {
        if (getGroupType(mtop.groups, SimulationAtomGroupType::MassCenterVelocityRemoval, i) > 0)
        {
            mdAtoms->impl_->bVCMgrps = true;
        }
    }

    /* Determine the total system mass and perturbed atom counts */
    mdAtoms->impl_->haveVsites      = false;
    gmx_mtop_atomloop_block_t aloop = gmx_mtop_atomloop_block_init(mtop);
    const t_atom*             atom;
    int                       nmol;
    while (gmx_mtop_atomloop_block_next(aloop, &atom, &nmol))
    {
        mdAtoms->impl_->tmassA += nmol * atom->m;
        mdAtoms->impl_->tmassB += nmol * atom->mB;

        if (atom->ptype == ParticleType::VSite)
        {
            mdAtoms->impl_->haveVsites = true;
        }

        if (ir.efep != FreeEnergyPerturbationType::No && PERTURBED(*atom))
        {
            mdAtoms->impl_->nPerturbed++;
            if (atom->mB != atom->m)
            {
                mdAtoms->impl_->nMassPerturbed += nmol;
            }
            if (atom->qB != atom->q)
            {
                mdAtoms->impl_->nChargePerturbed += nmol;
            }
            if (atom->typeB != atom->type)
            {
                mdAtoms->impl_->nTypePerturbed += nmol;
            }
        }
    }

    if (ir.efep != FreeEnergyPerturbationType::No && fp)
    {
        fprintf(fp,
                "There are %d atoms and %d charges for free energy perturbation\n",
                mdAtoms->impl_->nPerturbed,
                mdAtoms->impl_->nChargePerturbed);
    }

    for (int g = 0; g < ir.opts.ngfrz && !mdAtoms->impl_->havePartiallyFrozenAtoms; g++)
    {
        for (int d = YY; d < DIM; d++)
        {
            if (ir.opts.nFreeze[g][d] != ir.opts.nFreeze[g][XX])
            {
                mdAtoms->impl_->havePartiallyFrozenAtoms = true;
            }
        }
    }

    mdAtoms->impl_->bOrires = (gmx_mtop_ftype_count(mtop, F_ORIRES) != 0);

    return mdAtoms;
}

void MDAtoms::reinitialize(const gmx_mtop_t& mtop, const t_inputrec& inputrec, ArrayRef<const int> index, int homenr)
{
    int nthreads gmx_unused;

    const bool bLJPME = EVDW_PME(inputrec.vdwtype);

    const t_grpopts& opts = inputrec.opts;

    const SimulationGroups& groups = mtop.groups;

    const int newSize = index.empty() ? mtop.natoms : index.size();

    if (newSize > static_cast<int>(size()))
    {
        const int newAllocationSize = over_alloc_dd(newSize);
        impl_->nr                   = newSize;

        if (havePerturbedMasses())
        {
            impl_->massA.resize(newAllocationSize);
            impl_->massB.resize(newAllocationSize);
        }
        impl_->massT.resize(newAllocationSize);
        /* The SIMD version of the integrator needs this aligned and padded.
         * The padding needs to be with zeros, which we set later below.
         */
        impl_->invmass.resizeWithPadding(newAllocationSize);
        impl_->invMassPerDim.resize(newAllocationSize);
        resizeChargeA(newAllocationSize);
        impl_->typeA.resize(newAllocationSize);
        if (havePerturbed())
        {
            resizeChargeB(newAllocationSize);
            impl_->typeB.resize(newAllocationSize);
        }
        if (bLJPME)
        {
            impl_->sqrt_c6A.resize(newAllocationSize);
            impl_->sigmaA.resize(newAllocationSize);
            impl_->sigma3A.resize(newAllocationSize);
            if (havePerturbed())
            {
                impl_->sqrt_c6B.resize(newAllocationSize);
                impl_->sigmaB.resize(newAllocationSize);
                impl_->sigma3B.resize(newAllocationSize);
            }
        }
        impl_->ptype.resize(newAllocationSize);
        if (opts.ngtc > 1)
        {
            impl_->cTC.resize(newAllocationSize);
            /* We always copy cTC with domain decomposition */
        }
        impl_->cENER.resize(newAllocationSize);
        if (inputrecFrozenAtoms(&inputrec))
        {
            impl_->cFREEZE.resize(newAllocationSize);
        }
        if (impl_->bVCMgrps)
        {
            impl_->cVCM.resize(newAllocationSize);
        }
        if (impl_->bOrires)
        {
            impl_->cORF.resize(newAllocationSize);
        }
        if (havePerturbed())
        {
            impl_->bPerturbed.resize(newAllocationSize);
        }

        /* Note that these user t_mdatoms array pointers are NULL
         * when there is only one group present.
         * Therefore, when adding code, the user should use something like:
         * gprnrU1 = (md->cU1==NULL ? 0 : md->cU1[localatindex])
         */
        if (!mtop.groups.groupNumbers[SimulationAtomGroupType::User1].empty())
        {
            impl_->cU1.resize(newAllocationSize);
        }
        if (!mtop.groups.groupNumbers[SimulationAtomGroupType::User2].empty())
        {
            impl_->cU2.resize(newAllocationSize);
        }
    }

    int molb = 0;

    nthreads = gmx_omp_nthreads_get(ModuleMultiThread::Default);
#pragma omp parallel for num_threads(nthreads) schedule(static) firstprivate(molb)
    for (int i = 0; i < impl_->nr; i++)
    {
        try
        {
            real mA, mB, fac;

            const int     ag   = index.empty() ? i : index[i];
            const t_atom& atom = mtopGetAtomParameters(mtop, ag, &molb);

            if (!impl_->cFREEZE.empty())
            {
                impl_->cFREEZE[i] = getGroupType(groups, SimulationAtomGroupType::Freeze, ag);
            }
            if (EI_ENERGY_MINIMIZATION(inputrec.eI))
            {
                /* Displacement is proportional to F, masses used for constraints */
                mA = 1.0;
                mB = 1.0;
            }
            else if (inputrec.eI == IntegrationAlgorithm::BD)
            {
                /* With BD the physical masses are irrelevant.
                 * To keep the code simple we use most of the normal MD code path
                 * for BD. Thus for constraining the masses should be proportional
                 * to the friction coefficient. We set the absolute value such that
                 * m/2<(dx/dt)^2> = m/2*2kT/fric*dt = kT/2 => m=fric*dt/2
                 * Then if we set the (meaningless) velocity to v=dx/dt, we get the
                 * correct kinetic energy and temperature using the usual code path.
                 * Thus with BD v*dt will give the displacement and the reported
                 * temperature can signal bad integration (too large time step).
                 */
                if (inputrec.bd_fric > 0)
                {
                    mA = 0.5 * inputrec.bd_fric * inputrec.delta_t;
                    mB = 0.5 * inputrec.bd_fric * inputrec.delta_t;
                }
                else
                {
                    /* The friction coefficient is mass/tau_t */
                    fac = inputrec.delta_t
                          / opts.tau_t[impl_->cTC.empty() ? 0 : groups.groupNumbers[SimulationAtomGroupType::TemperatureCoupling][ag]];
                    mA = 0.5 * atom.m * fac;
                    mB = 0.5 * atom.mB * fac;
                }
            }
            else
            {
                mA = atom.m;
                mB = atom.mB;
            }
            if (havePerturbedMasses())
            {
                impl_->massA[i] = mA;
                impl_->massB[i] = mB;
            }
            impl_->massT[i] = mA;

            if (mA == 0.0)
            {
                impl_->invmass[i]           = 0;
                impl_->invMassPerDim[i][XX] = 0;
                impl_->invMassPerDim[i][YY] = 0;
                impl_->invMassPerDim[i][ZZ] = 0;
            }
            else if (!impl_->cFREEZE.empty())
            {
                const int g = impl_->cFREEZE[i];
                GMX_ASSERT(opts.nFreeze != nullptr, "Must have freeze groups to initialize masses");
                if (opts.nFreeze[g][XX] && opts.nFreeze[g][YY] && opts.nFreeze[g][ZZ])
                {
                    /* Set the mass of completely frozen particles to ALMOST_ZERO
                     * iso 0 to avoid div by zero in lincs or shake.
                     */
                    // TODO LINCS and SHAKE should assert on zero instead of using hacks like this!
                    impl_->invmass[i] = ALMOST_ZERO;
                }
                else
                {
                    /* Note: Partially frozen particles use the normal invmass.
                     * If such particles are constrained, the frozen dimensions
                     * should not be updated with the constrained coordinates.
                     */
                    impl_->invmass[i] = 1.0 / mA;
                }
                for (int d = 0; d < DIM; d++)
                {
                    impl_->invMassPerDim[i][d] = (opts.nFreeze[g][d] ? 0 : 1.0 / mA);
                }
            }
            else
            {
                impl_->invmass[i] = 1.0 / mA;
                for (int d = 0; d < DIM; d++)
                {
                    impl_->invMassPerDim[i][d] = 1.0 / mA;
                }
            }

            impl_->chargeA[i] = atom.q;
            impl_->typeA[i]   = atom.type;
            if (bLJPME)
            {
                const real c6  = mtop.ffparams.iparams[atom.type * (mtop.ffparams.atnr + 1)].lj.c6;
                const real c12 = mtop.ffparams.iparams[atom.type * (mtop.ffparams.atnr + 1)].lj.c12;
                impl_->sqrt_c6A[i] = std::sqrt(c6);
                if (c6 == 0.0 || c12 == 0)
                {
                    impl_->sigmaA[i] = 1.0;
                }
                else
                {
                    impl_->sigmaA[i] = gmx::sixthroot(c12 / c6);
                }
                impl_->sigma3A[i] = 1 / (impl_->sigmaA[i] * impl_->sigmaA[i] * impl_->sigmaA[i]);
            }
            if (havePerturbed())
            {
                impl_->bPerturbed[i] = PERTURBED(atom);
                impl_->chargeB[i]    = atom.qB;
                impl_->typeB[i]      = atom.typeB;
                if (bLJPME)
                {
                    const real c6 = mtop.ffparams.iparams[atom.typeB * (mtop.ffparams.atnr + 1)].lj.c6;
                    const real c12 = mtop.ffparams.iparams[atom.typeB * (mtop.ffparams.atnr + 1)].lj.c12;
                    impl_->sqrt_c6B[i] = std::sqrt(c6);
                    if (c6 == 0.0 || c12 == 0)
                    {
                        impl_->sigmaB[i] = 1.0;
                    }
                    else
                    {
                        impl_->sigmaB[i] = gmx::sixthroot(c12 / c6);
                    }
                    impl_->sigma3B[i] = 1 / (impl_->sigmaB[i] * impl_->sigmaB[i] * impl_->sigmaB[i]);
                }
            }
            impl_->ptype[i] = atom.ptype;
            if (!impl_->cTC.empty())
            {
                impl_->cTC[i] = groups.groupNumbers[SimulationAtomGroupType::TemperatureCoupling][ag];
            }
            impl_->cENER[i] = getGroupType(groups, SimulationAtomGroupType::EnergyOutput, ag);
            if (!impl_->cVCM.empty())
            {
                impl_->cVCM[i] =
                        groups.groupNumbers[SimulationAtomGroupType::MassCenterVelocityRemoval][ag];
            }
            if (!impl_->cORF.empty())
            {
                impl_->cORF[i] =
                        getGroupType(groups, SimulationAtomGroupType::OrientationRestraintsFit, ag);
            }

            if (!impl_->cU1.empty())
            {
                impl_->cU1[i] = groups.groupNumbers[SimulationAtomGroupType::User1][ag];
            }
            if (!impl_->cU2.empty())
            {
                impl_->cU2[i] = groups.groupNumbers[SimulationAtomGroupType::User2][ag];
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }

    if (size() > 0)
    {
        /* Pad invmass with 0 so a SIMD MD update does not change v and x */
        for (auto invmass = impl_->invmass.begin() + size(); invmass != impl_->invmass.end(); ++invmass)
        {
            *invmass = 0;
        }
    }

    impl_->homenr = homenr;
    /* We set mass, invmass, invMassPerDim and tmass for lambda=0.
     * For free-energy runs, these should be updated using update_mdatoms().
     */
    impl_->tmass  = impl_->tmassA;
    impl_->lambda = 0;
}

void MDAtoms::adjustToLambda(real lambda)
{
    if (havePerturbedMasses() && lambda != impl_->lambda)
    {
        real L1 = 1 - lambda;

        /* Update masses of perturbed atoms for the change in lambda */
        int gmx_unused nthreads = gmx_omp_nthreads_get(ModuleMultiThread::Default);
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int i = 0; i < size(); i++)
        {
            if (impl_->bPerturbed[i])
            {
                impl_->massT[i] = L1 * impl_->massA[i] + lambda * impl_->massB[i];
                /* Atoms with invmass 0 or ALMOST_ZERO are massless or frozen
                 * and their invmass does not depend on lambda.
                 */
                if (impl_->invmass[i] > 1.1 * ALMOST_ZERO)
                {
                    impl_->invmass[i] = 1.0 / impl_->massT[i];
                    for (int d = 0; d < DIM; d++)
                    {
                        if (impl_->invMassPerDim[i][d] > 1.1 * ALMOST_ZERO)
                        {
                            impl_->invMassPerDim[i][d] = impl_->invmass[i];
                        }
                    }
                }
            }
        }

        /* Update the system mass for the change in lambda */
        impl_->tmass = L1 * impl_->tmassA + lambda * impl_->tmassB;
    }

    impl_->lambda = lambda;
}

} // namespace gmx
