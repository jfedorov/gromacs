/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017 by the GROMACS development team.
 * Copyright (c) 2018,2019,2020,2021, by the GROMACS development team, led by
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

#include "nb_free_energy.h"

#include "config.h"

#include <cmath>

#include <algorithm>

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gmxlib/nonbonded/nonbonded.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/nblist.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/arrayref.h"


#define STATE_A 0
#define STATE_B 1
#define NSTATES 2


//! Scalar (non-SIMD) data types.
struct ScalarDataTypes
{
    using RealType                     = real; //!< The data type to use as real.
    using IntType                      = int;  //!< The data type to use as int.
    static constexpr int simdRealWidth = 1;    //!< The width of the RealType.
    static constexpr int simdIntWidth  = 1;    //!< The width of the IntType.
};

#if GMX_SIMD_HAVE_REAL && GMX_SIMD_HAVE_INT32_ARITHMETICS
//! SIMD data types.
struct SimdDataTypes
{
    using RealType                     = gmx::SimdReal;         //!< The data type to use as real.
    using IntType                      = gmx::SimdInt32;        //!< The data type to use as int.
    static constexpr int simdRealWidth = GMX_SIMD_REAL_WIDTH;   //!< The width of the RealType.
    static constexpr int simdIntWidth  = GMX_SIMD_FINT32_WIDTH; //!< The width of the IntType.
};
#endif


class SoftcoreCalculatorShared
{
public:
    //! Computes r^(1/p) and 1/r^(1/p) for the standard p=6
    template<class RealType>
    void pthRoot(const RealType r, RealType& pthRoot, RealType& invPthRoot)
    {
        invPthRoot = gmx::invsqrt(std::cbrt(r));
        pthRoot    = 1 / invPthRoot;
    }

    template<class RealType>
    RealType calculateRinv6(const RealType rInvV)
    {
        RealType rInv6 = rInvV * rInvV;
        return (rInv6 * rInv6 * rInv6);
    }

    template<class RealType>
    RealType calculateVdw6(const RealType c6, const RealType rInv6)
    {
        return (c6 * rInv6);
    }

    template<class RealType>
    RealType calculateVdw12(const RealType c12, const RealType rInv6)
    {
        return (c12 * rInv6 * rInv6);
    }

    /* reaction-field electrostatics */
    template<class RealType>
    RealType reactionFieldScalarForce(const RealType qq, const RealType rInv, const RealType r, const real krf)
    {
        return (qq * (rInv - two_ * krf * r * r));
    }

    template<class RealType>
    RealType reactionFieldPotential(const RealType qq,
                                    const RealType rInv,
                                    const RealType r,
                                    const real     krf,
                                    const real     potentialShift)
    {
        return (qq * (rInv + krf * r * r - potentialShift));
    }

    /* Ewald electrostatics */
    template<class RealType>
    RealType ewaldScalarForce(const RealType coulomb, const RealType rInv)
    {
        return (coulomb * rInv);
    }

    template<class RealType>
    RealType ewaldPotential(const RealType coulomb, const RealType rInv, const real potentialShift)
    {
        return (coulomb * (rInv - potentialShift));
    }

    /* cutoff LJ */
    template<class RealType>
    RealType lennardJonesScalarForce(const RealType v6, const RealType v12)
    {
        return (v12 - v6);
    }

    template<class RealType>
    RealType lennardJonesPotential(const RealType v6,
                                   const RealType v12,
                                   const RealType c6,
                                   const RealType c12,
                                   const real     repulsionShift,
                                   const real     dispersionShift)
    {
        return ((v12 + c12 * repulsionShift) * oneTwelfth_ - (v6 + c6 * dispersionShift) * oneSixth_);
    }

    /* Ewald LJ */
    real ewaldLennardJonesGridSubtract(const real c6grid, const real potentialShift)
    {
        return (c6grid * potentialShift * oneSixth_);
    }

    /* LJ Potential switch */
    template<class RealType>
    RealType potSwitchScalarForceMod(const RealType fScalarInp,
                                     const RealType potential,
                                     const RealType sw,
                                     const RealType r,
                                     const RealType rVdw,
                                     const RealType dsw)
    {
        if (r < rVdw)
        {
            real fScalar = fScalarInp * sw - r * potential * dsw;
            return (fScalar);
        }
        return (zero_);
    }

    template<class RealType>
    RealType potSwitchPotentialMod(const RealType potentialInp, const RealType sw, const RealType r, const RealType rVdw)
    {
        if (r < rVdw)
        {
            real potential = potentialInp * sw;
            return (potential);
        }
        return (zero_);
    }

    /* computations in the free energy kernel that are independent of the kind of soft-core potential used */
    template<bool elecInteractionTypeIsEwald, class RealType>
    void calculateCoulombInteraction(RealType (&vCoul)[NSTATES],
                                     const RealType qqValue,
                                     const RealType rInvC,
                                     const real     sh_ewald,
                                     RealType (&fScalC)[NSTATES],
                                     const RealType rC,
                                     const real     krf,
                                     const real     crf,
                                     const int      i)
    {
        if (elecInteractionTypeIsEwald)
        {
            vCoul[i]  = ewaldPotential(qqValue, rInvC, sh_ewald);
            fScalC[i] = ewaldScalarForce(qqValue, rInvC);
        }
        else
        {
            vCoul[i]  = reactionFieldPotential(qqValue, rInvC, rC, krf, crf);
            fScalC[i] = reactionFieldScalarForce(qqValue, rInvC, rC, krf);
        }
    }

    template<bool vdwInteractionTypeIsEwald, bool vdwModifierIsPotSwitch, class RealType>
    void calculateVdwInteraction(RealType (&vVdw)[NSTATES],
                                 const real     nbfpGridValue,
                                 const real     shLjEwald,
                                 const RealType rV,
                                 const real     rvdw_switch,
                                 const real     vdw_swV3,
                                 const real     vdw_swV4,
                                 const real     vdw_swV5,
                                 const real     vdw_swF2,
                                 const real     vdw_swF3,
                                 const real     vdw_swF4,
                                 RealType (&fScalV)[NSTATES],
                                 const real rVdw,
                                 const int  i)
    {
        if (vdwInteractionTypeIsEwald)
        {
            /* Subtract the grid potential at the cut-off */
            vVdw[i] += ewaldLennardJonesGridSubtract(nbfpGridValue, shLjEwald);
        }

        if (vdwModifierIsPotSwitch)
        {
            RealType d         = rV - rvdw_switch;
            d                  = (d > zero_) ? d : zero_;
            const RealType d2  = d * d;
            const RealType sw  = one_ + d2 * d * (vdw_swV3 + d * (vdw_swV4 + d * vdw_swV5));
            const RealType dsw = d2 * (vdw_swF2 + d * (vdw_swF3 + d * vdw_swF4));

            fScalV[i] = potSwitchScalarForceMod(fScalV[i], vVdw[i], sw, rV, rVdw, dsw);
            vVdw[i]   = potSwitchPotentialMod(vVdw[i], sw, rV, rVdw);
        }
    }

    template<class RealType>
    void calculateCoulombRF(const real     krf,
                            const RealType rSq,
                            const real     crf,
                            real&          vCTot,
                            RealType&      fScal,
                            real&          dvdlCoul,
                            RealType (&qq)[NSTATES],
                            real (&LFC)[NSTATES],
                            real (&DLF)[NSTATES],
                            const int ii,
                            const int jnr)
    {
        /* For excluded pairs, which are only in this pair list when
         * using the Verlet scheme, we don't use soft-core.
         * As there is no singularity, there is no need for soft-core.
         */
        const real FF = -two_ * krf;
        RealType   VV = krf * rSq - crf;

        if (ii == jnr)
        {
            VV *= half_;
        }

        for (int i = 0; i < NSTATES; i++)
        {
            vCTot += LFC[i] * qq[i] * VV;
            fScal += LFC[i] * qq[i] * FF;
            dvdlCoul += DLF[i] * qq[i] * VV;
        }
    }

    template<typename DataTypes, bool elecInteractionTypeIsEwald, class RealType>
    void calculateCoulombEwald(const RealType r,
                               const real     rCoulomb,
                               const real     coulombTableScale,
                               const real*    ewtab,
                               const real     coulombTableScaleInvHalf,
                               const RealType rInv,
                               real&          vCTot,
                               RealType&      fScal,
                               real&          dvdlCoul,
                               RealType (&qq)[NSTATES],
                               real (&LFC)[NSTATES],
                               real (&DLF)[NSTATES],
                               const int  ii,
                               const int  jnr,
                               const bool bPairIncluded)
    {
        if (elecInteractionTypeIsEwald && (r < rCoulomb || !bPairIncluded))
        {
            /* See comment in the preamble. When using Ewald interactions
             * (unless we use a switch modifier) we subtract the reciprocal-space
             * Ewald component here which made it possible to apply the free
             * energy interaction to 1/r (vanilla coulomb short-range part)
             * above. This gets us closer to the ideal case of applying
             * the softcore to the entire electrostatic interaction,
             * including the reciprocal-space component.
             */
            using IntType = typename DataTypes::IntType;

            real v_lr, f_lr;

            const RealType ewrt   = r * coulombTableScale;
            IntType        ewitab = static_cast<IntType>(ewrt);
            const RealType eweps  = ewrt - ewitab;
            ewitab                = 4 * ewitab;
            f_lr                  = ewtab[ewitab] + eweps * ewtab[ewitab + 1];
            v_lr = (ewtab[ewitab + 2] - coulombTableScaleInvHalf * eweps * (ewtab[ewitab] + f_lr));
            f_lr *= rInv;

            /* Note that any possible Ewald shift has already been applied in
             * the normal interaction part above.
             */

            if (ii == jnr)
            {
                /* If we get here, the i particle (ii) has itself (jnr)
                 * in its neighborlist. This can only happen with the Verlet
                 * scheme, and corresponds to a self-interaction that will
                 * occur twice. Scale it down by 50% to only include it once.
                 */
                v_lr *= half_;
            }

            for (int i = 0; i < NSTATES; i++)
            {
                vCTot -= LFC[i] * qq[i] * v_lr;
                fScal -= LFC[i] * qq[i] * f_lr;
                dvdlCoul -= (DLF[i] * qq[i]) * v_lr;
            }
        }
    }

    template<typename DataTypes, bool vdwInteractionTypeIsEwald, class RealType>
    void calculateVdwEwald(const RealType            r,
                           const real                rVdw,
                           const RealType            rSq,
                           const RealType            rInv,
                           const real                vdwTableScale,
                           const real*               tab_ewald_F_lj,
                           const real*               tab_ewald_V_lj,
                           const real                vdwTableScaleInvHalf,
                           gmx::ArrayRef<const real> nbfp_grid,
                           int (&tj)[NSTATES],
                           real&     vVTot,
                           RealType& fScal,
                           real&     dvdlVdw,
                           real (&LFV)[NSTATES],
                           real (&DLF)[NSTATES],
                           const int  ii,
                           const int  jnr,
                           const bool bPairIncluded)
    {
        if (vdwInteractionTypeIsEwald && (r < rVdw || !bPairIncluded))
        {
            /* See comment in the preamble. When using LJ-Ewald interactions
             * (unless we use a switch modifier) we subtract the reciprocal-space
             * Ewald component here which made it possible to apply the free
             * energy interaction to r^-6 (vanilla LJ6 short-range part)
             * above. This gets us closer to the ideal case of applying
             * the softcore to the entire VdW interaction,
             * including the reciprocal-space component.
             */
            /* We could also use the analytical form here
             * iso a table, but that can cause issues for
             * r close to 0 for non-interacting pairs.
             */

            using IntType = typename DataTypes::IntType;

            const RealType rs   = rSq * rInv * vdwTableScale;
            const IntType  ri   = static_cast<IntType>(rs);
            const RealType frac = rs - ri;
            const RealType f_lr = (1 - frac) * tab_ewald_F_lj[ri] + frac * tab_ewald_F_lj[ri + 1];
            /* TODO: Currently the Ewald LJ table does not contain
             * the factor 1/6, we should add this.
             */
            const RealType FF = f_lr * rInv / six_;
            RealType       VV =
                    (tab_ewald_V_lj[ri] - vdwTableScaleInvHalf * frac * (tab_ewald_F_lj[ri] + f_lr)) / six_;

            if (ii == jnr)
            {
                /* If we get here, the i particle (ii) has itself (jnr)
                 * in its neighborlist. This can only happen with the Verlet
                 * scheme, and corresponds to a self-interaction that will
                 * occur twice. Scale it down by 50% to only include it once.
                 */
                VV *= half_;
            }

            for (int i = 0; i < NSTATES; i++)
            {
                const real c6grid = nbfp_grid[tj[i]];
                vVTot += LFV[i] * c6grid * VV;
                fScal += LFV[i] * c6grid * FF;
                dvdlVdw += (DLF[i] * c6grid) * VV;
            }
        }
    }

    template<class RealType>
    void calculateForcesInnerLoop(real* gmx_restrict f,
                                  const real         dX,
                                  const real         dY,
                                  const real         dZ,
                                  const RealType     fScal,
                                  real&              fIX,
                                  real&              fIY,
                                  real&              fIZ,
                                  const int          j3,
                                  const bool         doForces)
    {
        if (doForces)
        {
            const real tX = fScal * dX;
            const real tY = fScal * dY;
            const real tZ = fScal * dZ;
            fIX           = fIX + tX;
            fIY           = fIY + tY;
            fIZ           = fIZ + tZ;
            /* OpenMP atomics are expensive, but this kernels is also
             * expensive, so we can take this hit, instead of using
             * thread-local output buffers and extra reduction.
             *
             * All the OpenMP regions in this file are trivial and should
             * not throw, so no need for try/catch.
             */
#pragma omp atomic
            f[j3] -= tX;
#pragma omp atomic
            f[j3 + 1] -= tY;
#pragma omp atomic
            f[j3 + 2] -= tZ;
        }
    }

    void calculateForcesAndPotentialOuterLoop(real* gmx_restrict f,
                                              real* gmx_restrict       fshift,
                                              const real               fIX,
                                              const real               fIY,
                                              const real               fIZ,
                                              gmx::ArrayRef<const int> gid,
                                              gmx::ArrayRef<real>      energygrp_elec,
                                              gmx::ArrayRef<real>      energygrp_vdw,
                                              const real               vCTot,
                                              const real               vVTot,
                                              const int                ii3,
                                              const int                is3,
                                              const int                npair_within_cutoff,
                                              const int                n,
                                              const bool               doForces,
                                              const bool               doShiftForces,
                                              const bool               doPotential)
    {
        /* The atomics below are expensive with many OpenMP threads.
         * Here unperturbed i-particles will usually only have a few
         * (perturbed) j-particles in the list. Thus with a buffered list
         * we can skip a significant number of i-reductions with a check.
         */
        if (npair_within_cutoff > 0)
        {
            if (doForces)
            {
#pragma omp atomic
                f[ii3] += fIX;
#pragma omp atomic
                f[ii3 + 1] += fIY;
#pragma omp atomic
                f[ii3 + 2] += fIZ;
            }
            if (doShiftForces)
            {
#pragma omp atomic
                fshift[is3] += fIX;
#pragma omp atomic
                fshift[is3 + 1] += fIY;
#pragma omp atomic
                fshift[is3 + 2] += fIZ;
            }
            if (doPotential)
            {
                int ggid = gid[n];
#pragma omp atomic
                energygrp_elec[ggid] += vCTot;
#pragma omp atomic
                energygrp_vdw[ggid] += vVTot;
            }
        }
    }

    /* FIXME: How should these be handled with SIMD? */
    static constexpr real oneTwelfth_ = 1.0 / 12.0;
    static constexpr real oneSixth_   = 1.0 / 6.0;
    static constexpr real zero_       = 0.0;
    static constexpr real half_       = 0.5;
    static constexpr real one_        = 1.0;
    static constexpr real two_        = 2.0;
    static constexpr real six_        = 6.0;
};

template<bool vdwInteractionTypeIsEwald, bool elecInteractionTypeIsEwald, bool vdwModifierIsPotSwitch>
struct KernelVariables
{
    //! constructor
    KernelVariables(const t_nblist&                nlist,
                    gmx::ArrayRef<const gmx::RVec> coords,
                    gmx::ForceWithShiftForces*     forceWithShiftForces,
                    const real                     rlist,
                    const interaction_const_t&     ic,
                    int                            flags,
                    gmx::ArrayRef<const real>      lambda) :
        nri(nlist.nri),       /* Extract pair list data */
        iinr(nlist.iinr),     /* Extract pair list data */
        jindex(nlist.jindex), /* Extract pair list data */
        jjnr(nlist.jjnr),     /* Extract pair list data */
        shift(nlist.shift),   /* Extract pair list data */
        gid(nlist.gid),       /* Extract pair list data */
        lambda_coul(lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)]),
        lambda_vdw(lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)]),
        doForces((flags & GMX_NONBONDED_DO_FORCE) != 0),
        doShiftForces((flags & GMX_NONBONDED_DO_SHIFTFORCE) != 0),
        doPotential((flags & GMX_NONBONDED_DO_POTENTIAL) != 0),
        facel(ic.epsfac),                          /* Extract data from interaction_const_t */
        rCoulomb(ic.rcoulomb),                     /* Extract data from interaction_const_t */
        krf(ic.reactionFieldCoefficient),          /* Extract data from interaction_const_t */
        crf(ic.reactionFieldShift),                /* Extract data from interaction_const_t */
        shLjEwald(ic.sh_lj_ewald),                 /* Extract data from interaction_const_t */
        rVdw(ic.rvdw),                             /* Extract data from interaction_const_t */
        dispersionShift(ic.dispersion_shift.cpot), /* Extract data from interaction_const_t */
        repulsionShift(ic.repulsion_shift.cpot),   /* Extract data from interaction_const_t */
        dvdlCoul(0),
        dvdlVdw(0),
        LFC{ static_cast<real>(1.0) - lambda_coul, lambda_coul }, /* Lambda factor for Coulomb*/
        LFV{ static_cast<real>(1.0) - lambda_vdw, lambda_vdw }, /* Lambda factor for van der Waals*/
        DLF{ -1, 1 }, /*derivative of the lambda factor for state A and B */
        x(coords[0]),
        f(&(forceWithShiftForces->force()[0][0])),
        fshift(&(forceWithShiftForces->shiftForces()[0][0])),
        rlistSquared(gmx::square(rlist)),
        numExcludedPairsBeyondRlist(0)
    {
        // Note that the nbnxm kernels do not support Coulomb potential switching at all
        GMX_ASSERT(ic.coulomb_modifier != InteractionModifiers::PotSwitch,
                   "Potential switching is not supported for Coulomb with FEP");

        /* Neither the assert nor the calculation of the vdW interaction depend on
         * whether soft-core potentials are used or not.
         * Therefore, I left the original comment but changed the assert such that
         * it now warns users that FEP and the particular vdW options are incompatible.
         */
        /* For Ewald/PME interactions we cannot easily apply the soft-core component to
         * reciprocal space. When we use non-switched Ewald interactions, we
         * assume the soft-coring does not significantly affect the grid contribution
         * and apply the soft-core only to the full 1/r (- shift) pair contribution.
         *
         * However, we cannot use this approach for switch-modified since we would then
         * effectively end up evaluating a significantly different interaction here compared to the
         * normal (non-free-energy) kernels, either by applying a cutoff at a different
         * position than what the user requested, or by switching different
         * things (1/r rather than short-range Ewald). For these settings, we just
         * use the traditional short-range Ewald interaction in that case.
         */
        GMX_RELEASE_ASSERT(!(vdwInteractionTypeIsEwald && vdwModifierIsPotSwitch),
                           "Combining switched Ewald potentials for van der Waals interactions and "
                           "free energy perturbation is not supported.");

        if (vdwModifierIsPotSwitch)
        {
            const real d = ic.rvdw - ic.rvdw_switch;
            vdw_swV3     = -10.0 / (d * d * d);
            vdw_swV4     = 15.0 / (d * d * d * d);
            vdw_swV5     = -6.0 / (d * d * d * d * d);
            vdw_swF2     = -30.0 / (d * d * d);
            vdw_swF3     = 60.0 / (d * d * d * d);
            vdw_swF4     = -30.0 / (d * d * d * d * d);
        }
        else
        {
            /* Avoid warnings from stupid compilers (looking at you, Clang!) */
            vdw_swV3 = vdw_swV4 = vdw_swV5 = vdw_swF2 = vdw_swF3 = vdw_swF4 = 0.0;
        }

        if (ic.eeltype == CoulombInteractionType::Cut || EEL_RF(ic.eeltype))
        {
            icoul = NbkernelElecType::ReactionField;
        }
        else
        {
            icoul = NbkernelElecType::None;
        }

        rcutoff_max2 = std::max(ic.rcoulomb, ic.rvdw);
        rcutoff_max2 = rcutoff_max2 * rcutoff_max2;

        tab_ewald_F_lj           = nullptr;
        tab_ewald_V_lj           = nullptr;
        ewtab                    = nullptr;
        coulombTableScale        = 0;
        coulombTableScaleInvHalf = 0;
        vdwTableScale            = 0;
        vdwTableScaleInvHalf     = 0;
        sh_ewald                 = 0;
        if (elecInteractionTypeIsEwald || vdwInteractionTypeIsEwald)
        {
            sh_ewald = ic.sh_ewald;
        }
        if (elecInteractionTypeIsEwald)
        {
            const auto& coulombTables = *ic.coulombEwaldTables;
            ewtab                     = coulombTables.tableFDV0.data();
            coulombTableScale         = coulombTables.scale;
            coulombTableScaleInvHalf  = 0.5 / coulombTableScale;
        }
        if (vdwInteractionTypeIsEwald)
        {
            const auto& vdwTables = *ic.vdwEwaldTables;
            tab_ewald_F_lj        = vdwTables.tableF.data();
            tab_ewald_V_lj        = vdwTables.tableV.data();
            vdwTableScale         = vdwTables.scale;
            vdwTableScaleInvHalf  = 0.5 / vdwTableScale;
        }
    }

    //! list of variables
    const int                nri;
    gmx::ArrayRef<const int> iinr;
    gmx::ArrayRef<const int> jindex;
    gmx::ArrayRef<const int> jjnr;
    gmx::ArrayRef<const int> shift;
    gmx::ArrayRef<const int> gid;

    const real lambda_coul;
    const real lambda_vdw;
    const bool doForces;
    const bool doShiftForces;
    const bool doPotential;

    const real facel;
    const real rCoulomb;
    const real krf;
    const real crf;
    const real shLjEwald;
    const real rVdw;
    const real dispersionShift;
    const real repulsionShift;

    real dvdlCoul;
    real dvdlVdw;

    real LFC[NSTATES];
    real LFV[NSTATES];
    real DLF[NSTATES];

    // TODO: We should get rid of using pointers to real
    const real* x;
    real* gmx_restrict f;
    real* gmx_restrict fshift;

    const real rlistSquared;

    int numExcludedPairsBeyondRlist;

    real vdw_swV3;
    real vdw_swV4;
    real vdw_swV5;
    real vdw_swF2;
    real vdw_swF3;
    real vdw_swF4;

    NbkernelElecType icoul;

    real rcutoff_max2;

    const real* tab_ewald_F_lj;
    const real* tab_ewald_V_lj;
    const real* ewtab;

    real coulombTableScale;
    real coulombTableScaleInvHalf;
    real vdwTableScale;
    real vdwTableScaleInvHalf;
    real sh_ewald;
};

class SoftcoreCalculatorNone
{
public:
    //! constructor
    SoftcoreCalculatorNone() :
        softcoreCalculatorShared_(std::make_unique<SoftcoreCalculatorShared>())
    {
    }

    //! helper functions for NBFreeEnergyKernel that are specific for this kind of soft-core potential
    template<class RealType>
    void prepareVdwAndSoftcoreParameters(RealType (&c12)[NSTATES],
                                         gmx::ArrayRef<const real> nbfp,
                                         int (&tj)[NSTATES])
    {
        for (int i = 0; i < NSTATES; i++)
        {
            c12[i] = nbfp[tj[i] + 1];
        }
    }

    template<class RealType>
    void calculateRadii(RealType&      rPInvC,
                        RealType&      rInvC,
                        RealType&      rC,
                        RealType&      rPInvV,
                        RealType&      rInvV,
                        RealType&      rV,
                        const RealType rInv,
                        const RealType r)
    {
        rPInvC = 1;
        rInvC  = rInv;
        rC     = r;

        rPInvV = 1;
        rInvV  = rInv;
        rV     = r;
    }

    template<class RealType>
    void prepareForVdwInteraction(const RealType rInvV,
                                  const RealType c6Value,
                                  const RealType c12Value,
                                  RealType (&vVdw)[NSTATES],
                                  const real repulsionShift,
                                  const real dispersionShift,
                                  RealType (&fScalV)[NSTATES],
                                  const int i)
    {
        RealType rInv6  = softcoreCalculatorShared_->calculateRinv6(rInvV);
        RealType vVdw6  = softcoreCalculatorShared_->calculateVdw6(c6Value, rInv6);
        RealType vVdw12 = softcoreCalculatorShared_->calculateVdw12(c12Value, rInv6);

        vVdw[i] = softcoreCalculatorShared_->lennardJonesPotential(
                vVdw6, vVdw12, c6Value, c12Value, repulsionShift, dispersionShift);
        fScalV[i] = softcoreCalculatorShared_->lennardJonesScalarForce(vVdw6, vVdw12);
    }

    template<class RealType>
    void assembleAAndBStates(real& vCTot,
                             real (&LFC)[NSTATES],
                             RealType (&vCoul)[NSTATES],
                             real& vVTot,
                             real (&LFV)[NSTATES],
                             RealType (&vVdw)[NSTATES],
                             RealType& fScal,
                             RealType (&fScalC)[NSTATES],
                             RealType (&fScalV)[NSTATES],
                             const RealType rpm2,
                             real&          dvdlCoul,
                             real&          dvdlVdw,
                             real (&DLF)[NSTATES])
    {
        /* Assemble A and B states */
        for (int i = 0; i < NSTATES; i++)
        {
            vCTot += LFC[i] * vCoul[i];
            vVTot += LFV[i] * vVdw[i];

            fScal += LFC[i] * fScalC[i] * rpm2;
            fScal += LFV[i] * fScalV[i] * rpm2;

            dvdlCoul += vCoul[i] * DLF[i];
            dvdlVdw += vVdw[i] * DLF[i];
        }
    }

    //! implementation of the nb_free_energy_kernel if no soft-core potentials are used
    template<typename DataTypes, bool scLambdasOrAlphasDiffer, bool vdwInteractionTypeIsEwald, bool elecInteractionTypeIsEwald, bool vdwModifierIsPotSwitch>
    void NBFreeEnergyKernel(const t_nblist&                nlist,
                            gmx::ArrayRef<const gmx::RVec> coords,
                            gmx::ForceWithShiftForces*     forceWithShiftForces,
                            const int                      ntype,
                            const real                     rlist,
                            const interaction_const_t&     ic,
                            gmx::ArrayRef<const gmx::RVec> shiftvec,
                            gmx::ArrayRef<const real>      nbfp,
                            gmx::ArrayRef<const real>      nbfp_grid,
                            gmx::ArrayRef<const real>      chargeA,
                            gmx::ArrayRef<const real>      chargeB,
                            gmx::ArrayRef<const int>       typeA,
                            gmx::ArrayRef<const int>       typeB,
                            int                            flags,
                            gmx::ArrayRef<const real>      lambda,
                            gmx::ArrayRef<real>            dvdl,
                            gmx::ArrayRef<real>            energygrp_elec,
                            gmx::ArrayRef<real>            energygrp_vdw,
                            t_nrnb* gmx_restrict nrnb)
    {
        using RealType = typename DataTypes::RealType;

        KernelVariables<vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, vdwModifierIsPotSwitch> kv{
            nlist, coords, forceWithShiftForces, rlist, ic, flags, lambda
        };

        for (int n = 0; n < kv.nri; n++)
        {
            int npair_within_cutoff = 0;

            const int  is    = kv.shift[n];
            const int  is3   = DIM * is;
            const real shX   = shiftvec[is][XX];
            const real shY   = shiftvec[is][YY];
            const real shZ   = shiftvec[is][ZZ];
            const int  nj0   = kv.jindex[n];
            const int  nj1   = kv.jindex[n + 1];
            const int  ii    = kv.iinr[n];
            const int  ii3   = 3 * ii;
            const real ix    = shX + kv.x[ii3 + 0];
            const real iy    = shY + kv.x[ii3 + 1];
            const real iz    = shZ + kv.x[ii3 + 2];
            const real iqA   = kv.facel * chargeA[ii];
            const real iqB   = kv.facel * chargeB[ii];
            const int  ntiA  = 2 * ntype * typeA[ii];
            const int  ntiB  = 2 * ntype * typeB[ii];
            real       vCTot = 0;
            real       vVTot = 0;
            real       fIX   = 0;
            real       fIY   = 0;
            real       fIZ   = 0;

            for (int k = nj0; k < nj1; k++)
            {
                int       tj[NSTATES];
                const int jnr = kv.jjnr[k];
                const int j3  = 3 * jnr;
                RealType  c6[NSTATES], c12[NSTATES], qq[NSTATES], vCoul[NSTATES], vVdw[NSTATES];
                RealType  r, rInv, rpm2;
                const RealType dX  = ix - kv.x[j3];
                const RealType dY  = iy - kv.x[j3 + 1];
                const RealType dZ  = iz - kv.x[j3 + 2];
                const RealType rSq = dX * dX + dY * dY + dZ * dZ;
                RealType       fScalC[NSTATES], fScalV[NSTATES];
                /* Check if this pair on the exlusions list.*/
                const bool bPairIncluded = nlist.excl_fep.empty() || nlist.excl_fep[k];

                if (rSq >= kv.rcutoff_max2 && bPairIncluded)
                {
                    /* We save significant time by skipping all code below.
                     * Exclusions outside the cutoff can not be skipped as
                     * when using Ewald: the reciprocal-space
                     * Ewald component still needs to be subtracted.
                     */

                    continue;
                }
                npair_within_cutoff++;

                if (rSq > kv.rlistSquared)
                {
                    kv.numExcludedPairsBeyondRlist++;
                }

                if (rSq > 0)
                {
                    /* Note that unlike in the nbnxn kernels, we do not need
                     * to clamp the value of rSq before taking the invsqrt
                     * to avoid NaN in the LJ calculation, since here we do
                     * not calculate LJ interactions when C6 and C12 are zero.
                     */

                    rInv = gmx::invsqrt(rSq);
                    r    = rSq * rInv;
                }
                else
                {
                    /* A pure precaution if no soft-core potential is used.
                     * Ideally, this else-loop will only be entered
                     * if both Coulomb and vdW have been switched off completely.
                     */
                    rInv = 0;
                    r    = 0;
                }

                rpm2 = rInv * rInv;

                RealType fScal = 0;

                qq[STATE_A] = iqA * chargeA[jnr];
                qq[STATE_B] = iqB * chargeB[jnr];

                tj[STATE_A] = ntiA + 2 * typeA[jnr];
                tj[STATE_B] = ntiB + 2 * typeB[jnr];

                if (bPairIncluded)
                {
                    c6[STATE_A] = nbfp[tj[STATE_A]];
                    c6[STATE_B] = nbfp[tj[STATE_B]];

                    prepareVdwAndSoftcoreParameters(c12, nbfp, tj);

                    for (int i = 0; i < NSTATES; i++)
                    {
                        fScalC[i] = 0;
                        fScalV[i] = 0;
                        vCoul[i]  = 0;
                        vVdw[i]   = 0;

                        RealType rInvC, rInvV, rC, rV, rPInvC, rPInvV;

                        /* Only spend time on A or B state if it is non-zero */
                        if ((qq[i] != 0) || (c6[i] != 0) || (c12[i] != 0))
                        {
                            calculateRadii(rPInvC, rInvC, rC, rPInvV, rInvV, rV, rInv, r);

                            /* Only process the coulomb interactions if we have charges,
                             * and if we either include all entries in the list (no cutoff
                             * used in the kernel), or if we are within the cutoff.
                             */
                            bool computeElecInteraction =
                                    (elecInteractionTypeIsEwald && r < kv.rCoulomb)
                                    || (!elecInteractionTypeIsEwald && rC < kv.rCoulomb);

                            if ((qq[i] != 0) && computeElecInteraction)
                            {
                                softcoreCalculatorShared_->calculateCoulombInteraction<elecInteractionTypeIsEwald>(
                                        vCoul, qq[i], rInvC, kv.sh_ewald, fScalC, rC, kv.krf, kv.crf, i);
                            }

                            /* Only process the VDW interactions if we have
                             * some non-zero parameters, and if we either
                             * include all entries in the list (no cutoff used
                             * in the kernel), or if we are within the cutoff.
                             */
                            bool computeVdwInteraction = (vdwInteractionTypeIsEwald && r < kv.rVdw)
                                                         || (!vdwInteractionTypeIsEwald && rV < kv.rVdw);
                            if ((c6[i] != 0 || c12[i] != 0) && computeVdwInteraction)
                            {
                                prepareForVdwInteraction(
                                        rInvV, c6[i], c12[i], vVdw, kv.repulsionShift, kv.dispersionShift, fScalV, i);
                                softcoreCalculatorShared_->calculateVdwInteraction<vdwInteractionTypeIsEwald, vdwModifierIsPotSwitch>(
                                        vVdw,
                                        nbfp_grid[tj[i]],
                                        kv.shLjEwald,
                                        rV,
                                        ic.rvdw_switch,
                                        kv.vdw_swV3,
                                        kv.vdw_swV4,
                                        kv.vdw_swV5,
                                        kv.vdw_swF2,
                                        kv.vdw_swF3,
                                        kv.vdw_swF4,
                                        fScalV,
                                        kv.rVdw,
                                        i);
                            }

                            /* fScalC (and fScalV) now contain: dV/drC * rC
                             * Now we multiply by rC^-p, so it will be: dV/drC * rC^1-p
                             * Further down we first multiply by r^p-2 and then by
                             * the vector r, which in total gives: dV/drC * (r/rC)^1-p
                             */
                            fScalC[i] *= rPInvC;
                            fScalV[i] *= rPInvV;
                        }
                    } // end for (int i = 0; i < NSTATES; i++)

                    assembleAAndBStates(vCTot,
                                        kv.LFC,
                                        vCoul,
                                        vVTot,
                                        kv.LFV,
                                        vVdw,
                                        fScal,
                                        fScalC,
                                        fScalV,
                                        rpm2,
                                        kv.dvdlCoul,
                                        kv.dvdlVdw,
                                        kv.DLF);

                } // end if (bPairIncluded)
                else if (kv.icoul == NbkernelElecType::ReactionField)
                {
                    softcoreCalculatorShared_->calculateCoulombRF(
                            kv.krf, rSq, kv.crf, vCTot, fScal, kv.dvdlCoul, qq, kv.LFC, kv.DLF, ii, jnr);
                }

                softcoreCalculatorShared_->calculateCoulombEwald<DataTypes, elecInteractionTypeIsEwald>(
                        r,
                        kv.rCoulomb,
                        kv.coulombTableScale,
                        kv.ewtab,
                        kv.coulombTableScaleInvHalf,
                        rInv,
                        vCTot,
                        fScal,
                        kv.dvdlCoul,
                        qq,
                        kv.LFC,
                        kv.DLF,
                        ii,
                        jnr,
                        bPairIncluded);
                softcoreCalculatorShared_->calculateVdwEwald<DataTypes, vdwInteractionTypeIsEwald>(
                        r,
                        kv.rVdw,
                        rSq,
                        rInv,
                        kv.vdwTableScale,
                        kv.tab_ewald_F_lj,
                        kv.tab_ewald_V_lj,
                        kv.vdwTableScaleInvHalf,
                        nbfp_grid,
                        tj,
                        vVTot,
                        fScal,
                        kv.dvdlVdw,
                        kv.LFV,
                        kv.DLF,
                        ii,
                        jnr,
                        bPairIncluded);
                softcoreCalculatorShared_->calculateForcesInnerLoop(
                        kv.f, dX, dY, dZ, fScal, fIX, fIY, fIZ, j3, kv.doForces);

            } // end for (int k = nj0; k < nj1; k++)

            softcoreCalculatorShared_->calculateForcesAndPotentialOuterLoop(kv.f,
                                                                            kv.fshift,
                                                                            fIX,
                                                                            fIY,
                                                                            fIZ,
                                                                            kv.gid,
                                                                            energygrp_elec,
                                                                            energygrp_vdw,
                                                                            vCTot,
                                                                            vVTot,
                                                                            ii3,
                                                                            is3,
                                                                            npair_within_cutoff,
                                                                            n,
                                                                            kv.doForces,
                                                                            kv.doShiftForces,
                                                                            kv.doPotential);
        } // end for (int n = 0; n < nri; n++)

#pragma omp atomic
        dvdl[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)] += kv.dvdlCoul;
#pragma omp atomic
        dvdl[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)] += kv.dvdlVdw;

        /* Estimate flops, average for free energy stuff:
         * 12  flops per outer iteration
         * 150 flops per inner iteration
         */
        atomicNrnbIncrement(nrnb, eNR_NBKERNEL_FREE_ENERGY, nlist.nri * 12 + nlist.jindex[kv.nri] * 150);

        if (kv.numExcludedPairsBeyondRlist > 0)
        {
            gmx_fatal(FARGS,
                      "There are %d perturbed non-bonded pair interactions beyond the pair-list "
                      "cutoff "
                      "of %g nm, which is not supported. This can happen because the system is "
                      "unstable or because intra-molecular interactions at long distances are "
                      "excluded. If the "
                      "latter is the case, you can try to increase nstlist or rlist to avoid this."
                      "The error is likely triggered by the use of couple-intramol=no "
                      "and the maximal distance in the decoupled molecule exceeding rlist.",
                      kv.numExcludedPairsBeyondRlist,
                      rlist);
        }
    }

private:
    std::unique_ptr<SoftcoreCalculatorShared> softcoreCalculatorShared_;
};

class SoftcoreCalculatorBeutler
{
public:
    //! constructor
    SoftcoreCalculatorBeutler() :
        softcoreCalculatorShared_(std::make_unique<SoftcoreCalculatorShared>())
    {
    }

    //! helper functions for NBFreeEnergyKernel that are specific for this kind of soft-core potential
    template<class RealType>
    void prepareVdwAndSoftcoreParameters(RealType (&c12)[NSTATES],
                                         gmx::ArrayRef<const real> nbfp,
                                         int (&tj)[NSTATES],
                                         RealType (&c6)[NSTATES],
                                         RealType (&sigma6)[NSTATES],
                                         const real sigma6_min,
                                         const real sigma6_def,
                                         RealType&  alphaVdwEff,
                                         RealType&  alphaCoulEff,
                                         const real alpha_vdw,
                                         const real alpha_coul)
    {
        for (int i = 0; i < NSTATES; i++)
        {
            c12[i] = nbfp[tj[i] + 1];
            if ((c6[i] > 0) && (c12[i] > 0))
            {
                /* c12 is stored scaled with 12.0 and c6 is scaled with 6.0 - correct for this */
                sigma6[i] = 0.5 * c12[i] / c6[i];
                if (sigma6[i] < sigma6_min) /* for disappearing coul and vdw with soft core at the same time */
                {
                    sigma6[i] = sigma6_min;
                }
            }
            else
            {
                sigma6[i] = sigma6_def;
            }
        }

        /* only use softcore if one of the states has a zero endstate - softcore is for avoiding infinities!*/
        if ((c12[STATE_A] > 0) && (c12[STATE_B] > 0))
        {
            alphaVdwEff  = 0;
            alphaCoulEff = 0;
        }
        else
        {
            alphaVdwEff  = alpha_vdw;
            alphaCoulEff = alpha_coul;
        }
    }

    template<bool scLambdasOrAlphasDiffer, class RealType>
    void calculateRadii(RealType&      rPInvC,
                        const RealType alphaCoulEff,
                        real (&lFacCoul)[NSTATES],
                        RealType (&sigma6)[NSTATES],
                        const RealType rp,
                        RealType&      rInvC,
                        RealType&      rC,
                        RealType&      rPInvV,
                        const RealType alphaVdwEff,
                        real (&lFacVdw)[NSTATES],
                        RealType& rInvV,
                        RealType& rV,
                        const int i)
    {
        /* this section has to be inside the loop because of the dependence on sigma6 */
        rPInvC = 1.0 / (alphaCoulEff * lFacCoul[i] * sigma6[i] + rp);
        softcoreCalculatorShared_->pthRoot(rPInvC, rInvC, rC);
        if (scLambdasOrAlphasDiffer)
        {
            rPInvV = 1.0 / (alphaVdwEff * lFacVdw[i] * sigma6[i] + rp);
            softcoreCalculatorShared_->pthRoot(rPInvV, rInvV, rV);
        }
        else
        {
            /* We can avoid one expensive pow and one / operation */
            rPInvV = rPInvC;
            rInvV  = rInvC;
            rV     = rC;
        }
    }

    template<class RealType>
    void prepareForVdwInteraction(const RealType rPInvV,
                                  const RealType c6Value,
                                  const RealType c12Value,
                                  RealType (&vVdw)[NSTATES],
                                  const real repulsionShift,
                                  const real dispersionShift,
                                  RealType (&fScalV)[NSTATES],
                                  const int i)
    {
        RealType rInv6  = rPInvV;
        RealType vVdw6  = softcoreCalculatorShared_->calculateVdw6(c6Value, rInv6);
        RealType vVdw12 = softcoreCalculatorShared_->calculateVdw12(c12Value, rInv6);

        vVdw[i] = softcoreCalculatorShared_->lennardJonesPotential(
                vVdw6, vVdw12, c6Value, c12Value, repulsionShift, dispersionShift);
        fScalV[i] = softcoreCalculatorShared_->lennardJonesScalarForce(vVdw6, vVdw12);
    }

    template<class RealType>
    void assembleAAndBStates(real& vCTot,
                             real (&LFC)[NSTATES],
                             RealType (&vCoul)[NSTATES],
                             real& vVTot,
                             real (&LFV)[NSTATES],
                             RealType (&vVdw)[NSTATES],
                             RealType& fScal,
                             RealType (&fScalC)[NSTATES],
                             RealType (&fScalV)[NSTATES],
                             const RealType rpm2,
                             real&          dvdlCoul,
                             real&          dvdlVdw,
                             real (&DLF)[NSTATES],
                             const RealType alphaCoulEff,
                             const RealType alphaVdwEff,
                             real (&dlFacCoul)[NSTATES],
                             real (&dlFacVdw)[NSTATES],
                             RealType (&sigma6)[NSTATES])
    {
        /* Assemble A and B states */
        for (int i = 0; i < NSTATES; i++)
        {
            vCTot += LFC[i] * vCoul[i];
            vVTot += LFV[i] * vVdw[i];

            fScal += LFC[i] * fScalC[i] * rpm2;
            fScal += LFV[i] * fScalV[i] * rpm2;

            dvdlCoul += vCoul[i] * DLF[i] + LFC[i] * alphaCoulEff * dlFacCoul[i] * fScalC[i] * sigma6[i];
            dvdlVdw += vVdw[i] * DLF[i] + LFV[i] * alphaVdwEff * dlFacVdw[i] * fScalV[i] * sigma6[i];
        }
    }

    //! implementation of the nb_free_energy_kernel if soft-core potentials of type Beutler are used
    template<typename DataTypes, bool scLambdasOrAlphasDiffer, bool vdwInteractionTypeIsEwald, bool elecInteractionTypeIsEwald, bool vdwModifierIsPotSwitch>
    void NBFreeEnergyKernel(const t_nblist&                nlist,
                            gmx::ArrayRef<const gmx::RVec> coords,
                            gmx::ForceWithShiftForces*     forceWithShiftForces,
                            const int                      ntype,
                            const real                     rlist,
                            const interaction_const_t&     ic,
                            gmx::ArrayRef<const gmx::RVec> shiftvec,
                            gmx::ArrayRef<const real>      nbfp,
                            gmx::ArrayRef<const real>      nbfp_grid,
                            gmx::ArrayRef<const real>      chargeA,
                            gmx::ArrayRef<const real>      chargeB,
                            gmx::ArrayRef<const int>       typeA,
                            gmx::ArrayRef<const int>       typeB,
                            int                            flags,
                            gmx::ArrayRef<const real>      lambda,
                            gmx::ArrayRef<real>            dvdl,
                            gmx::ArrayRef<real>            energygrp_elec,
                            gmx::ArrayRef<real>            energygrp_vdw,
                            t_nrnb* gmx_restrict nrnb)
    {
        using RealType = typename DataTypes::RealType;

        KernelVariables<vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, vdwModifierIsPotSwitch> kv{
            nlist, coords, forceWithShiftForces, rlist, ic, flags, lambda
        };

        const auto& scParams   = *ic.softCoreParameters;
        const real  alpha_coul = scParams.alphaCoulomb;
        const real  alpha_vdw  = scParams.alphaVdw;
        const real  lam_power  = scParams.lambdaPower;
        const real  sigma6_def = scParams.sigma6WithInvalidSigma;
        const real  sigma6_min = scParams.sigma6Minimum;

        real           lFacCoul[NSTATES], dlFacCoul[NSTATES], lFacVdw[NSTATES], dlFacVdw[NSTATES];
        constexpr real sc_r_power = 6.0_real;
        for (int i = 0; i < NSTATES; i++)
        {
            lFacCoul[i] = (lam_power == 2 ? (1 - kv.LFC[i]) * (1 - kv.LFC[i]) : (1 - kv.LFC[i]));
            dlFacCoul[i] = kv.DLF[i] * lam_power / sc_r_power * (lam_power == 2 ? (1 - kv.LFC[i]) : 1);
            lFacVdw[i] = (lam_power == 2 ? (1 - kv.LFV[i]) * (1 - kv.LFV[i]) : (1 - kv.LFV[i]));
            dlFacVdw[i] = kv.DLF[i] * lam_power / sc_r_power * (lam_power == 2 ? (1 - kv.LFV[i]) : 1);
        }

        for (int n = 0; n < kv.nri; n++)
        {
            int npair_within_cutoff = 0;

            const int  is    = kv.shift[n];
            const int  is3   = DIM * is;
            const real shX   = shiftvec[is][XX];
            const real shY   = shiftvec[is][YY];
            const real shZ   = shiftvec[is][ZZ];
            const int  nj0   = kv.jindex[n];
            const int  nj1   = kv.jindex[n + 1];
            const int  ii    = kv.iinr[n];
            const int  ii3   = 3 * ii;
            const real ix    = shX + kv.x[ii3 + 0];
            const real iy    = shY + kv.x[ii3 + 1];
            const real iz    = shZ + kv.x[ii3 + 2];
            const real iqA   = kv.facel * chargeA[ii];
            const real iqB   = kv.facel * chargeB[ii];
            const int  ntiA  = 2 * ntype * typeA[ii];
            const int  ntiB  = 2 * ntype * typeB[ii];
            real       vCTot = 0;
            real       vVTot = 0;
            real       fIX   = 0;
            real       fIY   = 0;
            real       fIZ   = 0;

            for (int k = nj0; k < nj1; k++)
            {
                int       tj[NSTATES];
                const int jnr = kv.jjnr[k];
                const int j3  = 3 * jnr;
                RealType  c6[NSTATES], c12[NSTATES], qq[NSTATES], vCoul[NSTATES], vVdw[NSTATES];
                RealType  r, rInv, rp, rpm2;
                RealType  alphaVdwEff, alphaCoulEff, sigma6[NSTATES];
                const RealType dX  = ix - kv.x[j3];
                const RealType dY  = iy - kv.x[j3 + 1];
                const RealType dZ  = iz - kv.x[j3 + 2];
                const RealType rSq = dX * dX + dY * dY + dZ * dZ;
                RealType       fScalC[NSTATES], fScalV[NSTATES];
                /* Check if this pair on the exlusions list.*/
                const bool bPairIncluded = nlist.excl_fep.empty() || nlist.excl_fep[k];

                if (rSq >= kv.rcutoff_max2 && bPairIncluded)
                {
                    /* We save significant time by skipping all code below.
                     * Note that with soft-core interactions, the actual cut-off
                     * check might be different. But since the soft-core distance
                     * is always larger than r, checking on r here is safe.
                     * Exclusions outside the cutoff can not be skipped as
                     * when using Ewald: the reciprocal-space
                     * Ewald component still needs to be subtracted.
                     */

                    continue;
                }
                npair_within_cutoff++;

                if (rSq > kv.rlistSquared)
                {
                    kv.numExcludedPairsBeyondRlist++;
                }

                if (rSq > 0)
                {
                    /* Note that unlike in the nbnxn kernels, we do not need
                     * to clamp the value of rSq before taking the invsqrt
                     * to avoid NaN in the LJ calculation, since here we do
                     * not calculate LJ interactions when C6 and C12 are zero.
                     */

                    rInv = gmx::invsqrt(rSq);
                    r    = rSq * rInv;
                }
                else
                {
                    /* The force at r=0 is zero, because of symmetry.
                     * But note that the potential is in general non-zero,
                     * since the soft-cored r will be non-zero.
                     */
                    rInv = 0;
                    r    = 0;
                }

                rpm2 = rSq * rSq;  /* r4 */
                rp   = rpm2 * rSq; /* r6 */

                RealType fScal = 0;

                qq[STATE_A] = iqA * chargeA[jnr];
                qq[STATE_B] = iqB * chargeB[jnr];

                tj[STATE_A] = ntiA + 2 * typeA[jnr];
                tj[STATE_B] = ntiB + 2 * typeB[jnr];

                if (bPairIncluded)
                {
                    c6[STATE_A] = nbfp[tj[STATE_A]];
                    c6[STATE_B] = nbfp[tj[STATE_B]];

                    prepareVdwAndSoftcoreParameters(
                            c12, nbfp, tj, c6, sigma6, sigma6_min, sigma6_def, alphaVdwEff, alphaCoulEff, alpha_vdw, alpha_coul);

                    for (int i = 0; i < NSTATES; i++)
                    {
                        fScalC[i] = 0;
                        fScalV[i] = 0;
                        vCoul[i]  = 0;
                        vVdw[i]   = 0;

                        RealType rInvC, rInvV, rC, rV, rPInvC, rPInvV;

                        /* Only spend time on A or B state if it is non-zero */
                        if ((qq[i] != 0) || (c6[i] != 0) || (c12[i] != 0))
                        {
                            calculateRadii<scLambdasOrAlphasDiffer>(rPInvC,
                                                                    alphaCoulEff,
                                                                    lFacCoul,
                                                                    sigma6,
                                                                    rp,
                                                                    rInvC,
                                                                    rC,
                                                                    rPInvV,
                                                                    alphaVdwEff,
                                                                    lFacVdw,
                                                                    rInvV,
                                                                    rV,
                                                                    i);

                            /* Only process the coulomb interactions if we have charges,
                             * and if we either include all entries in the list (no cutoff
                             * used in the kernel), or if we are within the cutoff.
                             */
                            bool computeElecInteraction =
                                    (elecInteractionTypeIsEwald && r < kv.rCoulomb)
                                    || (!elecInteractionTypeIsEwald && rC < kv.rCoulomb);

                            if ((qq[i] != 0) && computeElecInteraction)
                            {
                                softcoreCalculatorShared_->calculateCoulombInteraction<elecInteractionTypeIsEwald>(
                                        vCoul, qq[i], rInvC, kv.sh_ewald, fScalC, rC, kv.krf, kv.crf, i);
                            }

                            /* Only process the VDW interactions if we have
                             * some non-zero parameters, and if we either
                             * include all entries in the list (no cutoff used
                             * in the kernel), or if we are within the cutoff.
                             */
                            bool computeVdwInteraction = (vdwInteractionTypeIsEwald && r < kv.rVdw)
                                                         || (!vdwInteractionTypeIsEwald && rV < kv.rVdw);
                            if ((c6[i] != 0 || c12[i] != 0) && computeVdwInteraction)
                            {
                                prepareForVdwInteraction(
                                        rPInvV, c6[i], c12[i], vVdw, kv.repulsionShift, kv.dispersionShift, fScalV, i);
                                softcoreCalculatorShared_->calculateVdwInteraction<vdwInteractionTypeIsEwald, vdwModifierIsPotSwitch>(
                                        vVdw,
                                        nbfp_grid[tj[i]],
                                        kv.shLjEwald,
                                        rV,
                                        ic.rvdw_switch,
                                        kv.vdw_swV3,
                                        kv.vdw_swV4,
                                        kv.vdw_swV5,
                                        kv.vdw_swF2,
                                        kv.vdw_swF3,
                                        kv.vdw_swF4,
                                        fScalV,
                                        kv.rVdw,
                                        i);
                            }

                            /* fScalC (and fScalV) now contain: dV/drC * rC
                             * Now we multiply by rC^-p, so it will be: dV/drC * rC^1-p
                             * Further down we first multiply by r^p-2 and then by
                             * the vector r, which in total gives: dV/drC * (r/rC)^1-p
                             */
                            fScalC[i] *= rPInvC;
                            fScalV[i] *= rPInvV;
                        }
                    } // end for (int i = 0; i < NSTATES; i++)

                    assembleAAndBStates(vCTot,
                                        kv.LFC,
                                        vCoul,
                                        vVTot,
                                        kv.LFV,
                                        vVdw,
                                        fScal,
                                        fScalC,
                                        fScalV,
                                        rpm2,
                                        kv.dvdlCoul,
                                        kv.dvdlVdw,
                                        kv.DLF,
                                        alphaCoulEff,
                                        alphaVdwEff,
                                        dlFacCoul,
                                        dlFacVdw,
                                        sigma6);

                } // end if (bPairIncluded)
                else if (kv.icoul == NbkernelElecType::ReactionField)
                {
                    softcoreCalculatorShared_->calculateCoulombRF(
                            kv.krf, rSq, kv.crf, vCTot, fScal, kv.dvdlCoul, qq, kv.LFC, kv.DLF, ii, jnr);
                }

                softcoreCalculatorShared_->calculateCoulombEwald<DataTypes, elecInteractionTypeIsEwald>(
                        r,
                        kv.rCoulomb,
                        kv.coulombTableScale,
                        kv.ewtab,
                        kv.coulombTableScaleInvHalf,
                        rInv,
                        vCTot,
                        fScal,
                        kv.dvdlCoul,
                        qq,
                        kv.LFC,
                        kv.DLF,
                        ii,
                        jnr,
                        bPairIncluded);
                softcoreCalculatorShared_->calculateVdwEwald<DataTypes, vdwInteractionTypeIsEwald>(
                        r,
                        kv.rVdw,
                        rSq,
                        rInv,
                        kv.vdwTableScale,
                        kv.tab_ewald_F_lj,
                        kv.tab_ewald_V_lj,
                        kv.vdwTableScaleInvHalf,
                        nbfp_grid,
                        tj,
                        vVTot,
                        fScal,
                        kv.dvdlVdw,
                        kv.LFV,
                        kv.DLF,
                        ii,
                        jnr,
                        bPairIncluded);
                softcoreCalculatorShared_->calculateForcesInnerLoop(
                        kv.f, dX, dY, dZ, fScal, fIX, fIY, fIZ, j3, kv.doForces);

            } // end for (int k = nj0; k < nj1; k++)

            softcoreCalculatorShared_->calculateForcesAndPotentialOuterLoop(kv.f,
                                                                            kv.fshift,
                                                                            fIX,
                                                                            fIY,
                                                                            fIZ,
                                                                            kv.gid,
                                                                            energygrp_elec,
                                                                            energygrp_vdw,
                                                                            vCTot,
                                                                            vVTot,
                                                                            ii3,
                                                                            is3,
                                                                            npair_within_cutoff,
                                                                            n,
                                                                            kv.doForces,
                                                                            kv.doShiftForces,
                                                                            kv.doPotential);

        } // end for (int n = 0; n < nri; n++)

#pragma omp atomic
        dvdl[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)] += kv.dvdlCoul;
#pragma omp atomic
        dvdl[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)] += kv.dvdlVdw;

        /* Estimate flops, average for free energy stuff:
         * 12  flops per outer iteration
         * 150 flops per inner iteration
         */
        atomicNrnbIncrement(nrnb, eNR_NBKERNEL_FREE_ENERGY, nlist.nri * 12 + nlist.jindex[kv.nri] * 150);

        if (kv.numExcludedPairsBeyondRlist > 0)
        {
            gmx_fatal(FARGS,
                      "There are %d perturbed non-bonded pair interactions beyond the pair-list "
                      "cutoff "
                      "of %g nm, which is not supported. This can happen because the system is "
                      "unstable or because intra-molecular interactions at long distances are "
                      "excluded. If the "
                      "latter is the case, you can try to increase nstlist or rlist to avoid this."
                      "The error is likely triggered by the use of couple-intramol=no "
                      "and the maximal distance in the decoupled molecule exceeding rlist.",
                      kv.numExcludedPairsBeyondRlist,
                      rlist);
        }
    }

private:
    std::unique_ptr<SoftcoreCalculatorShared> softcoreCalculatorShared_;
};


//! Templating machinery to select the desired nonbonded kernel
template<bool scLambdasOrAlphasDiffer, bool vdwInteractionTypeIsEwald, bool elecInteractionTypeIsEwald, bool vdwModifierIsPotSwitch, class SoftcoreCalculator>
static void dispatchKernelOnUseSimd(const t_nblist&                nlist,
                                    gmx::ArrayRef<const gmx::RVec> coords,
                                    gmx::ForceWithShiftForces*     forceWithShiftForces,
                                    const int                      ntype,
                                    const real                     rlist,
                                    const interaction_const_t&     ic,
                                    gmx::ArrayRef<const gmx::RVec> shiftvec,
                                    gmx::ArrayRef<const real>      nbfp,
                                    gmx::ArrayRef<const real>      nbfp_grid,
                                    gmx::ArrayRef<const real>      chargeA,
                                    gmx::ArrayRef<const real>      chargeB,
                                    gmx::ArrayRef<const int>       typeA,
                                    gmx::ArrayRef<const int>       typeB,
                                    int                            flags,
                                    gmx::ArrayRef<const real>      lambda,
                                    gmx::ArrayRef<real>            dvdl,
                                    gmx::ArrayRef<real>            energygrp_elec,
                                    gmx::ArrayRef<real>            energygrp_vdw,
                                    t_nrnb* gmx_restrict nrnb,
                                    const bool           useSimd,
                                    SoftcoreCalculator&  softcoreCalculator)
{
    if (useSimd)
    {
#if GMX_SIMD_HAVE_REAL && GMX_SIMD_HAVE_INT32_ARITHMETICS && GMX_USE_SIMD_KERNELS
        /* FIXME: Here SimdDataTypes should be used to enable SIMD. So far, the code in NBFreeEnergyKernel is not adapted to SIMD */
        softcoreCalculator.template NBFreeEnergyKernel<ScalarDataTypes, scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, vdwModifierIsPotSwitch>(
                nlist,
                coords,
                forceWithShiftForces,
                ntype,
                rlist,
                ic,
                shiftvec,
                nbfp,
                nbfp_grid,
                chargeA,
                chargeB,
                typeA,
                typeB,
                flags,
                lambda,
                dvdl,
                energygrp_elec,
                energygrp_vdw,
                nrnb);
#else
        softcoreCalculator.template NBFreeEnergyKernel<ScalarDataTypes, scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, vdwModifierIsPotSwitch>(
                nlist,
                coords,
                forceWithShiftForces,
                ntype,
                rlist,
                ic,
                shiftvec,
                nbfp,
                nbfp_grid,
                chargeA,
                chargeB,
                typeA,
                typeB,
                flags,
                lambda,
                dvdl,
                energygrp_elec,
                energygrp_vdw,
                nrnb);
#endif
    }
    else
    {
        softcoreCalculator.template NBFreeEnergyKernel<ScalarDataTypes, scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, vdwModifierIsPotSwitch>(
                nlist,
                coords,
                forceWithShiftForces,
                ntype,
                rlist,
                ic,
                shiftvec,
                nbfp,
                nbfp_grid,
                chargeA,
                chargeB,
                typeA,
                typeB,
                flags,
                lambda,
                dvdl,
                energygrp_elec,
                energygrp_vdw,
                nrnb);
    }
}

template<bool scLambdasOrAlphasDiffer, bool vdwInteractionTypeIsEwald, bool elecInteractionTypeIsEwald, class SoftcoreCalculator>
static void dispatchKernelOnVdwModifier(const t_nblist&                nlist,
                                        gmx::ArrayRef<const gmx::RVec> coords,
                                        gmx::ForceWithShiftForces*     forceWithShiftForces,
                                        const int                      ntype,
                                        const real                     rlist,
                                        const interaction_const_t&     ic,
                                        gmx::ArrayRef<const gmx::RVec> shiftvec,
                                        gmx::ArrayRef<const real>      nbfp,
                                        gmx::ArrayRef<const real>      nbfp_grid,
                                        gmx::ArrayRef<const real>      chargeA,
                                        gmx::ArrayRef<const real>      chargeB,
                                        gmx::ArrayRef<const int>       typeA,
                                        gmx::ArrayRef<const int>       typeB,
                                        int                            flags,
                                        gmx::ArrayRef<const real>      lambda,
                                        gmx::ArrayRef<real>            dvdl,
                                        gmx::ArrayRef<real>            energygrp_elec,
                                        gmx::ArrayRef<real>            energygrp_vdw,
                                        t_nrnb* gmx_restrict nrnb,
                                        const bool           vdwModifierIsPotSwitch,
                                        const bool           useSimd,
                                        SoftcoreCalculator&  softcoreCalculator)
{
    if (vdwModifierIsPotSwitch)
    {
        dispatchKernelOnUseSimd<scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, true>(
                nlist,
                coords,
                forceWithShiftForces,
                ntype,
                rlist,
                ic,
                shiftvec,
                nbfp,
                nbfp_grid,
                chargeA,
                chargeB,
                typeA,
                typeB,
                flags,
                lambda,
                dvdl,
                energygrp_elec,
                energygrp_vdw,
                nrnb,
                useSimd,
                softcoreCalculator);
    }
    else
    {
        dispatchKernelOnUseSimd<scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, false>(
                nlist,
                coords,
                forceWithShiftForces,
                ntype,
                rlist,
                ic,
                shiftvec,
                nbfp,
                nbfp_grid,
                chargeA,
                chargeB,
                typeA,
                typeB,
                flags,
                lambda,
                dvdl,
                energygrp_elec,
                energygrp_vdw,
                nrnb,
                useSimd,
                softcoreCalculator);
    }
}

template<bool scLambdasOrAlphasDiffer, bool vdwInteractionTypeIsEwald, class SoftcoreCalculator>
static void dispatchKernelOnElecInteractionType(const t_nblist&                nlist,
                                                gmx::ArrayRef<const gmx::RVec> coords,
                                                gmx::ForceWithShiftForces*     forceWithShiftForces,
                                                const int                      ntype,
                                                const real                     rlist,
                                                const interaction_const_t&     ic,
                                                gmx::ArrayRef<const gmx::RVec> shiftvec,
                                                gmx::ArrayRef<const real>      nbfp,
                                                gmx::ArrayRef<const real>      nbfp_grid,
                                                gmx::ArrayRef<const real>      chargeA,
                                                gmx::ArrayRef<const real>      chargeB,
                                                gmx::ArrayRef<const int>       typeA,
                                                gmx::ArrayRef<const int>       typeB,
                                                int                            flags,
                                                gmx::ArrayRef<const real>      lambda,
                                                gmx::ArrayRef<real>            dvdl,
                                                gmx::ArrayRef<real>            energygrp_elec,
                                                gmx::ArrayRef<real>            energygrp_vdw,
                                                t_nrnb* gmx_restrict nrnb,
                                                const bool           elecInteractionTypeIsEwald,
                                                const bool           vdwModifierIsPotSwitch,
                                                const bool           useSimd,
                                                SoftcoreCalculator&  softcoreCalculator)
{
    if (elecInteractionTypeIsEwald)
    {
        dispatchKernelOnVdwModifier<scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, true>(
                nlist,
                coords,
                forceWithShiftForces,
                ntype,
                rlist,
                ic,
                shiftvec,
                nbfp,
                nbfp_grid,
                chargeA,
                chargeB,
                typeA,
                typeB,
                flags,
                lambda,
                dvdl,
                energygrp_elec,
                energygrp_vdw,
                nrnb,
                vdwModifierIsPotSwitch,
                useSimd,
                softcoreCalculator);
    }
    else
    {
        dispatchKernelOnVdwModifier<scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, false>(
                nlist,
                coords,
                forceWithShiftForces,
                ntype,
                rlist,
                ic,
                shiftvec,
                nbfp,
                nbfp_grid,
                chargeA,
                chargeB,
                typeA,
                typeB,
                flags,
                lambda,
                dvdl,
                energygrp_elec,
                energygrp_vdw,
                nrnb,
                vdwModifierIsPotSwitch,
                useSimd,
                softcoreCalculator);
    }
}

template<bool scLambdasOrAlphasDiffer, class SoftcoreCalculator>
static void dispatchKernelOnVdwInteractionType(const t_nblist&                nlist,
                                               gmx::ArrayRef<const gmx::RVec> coords,
                                               gmx::ForceWithShiftForces*     forceWithShiftForces,
                                               const int                      ntype,
                                               const real                     rlist,
                                               const interaction_const_t&     ic,
                                               gmx::ArrayRef<const gmx::RVec> shiftvec,
                                               gmx::ArrayRef<const real>      nbfp,
                                               gmx::ArrayRef<const real>      nbfp_grid,
                                               gmx::ArrayRef<const real>      chargeA,
                                               gmx::ArrayRef<const real>      chargeB,
                                               gmx::ArrayRef<const int>       typeA,
                                               gmx::ArrayRef<const int>       typeB,
                                               int                            flags,
                                               gmx::ArrayRef<const real>      lambda,
                                               gmx::ArrayRef<real>            dvdl,
                                               gmx::ArrayRef<real>            energygrp_elec,
                                               gmx::ArrayRef<real>            energygrp_vdw,
                                               t_nrnb* gmx_restrict nrnb,
                                               const bool           vdwInteractionTypeIsEwald,
                                               const bool           elecInteractionTypeIsEwald,
                                               const bool           vdwModifierIsPotSwitch,
                                               const bool           useSimd,
                                               SoftcoreCalculator&  softcoreCalculator)
{
    if (vdwInteractionTypeIsEwald)
    {
        dispatchKernelOnElecInteractionType<scLambdasOrAlphasDiffer, true>(nlist,
                                                                           coords,
                                                                           forceWithShiftForces,
                                                                           ntype,
                                                                           rlist,
                                                                           ic,
                                                                           shiftvec,
                                                                           nbfp,
                                                                           nbfp_grid,
                                                                           chargeA,
                                                                           chargeB,
                                                                           typeA,
                                                                           typeB,
                                                                           flags,
                                                                           lambda,
                                                                           dvdl,
                                                                           energygrp_elec,
                                                                           energygrp_vdw,
                                                                           nrnb,
                                                                           elecInteractionTypeIsEwald,
                                                                           vdwModifierIsPotSwitch,
                                                                           useSimd,
                                                                           softcoreCalculator);
    }
    else
    {
        dispatchKernelOnElecInteractionType<scLambdasOrAlphasDiffer, false>(nlist,
                                                                            coords,
                                                                            forceWithShiftForces,
                                                                            ntype,
                                                                            rlist,
                                                                            ic,
                                                                            shiftvec,
                                                                            nbfp,
                                                                            nbfp_grid,
                                                                            chargeA,
                                                                            chargeB,
                                                                            typeA,
                                                                            typeB,
                                                                            flags,
                                                                            lambda,
                                                                            dvdl,
                                                                            energygrp_elec,
                                                                            energygrp_vdw,
                                                                            nrnb,
                                                                            elecInteractionTypeIsEwald,
                                                                            vdwModifierIsPotSwitch,
                                                                            useSimd,
                                                                            softcoreCalculator);
    }
}

template<class SoftcoreCalculator>
static void dispatchKernelOnScLambdasOrAlphasDifference(const t_nblist&                nlist,
                                                        gmx::ArrayRef<const gmx::RVec> coords,
                                                        gmx::ForceWithShiftForces* forceWithShiftForces,
                                                        const int                  ntype,
                                                        const real                 rlist,
                                                        const interaction_const_t&     ic,
                                                        gmx::ArrayRef<const gmx::RVec> shiftvec,
                                                        gmx::ArrayRef<const real>      nbfp,
                                                        gmx::ArrayRef<const real>      nbfp_grid,
                                                        gmx::ArrayRef<const real>      chargeA,
                                                        gmx::ArrayRef<const real>      chargeB,
                                                        gmx::ArrayRef<const int>       typeA,
                                                        gmx::ArrayRef<const int>       typeB,
                                                        int                            flags,
                                                        gmx::ArrayRef<const real>      lambda,
                                                        gmx::ArrayRef<real>            dvdl,
                                                        gmx::ArrayRef<real> energygrp_elec,
                                                        gmx::ArrayRef<real> energygrp_vdw,
                                                        t_nrnb* gmx_restrict nrnb,
                                                        const bool scLambdasOrAlphasDiffer,
                                                        const bool vdwInteractionTypeIsEwald,
                                                        const bool elecInteractionTypeIsEwald,
                                                        const bool vdwModifierIsPotSwitch,
                                                        const bool useSimd,
                                                        SoftcoreCalculator& softcoreCalculator)
{
    if (scLambdasOrAlphasDiffer)
    {
        dispatchKernelOnVdwInteractionType<true>(nlist,
                                                 coords,
                                                 forceWithShiftForces,
                                                 ntype,
                                                 rlist,
                                                 ic,
                                                 shiftvec,
                                                 nbfp,
                                                 nbfp_grid,
                                                 chargeA,
                                                 chargeB,
                                                 typeA,
                                                 typeB,
                                                 flags,
                                                 lambda,
                                                 dvdl,
                                                 energygrp_elec,
                                                 energygrp_vdw,
                                                 nrnb,
                                                 vdwInteractionTypeIsEwald,
                                                 elecInteractionTypeIsEwald,
                                                 vdwModifierIsPotSwitch,
                                                 useSimd,
                                                 softcoreCalculator);
    }
    else
    {
        dispatchKernelOnVdwInteractionType<false>(nlist,
                                                  coords,
                                                  forceWithShiftForces,
                                                  ntype,
                                                  rlist,
                                                  ic,
                                                  shiftvec,
                                                  nbfp,
                                                  nbfp_grid,
                                                  chargeA,
                                                  chargeB,
                                                  typeA,
                                                  typeB,
                                                  flags,
                                                  lambda,
                                                  dvdl,
                                                  energygrp_elec,
                                                  energygrp_vdw,
                                                  nrnb,
                                                  vdwInteractionTypeIsEwald,
                                                  elecInteractionTypeIsEwald,
                                                  vdwModifierIsPotSwitch,
                                                  useSimd,
                                                  softcoreCalculator);
    }
}

static void dispatchKernel(const t_nblist&                nlist,
                           gmx::ArrayRef<const gmx::RVec> coords,
                           gmx::ForceWithShiftForces*     forceWithShiftForces,
                           const int                      ntype,
                           const real                     rlist,
                           const interaction_const_t&     ic,
                           gmx::ArrayRef<const gmx::RVec> shiftvec,
                           gmx::ArrayRef<const real>      nbfp,
                           gmx::ArrayRef<const real>      nbfp_grid,
                           gmx::ArrayRef<const real>      chargeA,
                           gmx::ArrayRef<const real>      chargeB,
                           gmx::ArrayRef<const int>       typeA,
                           gmx::ArrayRef<const int>       typeB,
                           int                            flags,
                           gmx::ArrayRef<const real>      lambda,
                           gmx::ArrayRef<real>            dvdl,
                           gmx::ArrayRef<real>            energygrp_elec,
                           gmx::ArrayRef<real>            energygrp_vdw,
                           t_nrnb* gmx_restrict nrnb,
                           const bool           scLambdasOrAlphasDiffer,
                           const bool           vdwInteractionTypeIsEwald,
                           const bool           elecInteractionTypeIsEwald,
                           const bool           vdwModifierIsPotSwitch,
                           const bool           useSimd)
{
    // TODO select the kind of soft-core potential to be used on the basis of an enum reflecting user input
    if (ic.softCoreParameters->alphaCoulomb == 0 && ic.softCoreParameters->alphaVdw == 0)
    {
        SoftcoreCalculatorNone softcoreCalculatorNone;
        dispatchKernelOnScLambdasOrAlphasDifference(nlist,
                                                    coords,
                                                    forceWithShiftForces,
                                                    ntype,
                                                    rlist,
                                                    ic,
                                                    shiftvec,
                                                    nbfp,
                                                    nbfp_grid,
                                                    chargeA,
                                                    chargeB,
                                                    typeA,
                                                    typeB,
                                                    flags,
                                                    lambda,
                                                    dvdl,
                                                    energygrp_elec,
                                                    energygrp_vdw,
                                                    nrnb,
                                                    scLambdasOrAlphasDiffer,
                                                    vdwInteractionTypeIsEwald,
                                                    elecInteractionTypeIsEwald,
                                                    vdwModifierIsPotSwitch,
                                                    useSimd,
                                                    softcoreCalculatorNone);
    }
    else
    {
        SoftcoreCalculatorBeutler softcoreCalculatorBeutler;
        dispatchKernelOnScLambdasOrAlphasDifference(nlist,
                                                    coords,
                                                    forceWithShiftForces,
                                                    ntype,
                                                    rlist,
                                                    ic,
                                                    shiftvec,
                                                    nbfp,
                                                    nbfp_grid,
                                                    chargeA,
                                                    chargeB,
                                                    typeA,
                                                    typeB,
                                                    flags,
                                                    lambda,
                                                    dvdl,
                                                    energygrp_elec,
                                                    energygrp_vdw,
                                                    nrnb,
                                                    scLambdasOrAlphasDiffer,
                                                    vdwInteractionTypeIsEwald,
                                                    elecInteractionTypeIsEwald,
                                                    vdwModifierIsPotSwitch,
                                                    useSimd,
                                                    softcoreCalculatorBeutler);
    }
}


void gmx_nb_free_energy_kernel(const t_nblist&                nlist,
                               gmx::ArrayRef<const gmx::RVec> coords,
                               gmx::ForceWithShiftForces*     ff,
                               const bool                     useSimd,
                               const int                      ntype,
                               const real                     rlist,
                               const interaction_const_t&     ic,
                               gmx::ArrayRef<const gmx::RVec> shiftvec,
                               gmx::ArrayRef<const real>      nbfp,
                               gmx::ArrayRef<const real>      nbfp_grid,
                               gmx::ArrayRef<const real>      chargeA,
                               gmx::ArrayRef<const real>      chargeB,
                               gmx::ArrayRef<const int>       typeA,
                               gmx::ArrayRef<const int>       typeB,
                               int                            flags,
                               gmx::ArrayRef<const real>      lambda,
                               gmx::ArrayRef<real>            dvdl,
                               gmx::ArrayRef<real>            energygrp_elec,
                               gmx::ArrayRef<real>            energygrp_vdw,
                               t_nrnb*                        nrnb)
{
    GMX_ASSERT(EEL_PME_EWALD(ic.eeltype) || ic.eeltype == CoulombInteractionType::Cut || EEL_RF(ic.eeltype),
               "Unsupported eeltype with free energy");
    GMX_ASSERT(ic.softCoreParameters, "We need soft-core parameters");

    const auto& scParams                   = *ic.softCoreParameters;
    const bool  vdwInteractionTypeIsEwald  = (EVDW_PME(ic.vdwtype));
    const bool  elecInteractionTypeIsEwald = (EEL_PME_EWALD(ic.eeltype));
    const bool  vdwModifierIsPotSwitch     = (ic.vdw_modifier == InteractionModifiers::PotSwitch);
    bool        scLambdasOrAlphasDiffer    = true;

    if (scParams.alphaCoulomb == 0 && scParams.alphaVdw == 0)
    {
        scLambdasOrAlphasDiffer = false;
    }
    else
    {
        if (lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Coul)]
                    == lambda[static_cast<int>(FreeEnergyPerturbationCouplingType::Vdw)]
            && scParams.alphaCoulomb == scParams.alphaVdw)
        {
            scLambdasOrAlphasDiffer = false;
        }
    }

    dispatchKernel(nlist,
                   coords,
                   ff,
                   ntype,
                   rlist,
                   ic,
                   shiftvec,
                   nbfp,
                   nbfp_grid,
                   chargeA,
                   chargeB,
                   typeA,
                   typeB,
                   flags,
                   lambda,
                   dvdl,
                   energygrp_elec,
                   energygrp_vdw,
                   nrnb,
                   scLambdasOrAlphasDiffer,
                   vdwInteractionTypeIsEwald,
                   elecInteractionTypeIsEwald,
                   vdwModifierIsPotSwitch,
                   useSimd);
}
