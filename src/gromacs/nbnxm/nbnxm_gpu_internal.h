/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2017 by the GROMACS development team.
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
/*! \libinternal \file
 *  \brief Declares functions that are only called from NBNXM GPU code.
 *
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 *  \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_NBNXM_GPU_INTERNAL_H
#define GMX_NBNXM_NBNXM_GPU_INTERNAL_H

#include "gromacs/hardware/device_information.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/range.h"

#if GMX_GPU_CUDA
#    include "gromacs/nbnxm/cuda/nbnxm_cuda_types.h"
#elif GMX_GPU_OPENCL
#    include "gromacs/nbnxm/opencl/nbnxm_ocl_types.h"
#elif GMX_GPU_SYCL
#    include "gromacs/nbnxm/sycl/nbnxm_sycl_types.h"
#endif

namespace gmx
{
}

namespace Nbnxm
{

/*! \brief Returns true if LJ combination rules are used in the non-bonded kernels.
 *
 *  \param[in] vdwType  The VdW interaction/implementation type as defined by VdwType
 *                      enumeration.
 *
 * \returns Whether combination rules are used by the run.
 */
static inline bool useLjCombRule(const enum VdwType vdwType)
{
    return (vdwType == VdwType::CutCombGeom || vdwType == VdwType::CutCombLB);
}


/*! \brief Calculate atom range and return start index and length.
 *
 * \param[in] atomData Atom descriptor data structure
 * \param[in] atomLocality Atom locality specifier
 * \returns Range of indexes for selected locality.
 */
static inline gmx::Range<int> getGpuAtomRange(const NBAtomDataGpu* atomData, const gmx::AtomLocality atomLocality)
{
    assert(atomData);

    /* calculate the atom data index range based on locality */
    if (atomLocality == AtomLocality::Local)
    {
        return gmx::Range<int>(0, atomData->numAtomsLocal);
    }
    else if (atomLocality == AtomLocality::NonLocal)
    {
        return gmx::Range<int>(atomData->numAtomsLocal, atomData->numAtoms);
    }
    else
    {
        GMX_THROW(gmx::InconsistentInputError(
                "Only Local and NonLocal atom locities can be used to get atom ranges in NBNXM."));
    }
}

static inline void issueClFlushInStream(const DeviceStream& deviceStream)
{
#if GMX_GPU_OPENCL
    /* Based on the v1.2 section 5.13 of the OpenCL spec, a flush is needed
     * in the stream after marking an event in it in order to be able to sync with
     * the event from another stream.
     */
    cl_int cl_error = clFlush(deviceStream.stream());
    if (cl_error != CL_SUCCESS)
    {
        GMX_THROW(gmx::InternalError("clFlush failed: " + ocl_get_error_string(cl_error)));
    }
#else
    GMX_UNUSED_VALUE(deviceStream);
#endif
}

static inline void init_timings(gmx_wallclock_gpu_nbnxn_t* t)
{
    t->nb_h2d_t = 0.0;
    t->nb_d2h_t = 0.0;
    t->nb_c     = 0;
    t->pl_h2d_t = 0.0;
    t->pl_h2d_c = 0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t->ktime[i][j].t = 0.0;
            t->ktime[i][j].c = 0;
        }
    }
    t->pruneTime.c        = 0;
    t->pruneTime.t        = 0.0;
    t->dynamicPruneTime.c = 0;
    t->dynamicPruneTime.t = 0.0;
}

static inline void init_ewald_coulomb_force_table(const EwaldCorrectionTables& tables,
                                                  NBParamGpu*                  nbp,
                                                  const DeviceContext&         deviceContext)
{
    if (nbp->coulomb_tab)
    {
        destroyParamLookupTable(&nbp->coulomb_tab, nbp->coulomb_tab_texobj);
    }

    nbp->coulomb_tab_scale = tables.scale;
    initParamLookupTable(
            &nbp->coulomb_tab, &nbp->coulomb_tab_texobj, tables.tableF.data(), tables.tableF.size(), deviceContext);
}

static inline ElecType nbnxn_gpu_pick_ewald_kernel_type(const interaction_const_t& ic,
                                                        const DeviceInformation gmx_unused& deviceInfo)
{
    bool bTwinCut = (ic.rcoulomb != ic.rvdw);

    /* Benchmarking/development environment variables to force the use of
       analytical or tabulated Ewald kernel. */
    const bool forceAnalyticalEwald = (getenv("GMX_GPU_NB_ANA_EWALD") != nullptr);
    const bool forceTabulatedEwald  = (getenv("GMX_GPU_NB_TAB_EWALD") != nullptr);
    const bool forceTwinCutoffEwald = (getenv("GMX_GPU_NB_EWALD_TWINCUT") != nullptr);

    if (forceAnalyticalEwald && forceTabulatedEwald)
    {
        gmx_incons(
                "Both analytical and tabulated Ewald GPU non-bonded kernels "
                "requested through environment variables.");
    }

    /* By default, use analytical Ewald except with CUDA on NVIDIA CC 7.0 and 8.0.
     */
    const bool c_useTabulatedEwaldDefault =
#if GMX_GPU_CUDA
            (deviceInfo.prop.major == 7 && deviceInfo.prop.minor == 0)
            || (deviceInfo.prop.major == 8 && deviceInfo.prop.minor == 0);
#else
            false;
#endif
    bool bUseAnalyticalEwald = !c_useTabulatedEwaldDefault;
    if (forceAnalyticalEwald)
    {
        bUseAnalyticalEwald = true;
        if (debug)
        {
            fprintf(debug, "Using analytical Ewald GPU kernels\n");
        }
    }
    else if (forceTabulatedEwald)
    {
        bUseAnalyticalEwald = false;

        if (debug)
        {
            fprintf(debug, "Using tabulated Ewald GPU kernels\n");
        }
    }

    /* Use twin cut-off kernels if requested by bTwinCut or the env. var.
       forces it (use it for debugging/benchmarking only). */
    if (!bTwinCut && !forceTwinCutoffEwald)
    {
        return bUseAnalyticalEwald ? ElecType::EwaldAna : ElecType::EwaldTab;
    }
    else
    {
        return bUseAnalyticalEwald ? ElecType::EwaldAnaTwin : ElecType::EwaldTabTwin;
    }
}


static inline void set_cutoff_parameters(NBParamGpu*                nbp,
                                         const interaction_const_t& ic,
                                         const PairlistParams&      listParams)
{
    nbp->ewald_beta        = ic.ewaldcoeff_q;
    nbp->sh_ewald          = ic.sh_ewald;
    nbp->epsfac            = ic.epsfac;
    nbp->two_k_rf          = 2.0 * ic.reactionFieldCoefficient;
    nbp->c_rf              = ic.reactionFieldShift;
    nbp->rvdw_sq           = ic.rvdw * ic.rvdw;
    nbp->rcoulomb_sq       = ic.rcoulomb * ic.rcoulomb;
    nbp->rlistOuter_sq     = listParams.rlistOuter * listParams.rlistOuter;
    nbp->rlistInner_sq     = listParams.rlistInner * listParams.rlistInner;
    nbp->useDynamicPruning = listParams.useDynamicPruning;

    nbp->sh_lj_ewald   = ic.sh_lj_ewald;
    nbp->ewaldcoeff_lj = ic.ewaldcoeff_lj;

    nbp->rvdw_switch      = ic.rvdw_switch;
    nbp->dispersion_shift = ic.dispersion_shift;
    nbp->repulsion_shift  = ic.repulsion_shift;
    nbp->vdw_switch       = ic.vdw_switch;
}

/*! \brief Sync the nonlocal stream with dependent tasks in the local queue.
 *
 *  As the point where the local stream tasks can be considered complete happens
 *  at the same call point where the nonlocal stream should be synced with the
 *  the local, this function records the event if called with the local stream as
 *  argument and inserts in the GPU stream a wait on the event on the nonlocal.
 *
 * \param[in] nb                   The nonbonded data GPU structure
 * \param[in] interactionLocality  Local or NonLocal sync point
 */
void nbnxnInsertNonlocalGpuDependency(NbnxmGpu gmx_unused*     nb,
                                      gmx::InteractionLocality gmx_unused interactionLocality);

/*! \brief An early return condition for empty NB GPU workloads
 *
 * This is currently used for non-local kernels/transfers only.
 * Skipping the local kernel is more complicated, since the
 * local part of the force array also depends on the non-local kernel.
 * The skip of the local kernel is taken care of separately.
 */
static inline bool canSkipNonbondedWork(const NbnxmGpu& nb, InteractionLocality iloc)
{
    assert(nb.plist[iloc]);
    return (iloc == InteractionLocality::NonLocal && nb.plist[iloc]->nsci == 0);
}

/*! \brief Initializes the NBNXM GPU data structures. */
void gpu_init_platform_specific(NbnxmGpu* nb);

/*! \brief Releases the NBNXM GPU data structures. */
void gpu_free_platform_specific(NbnxmGpu* nb);

} // namespace Nbnxm

#endif // GMX_NBNXM_NBNXM_GPU_INTERNAL_H
