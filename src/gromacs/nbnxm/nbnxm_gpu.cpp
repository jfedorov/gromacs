/*
 * This file is part of the GROMACS molecular simulation package.
 *
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
/*! \internal \file
 *  \brief Define common implementations of GPU NBNXM data management.
 *
 *  \author Anca Hamuraru <anca@streamcomputing.eu>
 *  \author Dimitrios Karkoulis <dimitris.karkoulis@gmail.com>
 *  \author Teemu Virolainen <teemu@streamcomputing.eu>
 *  \author Szilárd Páll <pall.szilard@gmail.com>
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "config.h"

#if GMX_GPU_CUDA
#    include "cuda/nbnxm_cuda_types.h"
#endif

#if GMX_GPU_OPENCL
#    include "opencl/nbnxm_ocl_types.h"
#endif

#if GMX_GPU_SYCL
#    include "sycl/nbnxm_sycl_types.h"
#endif

#include "nbnxm_gpu_internal.h"

#include "gromacs/gpu_utils/device_stream_manager.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/gpu_utils/pmalloc.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/listed_forces/gpubonded.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/gridset.h"
#include "gromacs/nbnxm/nbnxm_gpu_public.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"

#include "nbnxm_gpu.h"
#include "pairlistsets.h"

namespace Nbnxm
{

static inline void init_plist(gpu_plist* pl)
{
    /* initialize to nullptr pointers to data that is not allocated here and will
       need reallocation in nbnxn_gpu_init_pairlist */
    pl->sci   = nullptr;
    pl->cj4   = nullptr;
    pl->imask = nullptr;
    pl->excl  = nullptr;

    /* size -1 indicates that the respective array hasn't been initialized yet */
    pl->na_c                   = -1;
    pl->nsci                   = -1;
    pl->sci_nalloc             = -1;
    pl->ncj4                   = -1;
    pl->cj4_nalloc             = -1;
    pl->nimask                 = -1;
    pl->imask_nalloc           = -1;
    pl->nexcl                  = -1;
    pl->excl_nalloc            = -1;
    pl->haveFreshList          = false;
    pl->rollingPruningNumParts = 0;
    pl->rollingPruningPart     = 0;
}

/*! \brief Initialize \p atomdata first time; it only gets filled at pair-search. */
static inline void initAtomdataFirst(NBAtomDataGpu*       atomdata,
                                     int                  numTypes,
                                     const DeviceContext& deviceContext,
                                     const DeviceStream&  localStream)
{
    atomdata->numTypes = numTypes;
    allocateDeviceBuffer(&atomdata->shiftVec, gmx::c_numShiftVectors, deviceContext);
    atomdata->shiftVecUploaded = false;

    allocateDeviceBuffer(&atomdata->fShift, gmx::c_numShiftVectors, deviceContext);
    allocateDeviceBuffer(&atomdata->eLJ, 1, deviceContext);
    allocateDeviceBuffer(&atomdata->eElec, 1, deviceContext);

    clearDeviceBufferAsync(&atomdata->fShift, 0, gmx::c_numShiftVectors, localStream);
    clearDeviceBufferAsync(&atomdata->eElec, 0, 1, localStream);
    clearDeviceBufferAsync(&atomdata->eLJ, 0, 1, localStream);

    /* initialize to nullptr pointers to data that is not allocated here and will
       need reallocation in later */
    atomdata->xq = nullptr;
    atomdata->f  = nullptr;

    /* size -1 indicates that the respective array hasn't been initialized yet */
    atomdata->numAtoms      = -1;
    atomdata->numAtomsAlloc = -1;
}

static inline VdwType nbnxmGpuPickVdwKernelType(const interaction_const_t& ic,
                                                LJCombinationRule          ljCombinationRule)
{
    if (ic.vdwtype == VanDerWaalsType::Cut)
    {
        switch (ic.vdw_modifier)
        {
            case InteractionModifiers::None:
            case InteractionModifiers::PotShift:
                switch (ljCombinationRule)
                {
                    case LJCombinationRule::None: return VdwType::Cut;
                    case LJCombinationRule::Geometric: return VdwType::CutCombGeom;
                    case LJCombinationRule::LorentzBerthelot: return VdwType::CutCombLB;
                    default:
                        GMX_THROW(gmx::InconsistentInputError(gmx::formatString(
                                "The requested LJ combination rule %s is not implemented in "
                                "the GPU accelerated kernels!",
                                enumValueToString(ljCombinationRule))));
                }
            case InteractionModifiers::ForceSwitch: return VdwType::FSwitch;
            case InteractionModifiers::PotSwitch: return VdwType::PSwitch;
            default:
                GMX_THROW(gmx::InconsistentInputError(
                        gmx::formatString("The requested VdW interaction modifier %s is not "
                                          "implemented in the GPU accelerated kernels!",
                                          enumValueToString(ic.vdw_modifier))));
        }
    }
    else if (ic.vdwtype == VanDerWaalsType::Pme)
    {
        if (ic.ljpme_comb_rule == LongRangeVdW::Geom)
        {
            GMX_RELEASE_ASSERT(
                    ljCombinationRule == LJCombinationRule::Geometric,
                    "Combination rules for long- and short-range interactions should match.");
            return VdwType::EwaldGeom;
        }
        else
        {
            GMX_RELEASE_ASSERT(
                    ljCombinationRule == LJCombinationRule::LorentzBerthelot,
                    "Combination rules for long- and short-range interactions should match.");
            return VdwType::EwaldLB;
        }
    }
    else
    {
        GMX_THROW(gmx::InconsistentInputError(gmx::formatString(
                "The requested VdW type %s is not implemented in the GPU accelerated kernels!",
                enumValueToString(ic.vdwtype))));
    }
}

static inline ElecType nbnxmGpuPickElectrostaticsKernelType(const interaction_const_t& ic,
                                                            const DeviceInformation&   deviceInfo)
{
    if (ic.eeltype == CoulombInteractionType::Cut)
    {
        return ElecType::Cut;
    }
    else if (EEL_RF(ic.eeltype))
    {
        return ElecType::RF;
    }
    else if ((EEL_PME(ic.eeltype) || ic.eeltype == CoulombInteractionType::Ewald))
    {
        return nbnxn_gpu_pick_ewald_kernel_type(ic, deviceInfo);
    }
    else
    {
        /* Shouldn't happen, as this is checked when choosing Verlet-scheme */
        GMX_THROW(gmx::InconsistentInputError(
                gmx::formatString("The requested electrostatics type %s is not implemented in "
                                  "the GPU accelerated kernels!",
                                  enumValueToString(ic.eeltype))));
    }
}

/*! \brief Initialize the nonbonded parameter data structure. */
static inline void initNbparam(NBParamGpu*                     nbp,
                               const interaction_const_t&      ic,
                               const PairlistParams&           listParams,
                               const nbnxn_atomdata_t::Params& nbatParams,
                               const DeviceContext&            deviceContext)
{
    const int numTypes = nbatParams.numTypes;

    set_cutoff_parameters(nbp, ic, listParams);

    nbp->vdwType  = nbnxmGpuPickVdwKernelType(ic, nbatParams.ljCombinationRule);
    nbp->elecType = nbnxmGpuPickElectrostaticsKernelType(ic, deviceContext.deviceInfo());

    if (ic.vdwtype == VanDerWaalsType::Pme)
    {
        if (ic.ljpme_comb_rule == LongRangeVdW::Geom)
        {
            GMX_ASSERT(nbatParams.ljCombinationRule == LJCombinationRule::Geometric,
                       "Combination rule mismatch!");
        }
        else
        {
            GMX_ASSERT(nbatParams.ljCombinationRule == LJCombinationRule::LorentzBerthelot,
                       "Combination rule mismatch!");
        }
    }

    /* generate table for PME */
    if (nbp->elecType == ElecType::EwaldTab || nbp->elecType == ElecType::EwaldTabTwin)
    {
        GMX_RELEASE_ASSERT(ic.coulombEwaldTables, "Need valid Coulomb Ewald correction tables");
        init_ewald_coulomb_force_table(*ic.coulombEwaldTables, nbp, deviceContext);
    }

    /* set up LJ parameter lookup table */
    if (!useLjCombRule(nbp->vdwType))
    {
        static_assert(sizeof(decltype(nbp->nbfp)) == 2 * sizeof(decltype(*nbatParams.nbfp.data())),
                      "Mismatch in the size of host / device data types");
        initParamLookupTable(&nbp->nbfp,
                             &nbp->nbfp_texobj,
                             reinterpret_cast<const Float2*>(nbatParams.nbfp.data()),
                             numTypes * numTypes,
                             deviceContext);
    }

    /* set up LJ-PME parameter lookup table */
    if (ic.vdwtype == VanDerWaalsType::Pme)
    {
        static_assert(sizeof(decltype(nbp->nbfp_comb))
                              == 2 * sizeof(decltype(*nbatParams.nbfp_comb.data())),
                      "Mismatch in the size of host / device data types");
        initParamLookupTable(&nbp->nbfp_comb,
                             &nbp->nbfp_comb_texobj,
                             reinterpret_cast<const Float2*>(nbatParams.nbfp_comb.data()),
                             numTypes,
                             deviceContext);
    }
}

NbnxmGpu* gpu_init(const gmx::DeviceStreamManager& deviceStreamManager,
                   const interaction_const_t*      ic,
                   const PairlistParams&           listParams,
                   const nbnxn_atomdata_t*         nbat,
                   const bool                      bLocalAndNonlocal)
{
    auto* nb                              = new NbnxmGpu();
    nb->deviceContext_                    = &deviceStreamManager.context();
    nb->atdat                             = new NBAtomDataGpu;
    nb->nbparam                           = new NBParamGpu;
    nb->plist[InteractionLocality::Local] = new Nbnxm::gpu_plist;
    if (bLocalAndNonlocal)
    {
        nb->plist[InteractionLocality::NonLocal] = new Nbnxm::gpu_plist;
    }

    nb->bUseTwoStreams = bLocalAndNonlocal;

    nb->timers = new Nbnxm::GpuTimers();
    snew(nb->timings, 1);

    /* WARNING: CUDA timings are incorrect with multiple streams.
     * This is the main reason why they are disabled by default.
     * Can be enabled by setting GMX_ENABLE_GPU_TIMING environment variable.
     * TODO: Consider turning on by default when we can detect nr of streams.
     *
     * OpenCL timing is enabled by default and can be disabled by
     * GMX_DISABLE_GPU_TIMING environment variable.
     *
     * Timing is disabled in SYCL.
     */
    nb->bDoTime = (GMX_GPU_CUDA && (getenv("GMX_ENABLE_GPU_TIMING") != nullptr))
                  || (GMX_GPU_OPENCL && (getenv("GMX_DISABLE_GPU_TIMING") == nullptr));

    if (nb->bDoTime)
    {
        init_timings(nb->timings);
    }

    /* init nbst */
    pmalloc(reinterpret_cast<void**>(&nb->nbst.eLJ), sizeof(*nb->nbst.eLJ));
    pmalloc(reinterpret_cast<void**>(&nb->nbst.eElec), sizeof(*nb->nbst.eElec));
    pmalloc(reinterpret_cast<void**>(&nb->nbst.fShift), gmx::c_numShiftVectors * sizeof(*nb->nbst.fShift));

    init_plist(nb->plist[InteractionLocality::Local]);

    /* local/non-local GPU streams */
    GMX_RELEASE_ASSERT(deviceStreamManager.streamIsValid(gmx::DeviceStreamType::NonBondedLocal),
                       "Local non-bonded stream should be initialized to use GPU for non-bonded.");
    const DeviceStream& localStream = deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal);
    nb->deviceStreams[InteractionLocality::Local] = &localStream;
    // In general, it's not strictly necessary to use 2 streams for SYCL, since they are
    // out-of-order. But for the time being, it will be less disruptive to keep them.
    if (nb->bUseTwoStreams)
    {
        init_plist(nb->plist[InteractionLocality::NonLocal]);

        GMX_RELEASE_ASSERT(deviceStreamManager.streamIsValid(gmx::DeviceStreamType::NonBondedNonLocal),
                           "Non-local non-bonded stream should be initialized to use GPU for "
                           "non-bonded with domain decomposition.");
        nb->deviceStreams[InteractionLocality::NonLocal] =
                &deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedNonLocal);
    }

    const nbnxn_atomdata_t::Params& nbatParams    = nbat->params();
    const DeviceContext&            deviceContext = *nb->deviceContext_;

    initNbparam(nb->nbparam, *ic, listParams, nbatParams, deviceContext);
    initAtomdataFirst(nb->atdat, nbatParams.numTypes, deviceContext, localStream);

    gpu_init_platform_specific(nb);

    if (debug)
    {
        fprintf(debug, "Initialized NBNXM GPU data structures.\n");
    }

    return nb;
}

//! This function is documented in the header file
void gpu_init_pairlist(NbnxmGpu* nb, const NbnxnPairlistGpu* h_plist, const InteractionLocality iloc)
{
    char sbuf[STRLEN];
    // Timing accumulation should happen only if there was work to do
    // because getLastRangeTime() gets skipped with empty lists later
    // which leads to the counter not being reset.
    bool                bDoTime      = (nb->bDoTime && !h_plist->sci.empty());
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];
    gpu_plist*          d_plist      = nb->plist[iloc];

    if (d_plist->na_c < 0)
    {
        d_plist->na_c = h_plist->na_ci;
    }
    else
    {
        if (d_plist->na_c != h_plist->na_ci)
        {
            sprintf(sbuf,
                    "In init_plist: the #atoms per cell has changed (from %d to %d)",
                    d_plist->na_c,
                    h_plist->na_ci);
            gmx_incons(sbuf);
        }
    }

    GpuTimers::Interaction& iTimers = nb->timers->interaction[iloc];

    if (bDoTime)
    {
        iTimers.pl_h2d.openTimingRegion(deviceStream);
        iTimers.didPairlistH2D = true;
    }

    // TODO most of this function is same in CUDA and OpenCL, move into the header
    const DeviceContext& deviceContext = *nb->deviceContext_;

    reallocateDeviceBuffer(
            &d_plist->sci, h_plist->sci.size(), &d_plist->nsci, &d_plist->sci_nalloc, deviceContext);
    copyToDeviceBuffer(&d_plist->sci,
                       h_plist->sci.data(),
                       0,
                       h_plist->sci.size(),
                       deviceStream,
                       GpuApiCallBehavior::Async,
                       bDoTime ? iTimers.pl_h2d.fetchNextEvent() : nullptr);

    reallocateDeviceBuffer(
            &d_plist->cj4, h_plist->cj4.size(), &d_plist->ncj4, &d_plist->cj4_nalloc, deviceContext);
    copyToDeviceBuffer(&d_plist->cj4,
                       h_plist->cj4.data(),
                       0,
                       h_plist->cj4.size(),
                       deviceStream,
                       GpuApiCallBehavior::Async,
                       bDoTime ? iTimers.pl_h2d.fetchNextEvent() : nullptr);

    reallocateDeviceBuffer(&d_plist->imask,
                           h_plist->cj4.size() * c_nbnxnGpuClusterpairSplit,
                           &d_plist->nimask,
                           &d_plist->imask_nalloc,
                           deviceContext);

    reallocateDeviceBuffer(
            &d_plist->excl, h_plist->excl.size(), &d_plist->nexcl, &d_plist->excl_nalloc, deviceContext);
    copyToDeviceBuffer(&d_plist->excl,
                       h_plist->excl.data(),
                       0,
                       h_plist->excl.size(),
                       deviceStream,
                       GpuApiCallBehavior::Async,
                       bDoTime ? iTimers.pl_h2d.fetchNextEvent() : nullptr);

    if (bDoTime)
    {
        iTimers.pl_h2d.closeTimingRegion(deviceStream);
    }

    /* need to prune the pair list during the next step */
    d_plist->haveFreshList = true;
}

bool gpu_is_kernel_ewald_analytical(const NbnxmGpu* nb)
{
    return ((nb->nbparam->elecType == ElecType::EwaldAna)
            || (nb->nbparam->elecType == ElecType::EwaldAnaTwin));
}

bool haveGpuShortRangeWork(const NbnxmGpu* nb, const gmx::InteractionLocality interactionLocality)
{
    GMX_ASSERT(nb, "Need a valid nbnxn_gpu object");

    return nb->haveWork[interactionLocality];
}

/* Initialization for X buffer operations on GPU. */
void nbnxn_gpu_init_x_to_nbat_x(const Nbnxm::GridSet& gridSet, NbnxmGpu* gpu_nbv)
{
    const DeviceStream& localStream   = *gpu_nbv->deviceStreams[InteractionLocality::Local];
    const bool          bDoTime       = gpu_nbv->bDoTime;
    const int           maxNumColumns = gridSet.numColumnsMax();

    reallocateDeviceBuffer(&gpu_nbv->cxy_na,
                           maxNumColumns * gridSet.grids().size(),
                           &gpu_nbv->ncxy_na,
                           &gpu_nbv->ncxy_na_alloc,
                           *gpu_nbv->deviceContext_);
    reallocateDeviceBuffer(&gpu_nbv->cxy_ind,
                           maxNumColumns * gridSet.grids().size(),
                           &gpu_nbv->ncxy_ind,
                           &gpu_nbv->ncxy_ind_alloc,
                           *gpu_nbv->deviceContext_);

    for (unsigned int g = 0; g < gridSet.grids().size(); g++)
    {
        const Nbnxm::Grid& grid = gridSet.grids()[g];

        const int  numColumns      = grid.numColumns();
        const int* atomIndices     = gridSet.atomIndices().data();
        const int  atomIndicesSize = gridSet.atomIndices().size();
        const int* cxy_na          = grid.cxy_na().data();
        const int* cxy_ind         = grid.cxy_ind().data();

        auto* timerH2D = bDoTime ? &gpu_nbv->timers->xf[AtomLocality::Local].nb_h2d : nullptr;

        reallocateDeviceBuffer(&gpu_nbv->atomIndices,
                               atomIndicesSize,
                               &gpu_nbv->atomIndicesSize,
                               &gpu_nbv->atomIndicesSize_alloc,
                               *gpu_nbv->deviceContext_);

        if (atomIndicesSize > 0)
        {
            if (bDoTime)
            {
                timerH2D->openTimingRegion(localStream);
            }

            copyToDeviceBuffer(&gpu_nbv->atomIndices,
                               atomIndices,
                               0,
                               atomIndicesSize,
                               localStream,
                               GpuApiCallBehavior::Async,
                               bDoTime ? timerH2D->fetchNextEvent() : nullptr);

            if (bDoTime)
            {
                timerH2D->closeTimingRegion(localStream);
            }
        }

        if (numColumns > 0)
        {
            if (bDoTime)
            {
                timerH2D->openTimingRegion(localStream);
            }

            copyToDeviceBuffer(&gpu_nbv->cxy_na,
                               cxy_na,
                               maxNumColumns * g,
                               numColumns,
                               localStream,
                               GpuApiCallBehavior::Async,
                               bDoTime ? timerH2D->fetchNextEvent() : nullptr);

            if (bDoTime)
            {
                timerH2D->closeTimingRegion(localStream);
            }

            if (bDoTime)
            {
                timerH2D->openTimingRegion(localStream);
            }

            copyToDeviceBuffer(&gpu_nbv->cxy_ind,
                               cxy_ind,
                               maxNumColumns * g,
                               numColumns,
                               localStream,
                               GpuApiCallBehavior::Async,
                               bDoTime ? timerH2D->fetchNextEvent() : nullptr);

            if (bDoTime)
            {
                timerH2D->closeTimingRegion(localStream);
            }
        }
    }

    // The above data is transferred on the local stream but is a
    // dependency of the nonlocal stream (specifically the nonlocal X
    // buf ops kernel).  We therefore set a dependency to ensure
    // that the nonlocal stream waits on the local stream here.
    // This call records an event in the local stream:
    nbnxnInsertNonlocalGpuDependency(gpu_nbv, Nbnxm::InteractionLocality::Local);
    // ...and this call instructs the nonlocal stream to wait on that event:
    nbnxnInsertNonlocalGpuDependency(gpu_nbv, Nbnxm::InteractionLocality::NonLocal);
}

//! This function is documented in the header file
void gpu_free(NbnxmGpu* nb)
{
    if (nb == nullptr)
    {
        return;
    }

    gpu_free_platform_specific(nb);

    delete nb->timers;
    sfree(nb->timings);

    NBAtomDataGpu* atdat   = nb->atdat;
    NBParamGpu*    nbparam = nb->nbparam;

    /* Free atdat */
    freeDeviceBuffer(&(nb->atdat->xq));
    freeDeviceBuffer(&(nb->atdat->f));
    freeDeviceBuffer(&(nb->atdat->eLJ));
    freeDeviceBuffer(&(nb->atdat->eElec));
    freeDeviceBuffer(&(nb->atdat->fShift));
    freeDeviceBuffer(&(nb->atdat->shiftVec));
    if (useLjCombRule(nb->nbparam->vdwType))
    {
        freeDeviceBuffer(&atdat->ljComb);
    }
    else
    {
        freeDeviceBuffer(&atdat->atomTypes);
    }

    /* Free nbparam */
    if (nbparam->elecType == ElecType::EwaldTab || nbparam->elecType == ElecType::EwaldTabTwin)
    {
        destroyParamLookupTable(&nbparam->coulomb_tab, nbparam->coulomb_tab_texobj);
    }

    if (!useLjCombRule(nb->nbparam->vdwType))
    {
        destroyParamLookupTable(&nbparam->nbfp, nbparam->nbfp_texobj);
    }

    if (nbparam->vdwType == VdwType::EwaldGeom || nbparam->vdwType == VdwType::EwaldLB)
    {
        destroyParamLookupTable(&nbparam->nbfp_comb, nbparam->nbfp_comb_texobj);
    }

    /* Free plist */
    auto* plist = nb->plist[InteractionLocality::Local];
    freeDeviceBuffer(&plist->sci);
    freeDeviceBuffer(&plist->cj4);
    freeDeviceBuffer(&plist->imask);
    freeDeviceBuffer(&plist->excl);
    delete plist;
    if (nb->bUseTwoStreams)
    {
        auto* plist_nl = nb->plist[InteractionLocality::NonLocal];
        freeDeviceBuffer(&plist_nl->sci);
        freeDeviceBuffer(&plist_nl->cj4);
        freeDeviceBuffer(&plist_nl->imask);
        freeDeviceBuffer(&plist_nl->excl);
        delete plist_nl;
    }

    /* Free nbst */
    pfree(nb->nbst.eLJ);
    nb->nbst.eLJ = nullptr;

    pfree(nb->nbst.eElec);
    nb->nbst.eElec = nullptr;

    pfree(nb->nbst.fShift);
    nb->nbst.fShift = nullptr;

    delete atdat;
    delete nbparam;
    delete nb;

    if (debug)
    {
        fprintf(debug, "Cleaned up NBNXM GPU data structures.\n");
    }
}

} // namespace Nbnxm
