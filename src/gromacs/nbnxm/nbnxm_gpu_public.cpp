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
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/nbnxm_gpu_internal.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"

#include "nbnxm_gpu_public.h"

namespace Nbnxm
{

void gpu_init_atomdata(NbnxmGpu* nb, const nbnxn_atomdata_t* nbat)
{
    bool                 bDoTime       = nb->bDoTime;
    Nbnxm::GpuTimers*    timers        = bDoTime ? nb->timers : nullptr;
    NBAtomDataGpu*       atdat         = nb->atdat;
    const DeviceContext& deviceContext = *nb->deviceContext_;
    const DeviceStream&  localStream   = *nb->deviceStreams[InteractionLocality::Local];

    int  numAtoms  = nbat->numAtoms();
    bool realloced = false;

    if (bDoTime)
    {
        /* time async copy */
        timers->atdat.openTimingRegion(localStream);
    }

    /* need to reallocate if we have to copy more atoms than the amount of space
       available and only allocate if we haven't initialized yet, i.e atdat->natoms == -1 */
    if (numAtoms > atdat->numAtomsAlloc)
    {
        int numAlloc = over_alloc_small(numAtoms);

        /* free up first if the arrays have already been initialized */
        if (atdat->numAtomsAlloc != -1)
        {
            freeDeviceBuffer(&atdat->f);
            freeDeviceBuffer(&atdat->xq);
            if (useLjCombRule(nb->nbparam->vdwType))
            {
                freeDeviceBuffer(&atdat->ljComb);
            }
            else
            {
                freeDeviceBuffer(&atdat->atomTypes);
            }
        }


        allocateDeviceBuffer(&atdat->f, numAlloc, deviceContext);
        allocateDeviceBuffer(&atdat->xq, numAlloc, deviceContext);

        if (useLjCombRule(nb->nbparam->vdwType))
        {
            // Two Lennard-Jones parameters per atom
            allocateDeviceBuffer(&atdat->ljComb, numAlloc, deviceContext);
        }
        else
        {
            allocateDeviceBuffer(&atdat->atomTypes, numAlloc, deviceContext);
        }

        atdat->numAtomsAlloc = numAlloc;
        realloced            = true;
    }

    atdat->numAtoms      = numAtoms;
    atdat->numAtomsLocal = nbat->natoms_local;

    /* need to clear GPU f output if realloc happened */
    if (realloced)
    {
        clearDeviceBufferAsync(&atdat->f, 0, atdat->numAtomsAlloc, localStream);
    }

    if (useLjCombRule(nb->nbparam->vdwType))
    {
        static_assert(
                sizeof(Float2) == 2 * sizeof(*nbat->params().lj_comb.data()),
                "Size of a pair of LJ parameters elements should be equal to the size of Float2.");
        copyToDeviceBuffer(&atdat->ljComb,
                           reinterpret_cast<const Float2*>(nbat->params().lj_comb.data()),
                           0,
                           numAtoms,
                           localStream,
                           GpuApiCallBehavior::Async,
                           bDoTime ? timers->atdat.fetchNextEvent() : nullptr);
    }
    else
    {
        static_assert(sizeof(int) == sizeof(*nbat->params().type.data()),
                      "Sizes of host- and device-side atom types should be the same.");
        copyToDeviceBuffer(&atdat->atomTypes,
                           nbat->params().type.data(),
                           0,
                           numAtoms,
                           localStream,
                           GpuApiCallBehavior::Async,
                           bDoTime ? timers->atdat.fetchNextEvent() : nullptr);
    }

    if (bDoTime)
    {
        timers->atdat.closeTimingRegion(localStream);
    }

    /* kick off the tasks enqueued above to ensure concurrency with the search */
    issueClFlushInStream(localStream);
}

void setupGpuShortRangeWork(NbnxmGpu* nb, const gmx::GpuBonded* gpuBonded, const gmx::InteractionLocality iLocality)
{
    GMX_ASSERT(nb, "Need a valid nbnxn_gpu object");

    // There is short-range work if the pair list for the provided
    // interaction locality contains entries or if there is any
    // bonded work (as this is not split into local/nonlocal).
    nb->haveWork[iLocality] = ((nb->plist[iLocality]->nsci != 0)
                               || (gpuBonded != nullptr && gpuBonded->haveInteractions()));
}

void gpu_pme_loadbal_update_param(const nonbonded_verlet_t* nbv, const interaction_const_t& ic)
{
    if (!nbv || !nbv->useGpu())
    {
        return;
    }
    NbnxmGpu*   nb  = nbv->gpu_nbv;
    NBParamGpu* nbp = nb->nbparam;

    set_cutoff_parameters(nbp, ic, nbv->pairlistSets().params());

    nbp->elecType = nbnxn_gpu_pick_ewald_kernel_type(ic, nb->deviceContext_->deviceInfo());

    GMX_RELEASE_ASSERT(ic.coulombEwaldTables, "Need valid Coulomb Ewald correction tables");
    init_ewald_coulomb_force_table(*ic.coulombEwaldTables, nbp, *nb->deviceContext_);
}

void gpu_upload_shiftvec(NbnxmGpu* nb, const nbnxn_atomdata_t* nbatom)
{
    NBAtomDataGpu*      adat        = nb->atdat;
    const DeviceStream& localStream = *nb->deviceStreams[InteractionLocality::Local];

    /* only if we have a dynamic box */
    if (nbatom->bDynamicBox || !adat->shiftVecUploaded)
    {
        copyToDeviceBuffer(&adat->shiftVec,
                           gmx::asGenericFloat3Pointer(nbatom->shift_vec),
                           0,
                           gmx::c_numShiftVectors,
                           localStream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        adat->shiftVecUploaded = true;
    }
}

void gpu_clear_outputs(NbnxmGpu* nb, bool computeVirial)
{
    NBAtomDataGpu*      adat        = nb->atdat;
    const DeviceStream& localStream = *nb->deviceStreams[InteractionLocality::Local];
    // Clear forces
    clearDeviceBufferAsync(&adat->f, 0, nb->atdat->numAtoms, localStream);
    // Clear shift force array and energies if the outputs were used in the current step
    if (computeVirial)
    {
        clearDeviceBufferAsync(&adat->fShift, 0, gmx::c_numShiftVectors, localStream);
        clearDeviceBufferAsync(&adat->eLJ, 0, 1, localStream);
        clearDeviceBufferAsync(&adat->eElec, 0, 1, localStream);
    }
    issueClFlushInStream(localStream);
}

//! This function is documented in the header file
gmx_wallclock_gpu_nbnxn_t* gpu_get_timings(NbnxmGpu* nb)
{
    return (nb != nullptr && nb->bDoTime) ? nb->timings : nullptr;
}

//! This function is documented in the header file
void gpu_reset_timings(nonbonded_verlet_t* nbv)
{
    if (nbv->gpu_nbv && nbv->gpu_nbv->bDoTime)
    {
        init_timings(nbv->gpu_nbv->timings);
    }
}

/*! \brief Launch asynchronously the xq buffer host to device copy. */
void gpu_copy_xq_to_gpu(NbnxmGpu* nb, const nbnxn_atomdata_t* nbatom, const AtomLocality atomLocality)
{
    GMX_ASSERT(nb, "Need a valid nbnxn_gpu object");

    const InteractionLocality iloc = atomToInteractionLocality(atomLocality);

    NBAtomDataGpu*      adat         = nb->atdat;
    gpu_plist*          plist        = nb->plist[iloc];
    Nbnxm::GpuTimers*   timers       = nb->timers;
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    const bool bDoTime = nb->bDoTime;

    /* Don't launch the non-local H2D copy if there is no dependent
       work to do: neither non-local nor other (e.g. bonded) work
       to do that has as input the nbnxn coordaintes.
       Doing the same for the local kernel is more complicated, since the
       local part of the force array also depends on the non-local kernel.
       So to avoid complicating the code and to reduce the risk of bugs,
       we always call the local local x+q copy (and the rest of the local
       work in nbnxn_gpu_launch_kernel().
     */
    if ((iloc == InteractionLocality::NonLocal) && !haveGpuShortRangeWork(nb, iloc))
    {
        plist->haveFreshList = false;

        // The event is marked for Local interactions unconditionally,
        // so it has to be released here because of the early return
        // for NonLocal interactions.
        nb->misc_ops_and_local_H2D_done.reset();

        return;
    }

    /* local/nonlocal offset and length used for xq and f */
    const auto atomsRange = getGpuAtomRange(adat, atomLocality);

    /* beginning of timed HtoD section */
    if (bDoTime)
    {
        timers->xf[atomLocality].nb_h2d.openTimingRegion(deviceStream);
    }

    /* HtoD x, q */
    GMX_ASSERT(nbatom->XFormat == nbatXYZQ,
               "The coordinates should be in xyzq format to copy to the Float4 device buffer.");
    copyToDeviceBuffer(&adat->xq,
                       reinterpret_cast<const Float4*>(nbatom->x().data()) + atomsRange.begin(),
                       atomsRange.begin(),
                       atomsRange.size(),
                       deviceStream,
                       GpuApiCallBehavior::Async,
                       nullptr);

    if (bDoTime)
    {
        timers->xf[atomLocality].nb_h2d.closeTimingRegion(deviceStream);
    }

    /* When we get here all misc operations issued in the local stream as well as
       the local xq H2D are done,
       so we record that in the local stream and wait for it in the nonlocal one.
       This wait needs to precede any PP tasks, bonded or nonbonded, that may
       compute on interactions between local and nonlocal atoms.
     */
    nbnxnInsertNonlocalGpuDependency(nb, iloc);
}

/*! \brief
 * Launch asynchronously the download of nonbonded forces from the GPU
 * (and energies/shift forces if required).
 */
void gpu_launch_cpyback(NbnxmGpu*                nb,
                        struct nbnxn_atomdata_t* nbatom,
                        const gmx::StepWorkload& stepWork,
                        const AtomLocality       atomLocality)
{
    GMX_ASSERT(nb, "Need a valid nbnxn_gpu object");

    /* determine interaction locality from atom locality */
    const InteractionLocality iloc = atomToInteractionLocality(atomLocality);
    GMX_ASSERT(iloc == InteractionLocality::Local
                       || (iloc == InteractionLocality::NonLocal && nb->bNonLocalStreamDoneMarked == false),
               "Non-local stream is indicating that the copy back event is enqueued at the "
               "beginning of the copy back function.");

    /* extract the data */
    NBAtomDataGpu*      adat         = nb->atdat;
    Nbnxm::GpuTimers*   timers       = nb->timers;
    bool                bDoTime      = nb->bDoTime;
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    /* don't launch non-local copy-back if there was no non-local work to do */
    if ((iloc == InteractionLocality::NonLocal) && !haveGpuShortRangeWork(nb, iloc))
    {
        /* TODO An alternative way to signal that non-local work is
           complete is to use a clEnqueueMarker+clEnqueueBarrier
           pair. However, the use of bNonLocalStreamDoneMarked has the
           advantage of being local to the host, so probably minimizes
           overhead. Curiously, for NVIDIA OpenCL with an empty-domain
           test case, overall simulation performance was higher with
           the API calls, but this has not been tested on AMD OpenCL,
           so could be worth considering in future. */
        nb->bNonLocalStreamDoneMarked = false;
        return;
    }

    /* local/nonlocal offset and length used for xq and f */
    auto atomsRange = getGpuAtomRange(adat, atomLocality);

    /* beginning of timed D2H section */
    if (bDoTime)
    {
        timers->xf[atomLocality].nb_d2h.openTimingRegion(deviceStream);
    }

    /* With DD the local D2H transfer can only start after the non-local
       has been launched. */
    if (iloc == InteractionLocality::Local && nb->bNonLocalStreamDoneMarked)
    {
        nb->nonlocal_done.enqueueWaitEvent(deviceStream);
        nb->bNonLocalStreamDoneMarked = false;
    }

    /* DtoH f */
    if (!stepWork.useGpuFBufferOps)
    {
        static_assert(
                sizeof(*nbatom->out[0].f.data()) == sizeof(float),
                "The host force buffer should be in single precision to match device data size.");
        copyFromDeviceBuffer(reinterpret_cast<Float3*>(nbatom->out[0].f.data()) + atomsRange.begin(),
                             &adat->f,
                             atomsRange.begin(),
                             atomsRange.size(),
                             deviceStream,
                             GpuApiCallBehavior::Async,
                             bDoTime ? timers->xf[atomLocality].nb_d2h.fetchNextEvent() : nullptr);

        issueClFlushInStream(deviceStream);
    }

    /* After the non-local D2H is launched the nonlocal_done event can be
       recorded which signals that the local D2H can proceed. This event is not
       placed after the non-local kernel because we first need the non-local
       data back first. */
    if (iloc == InteractionLocality::NonLocal)
    {
        nb->nonlocal_done.markEvent(deviceStream);
        nb->bNonLocalStreamDoneMarked = true;
    }

    /* only transfer energies in the local stream */
    if (iloc == InteractionLocality::Local)
    {
        /* DtoH fshift when virial is needed */
        if (stepWork.computeVirial)
        {
            static_assert(
                    sizeof(*nb->nbst.fShift) == sizeof(Float3),
                    "Sizes of host- and device-side shift vector elements should be the same.");
            copyFromDeviceBuffer(nb->nbst.fShift,
                                 &adat->fShift,
                                 0,
                                 gmx::c_numShiftVectors,
                                 deviceStream,
                                 GpuApiCallBehavior::Async,
                                 bDoTime ? timers->xf[atomLocality].nb_d2h.fetchNextEvent() : nullptr);
        }

        /* DtoH energies */
        if (stepWork.computeEnergy)
        {
            static_assert(sizeof(*nb->nbst.eLJ) == sizeof(float),
                          "Sizes of host- and device-side LJ energy terms should be the same.");
            copyFromDeviceBuffer(nb->nbst.eLJ,
                                 &adat->eLJ,
                                 0,
                                 1,
                                 deviceStream,
                                 GpuApiCallBehavior::Async,
                                 bDoTime ? timers->xf[atomLocality].nb_d2h.fetchNextEvent() : nullptr);
            static_assert(sizeof(*nb->nbst.eElec) == sizeof(float),
                          "Sizes of host- and device-side electrostatic energy terms should be the "
                          "same.");
            copyFromDeviceBuffer(nb->nbst.eElec,
                                 &adat->eElec,
                                 0,
                                 1,
                                 deviceStream,
                                 GpuApiCallBehavior::Async,
                                 bDoTime ? timers->xf[atomLocality].nb_d2h.fetchNextEvent() : nullptr);
        }
    }

    if (bDoTime)
    {
        timers->xf[atomLocality].nb_d2h.closeTimingRegion(deviceStream);
    }
}

/*! \brief Count pruning kernel time if either kernel has been triggered
 *
 *  We do the accounting for either of the two pruning kernel flavors:
 *   - 1st pass prune: ran during the current step (prior to the force kernel);
 *   - rolling prune:  ran at the end of the previous step (prior to the current step H2D xq);
 *
 * Note that the resetting of Nbnxm::GpuTimers::didPrune and Nbnxm::GpuTimers::didRollingPrune
 * should happen after calling this function.
 *
 * \param[in] timers   structs with GPU timer objects
 * \param[inout] timings  GPU task timing data
 * \param[in] iloc        interaction locality
 */
static inline void countPruneKernelTime(Nbnxm::GpuTimers*          timers,
                                        gmx_wallclock_gpu_nbnxn_t* timings,
                                        const InteractionLocality  iloc)
{
    GpuTimers::Interaction& iTimers = timers->interaction[iloc];

    // We might have not done any pruning (e.g. if we skipped with empty domains).
    if (!iTimers.didPrune && !iTimers.didRollingPrune)
    {
        return;
    }

    if (iTimers.didPrune)
    {
        timings->pruneTime.c++;
        timings->pruneTime.t += iTimers.prune_k.getLastRangeTime();
    }

    if (iTimers.didRollingPrune)
    {
        timings->dynamicPruneTime.c++;
        timings->dynamicPruneTime.t += iTimers.rollingPrune_k.getLastRangeTime();
    }
}

/*! \brief Do the per-step timing accounting of the nonbonded tasks.
 *
 *  Does timing accumulation and call-count increments for the nonbonded kernels.
 *  Note that this function should be called after the current step's nonbonded
 *  nonbonded tasks have completed with the exception of the rolling pruning kernels
 *  that are accounted for during the following step.
 *
 * NOTE: if timing with multiple GPUs (streams) becomes possible, the
 *      counters could end up being inconsistent due to not being incremented
 *      on some of the node when this is skipped on empty local domains!
 *
 * \tparam     GpuPairlist       Pair list type
 * \param[out] timings           Pointer to the NB GPU timings data
 * \param[in]  timers            Pointer to GPU timers data
 * \param[in]  plist             Pointer to the pair list data
 * \param[in]  atomLocality      Atom locality specifier
 * \param[in]  stepWork          Force schedule flags
 * \param[in]  doTiming          True if timing is enabled.
 *
 */
template<typename GpuPairlist>
static inline void gpu_accumulate_timings(gmx_wallclock_gpu_nbnxn_t* timings,
                                          Nbnxm::GpuTimers*          timers,
                                          const GpuPairlist*         plist,
                                          AtomLocality               atomLocality,
                                          const gmx::StepWorkload&   stepWork,
                                          bool                       doTiming)
{
    /* timing data accumulation */
    if (!doTiming)
    {
        return;
    }

    /* determine interaction locality from atom locality */
    const InteractionLocality iLocality        = atomToInteractionLocality(atomLocality);
    const bool                didEnergyKernels = stepWork.computeEnergy;

    /* only increase counter once (at local F wait) */
    if (iLocality == InteractionLocality::Local)
    {
        timings->nb_c++;
        timings->ktime[plist->haveFreshList ? 1 : 0][didEnergyKernels ? 1 : 0].c += 1;
    }

    /* kernel timings */
    timings->ktime[plist->haveFreshList ? 1 : 0][didEnergyKernels ? 1 : 0].t +=
            timers->interaction[iLocality].nb_k.getLastRangeTime();

    /* X/q H2D and F D2H timings */
    timings->nb_h2d_t += timers->xf[atomLocality].nb_h2d.getLastRangeTime();
    timings->nb_d2h_t += timers->xf[atomLocality].nb_d2h.getLastRangeTime();

    /* Count the pruning kernel times for both cases:1st pass (at search step)
       and rolling pruning (if called at the previous step).
       We do the accounting here as this is the only sync point where we
       know (without checking or additional sync-ing) that prune tasks in
       in the current stream have completed (having just blocking-waited
       for the force D2H). */
    countPruneKernelTime(timers, timings, iLocality);

    /* only count atdat at pair-search steps (add only once, at local F wait) */
    if (stepWork.doNeighborSearch && atomLocality == AtomLocality::Local)
    {
        /* atdat transfer timing */
        timings->pl_h2d_c++;
        timings->pl_h2d_t += timers->atdat.getLastRangeTime();
    }

    /* only count pair-list H2D when actually performed */
    if (timers->interaction[iLocality].didPairlistH2D)
    {
        timings->pl_h2d_t += timers->interaction[iLocality].pl_h2d.getLastRangeTime();

        /* Clear the timing flag for the next step */
        timers->interaction[iLocality].didPairlistH2D = false;
    }
}

/*! \brief Reduce data staged internally in the nbnxn module.
 *
 * Shift forces and electrostatic/LJ energies copied from the GPU into
 * a module-internal staging area are immediately reduced (CPU-side buffers passed)
 * after having waited for the transfers' completion.
 *
 * Note that this function should always be called after the transfers into the
 * staging buffers has completed.
 *
 * \param[in]  nbst           Nonbonded staging data
 * \param[in]  iLocality      Interaction locality specifier
 * \param[in]  reduceEnergies True if energy reduction should be done
 * \param[in]  reduceFshift   True if shift force reduction should be done
 * \param[out] e_lj           Variable to accumulate LJ energy into
 * \param[out] e_el           Variable to accumulate electrostatic energy into
 * \param[out] fshift         Pointer to the array of shift forces to accumulate into
 */
static inline void gpu_reduce_staged_outputs(const NBStagingData&      nbst,
                                             const InteractionLocality iLocality,
                                             const bool                reduceEnergies,
                                             const bool                reduceFshift,
                                             real*                     e_lj,
                                             real*                     e_el,
                                             rvec*                     fshift)
{
    /* add up energies and shift forces (only once at local F wait) */
    if (iLocality == InteractionLocality::Local)
    {
        if (reduceEnergies)
        {
            *e_lj += *nbst.eLJ;
            *e_el += *nbst.eElec;
        }

        if (reduceFshift)
        {
            for (int i = 0; i < gmx::c_numShiftVectors; i++)
            {
                rvec_inc(fshift[i], nbst.fShift[i]);
            }
        }
    }
}

/*! \brief Attempts to complete nonbonded GPU task.
 *
 * See documentation in nbnxm_gpu.h for details.
 *
 * \todo Move into shared source file, perhaps including
 * cuda_runtime.h if needed for any remaining CUDA-specific
 * objects.
 */
//NOLINTNEXTLINE(misc-definitions-in-headers)
bool gpu_try_finish_task(NbnxmGpu*                nb,
                         const gmx::StepWorkload& stepWork,
                         const AtomLocality       aloc,
                         real*                    e_lj,
                         real*                    e_el,
                         gmx::ArrayRef<gmx::RVec> shiftForces,
                         GpuTaskCompletion        completionKind,
                         gmx_wallcycle*           wcycle)
{
    GMX_ASSERT(nb, "Need a valid nbnxn_gpu object");

    /* determine interaction locality from atom locality */
    const InteractionLocality iLocality = atomToInteractionLocality(aloc);


    // Transfers are launched and therefore need to be waited on if:
    // - buffer ops is not offloaded
    // - energies or virials are needed (on the local stream)
    //
    // (Note that useGpuFBufferOps and computeVirial are mutually exclusive
    // in current code as virial steps do CPU reduction.)
    const bool haveResultToWaitFor =
            (!stepWork.useGpuFBufferOps
             || (aloc == AtomLocality::Local && (stepWork.computeEnergy || stepWork.computeVirial)));

    //  We skip when during the non-local phase there was actually no work to do.
    //  This is consistent with nbnxn_gpu_launch_kernel but it also considers possible
    //  bonded GPU work.
    if ((iLocality == InteractionLocality::Local) || haveGpuShortRangeWork(nb, iLocality))
    {
        // Query the state of the GPU stream and return early if we're not done
        if (completionKind == GpuTaskCompletion::Check)
        {
            // To get the wcycle call count right, when in GpuTaskCompletion::Check mode,
            // we start without counting and only when the task finished we issue a
            // start/stop to increment.
            // GpuTaskCompletion::Wait mode the timing is expected to be done in the caller.
            wallcycle_start_nocount(wcycle, WallCycleCounter::WaitGpuNbL);

            if (!haveStreamTasksCompleted(*nb->deviceStreams[iLocality]))
            {
                wallcycle_stop(wcycle, WallCycleCounter::WaitGpuNbL);

                // Early return to skip the steps below that we have to do only
                // after the NB task completed
                return false;
            }

            wallcycle_increment_event_count(wcycle, WallCycleCounter::WaitGpuNbL);
        }
        else if (haveResultToWaitFor)
        {
            nb->deviceStreams[iLocality]->synchronize();
        }

        // TODO: this needs to be moved later because conditional wait could brake timing
        // with a future OpenCL implementation, but with CUDA timing is anyway disabled
        // in all cases where we skip the wait.
        gpu_accumulate_timings(nb->timings, nb->timers, nb->plist[iLocality], aloc, stepWork, nb->bDoTime);

        if (stepWork.computeEnergy || stepWork.computeVirial)
        {
            gpu_reduce_staged_outputs(nb->nbst,
                                      iLocality,
                                      stepWork.computeEnergy,
                                      stepWork.computeVirial,
                                      e_lj,
                                      e_el,
                                      as_rvec_array(shiftForces.data()));
        }
    }

    /* Reset both pruning flags. */
    if (nb->bDoTime)
    {
        nb->timers->interaction[iLocality].didPrune =
                nb->timers->interaction[iLocality].didRollingPrune = false;
    }

    /* Turn off initial list pruning (doesn't hurt if this is not pair-search step). */
    nb->plist[iLocality]->haveFreshList = false;

    return true;
}

/*! \brief
 * Wait for the asynchronously launched nonbonded tasks and data
 * transfers to finish.
 *
 * Also does timing accounting and reduction of the internal staging buffers.
 * As this is called at the end of the step, it also resets the pair list and
 * pruning flags.
 *
 * \param[in] nb The nonbonded data GPU structure
 * \param[in]  stepWork     Force schedule flags
 * \param[in] aloc Atom locality identifier
 * \param[out] e_lj Pointer to the LJ energy output to accumulate into
 * \param[out] e_el Pointer to the electrostatics energy output to accumulate into
 * \param[out] shiftForces Shift forces buffer to accumulate into
 * \param[out] wcycle Pointer to wallcycle data structure
 * \return            The number of cycles the gpu wait took
 */
//NOLINTNEXTLINE(misc-definitions-in-headers) TODO: move into source file
float gpu_wait_finish_task(NbnxmGpu*                nb,
                           const gmx::StepWorkload& stepWork,
                           AtomLocality             aloc,
                           real*                    e_lj,
                           real*                    e_el,
                           gmx::ArrayRef<gmx::RVec> shiftForces,
                           gmx_wallcycle*           wcycle)
{
    auto cycleCounter = (atomToInteractionLocality(aloc) == InteractionLocality::Local)
                                ? WallCycleCounter::WaitGpuNbL
                                : WallCycleCounter::WaitGpuNbNL;

    wallcycle_start(wcycle, cycleCounter);
    gpu_try_finish_task(nb, stepWork, aloc, e_lj, e_el, shiftForces, GpuTaskCompletion::Wait, wcycle);
    float waitTime = wallcycle_stop(wcycle, cycleCounter);

    return waitTime;
}

} // namespace Nbnxm
