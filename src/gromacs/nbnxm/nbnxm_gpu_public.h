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
 *  \brief Declare external interfaces for NBNXN GPU module.
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_NBNXM_GPU_PUBLIC_H
#define GMX_NBNXM_NBNXM_GPU_PUBLIC_H

#include "gromacs/gpu_utils/gpu_macros.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

#include "nbnxm.h"

struct gmx_wallclock_gpu_nbnxn_t;
struct interaction_const_t;
struct nbnxn_atomdata_t;
struct gmx_wallcycle;
enum class GpuTaskCompletion;

namespace gmx
{
class GpuBonded;
class StepWorkload;
} // namespace gmx

namespace Nbnxm
{

class Grid;

/** Initializes atom-data on the GPU, called at every pair search step. */
GPU_FUNC_QUALIFIER
void gpu_init_atomdata(NbnxmGpu gmx_unused* nb, const nbnxn_atomdata_t gmx_unused* nbat) GPU_FUNC_TERM;

/*! \brief Set up internal flags that indicate what type of short-range work there is.
 *
 * As nonbondeds and bondeds share input/output buffers and GPU queues,
 * both are considered when checking for work in the current domain.
 *
 * This function is expected to be called every time the work-distribution
 * can change (i.e. at search/domain decomposition steps).
 *
 * \param[inout]  nb         Pointer to the nonbonded GPU data structure
 * \param[in]     gpuBonded  Pointer to the GPU bonded data structure
 * \param[in]     iLocality  Interaction locality identifier
 */
GPU_FUNC_QUALIFIER
void setupGpuShortRangeWork(NbnxmGpu gmx_unused* nb,
                            const gmx::GpuBonded gmx_unused* gpuBonded,
                            gmx::InteractionLocality gmx_unused iLocality) GPU_FUNC_TERM;

/*! \brief Re-generate the GPU Ewald force table, resets rlist, and update the
 *  electrostatic type switching to twin cut-off (or back) if needed.
 */
GPU_FUNC_QUALIFIER
void gpu_pme_loadbal_update_param(const struct nonbonded_verlet_t gmx_unused* nbv,
                                  const interaction_const_t gmx_unused& ic) GPU_FUNC_TERM;

/** Uploads shift vector to the GPU if the box is dynamic (otherwise just returns). */
GPU_FUNC_QUALIFIER
void gpu_upload_shiftvec(NbnxmGpu gmx_unused* nb, const nbnxn_atomdata_t gmx_unused* nbatom) GPU_FUNC_TERM;

/** Clears GPU outputs: nonbonded force, shift force and energy. */
GPU_FUNC_QUALIFIER
void gpu_clear_outputs(NbnxmGpu gmx_unused* nb, bool gmx_unused computeVirial) GPU_FUNC_TERM;

/** Returns the GPU timings structure or NULL if GPU is not used or timing is off. */
GPU_FUNC_QUALIFIER
struct gmx_wallclock_gpu_nbnxn_t* gpu_get_timings(NbnxmGpu gmx_unused* nb)
        GPU_FUNC_TERM_WITH_RETURN(nullptr);

/** Resets nonbonded GPU timings. */
GPU_FUNC_QUALIFIER
void gpu_reset_timings(struct nonbonded_verlet_t gmx_unused* nbv) GPU_FUNC_TERM;

/*! \brief
 * Launch asynchronously the xq buffer host to device copy.
 *
 * The nonlocal copy is skipped if there is no dependent work to do,
 * neither non-local nonbonded interactions nor bonded GPU work.
 *
 * \param [in]    nb        GPU nonbonded data.
 * \param [in]    nbdata    Host-side atom data structure.
 * \param [in]    aloc      Atom locality flag.
 */
GPU_FUNC_QUALIFIER
void gpu_copy_xq_to_gpu(NbnxmGpu gmx_unused*          nb,
                        const struct nbnxn_atomdata_t gmx_unused* nbdata,
                        gmx::AtomLocality gmx_unused aloc) GPU_FUNC_TERM;

/*! \brief
 * Launch asynchronously the download of short-range forces from the GPU
 * (and energies/shift forces if required).
 */
GPU_FUNC_QUALIFIER
void gpu_launch_cpyback(NbnxmGpu gmx_unused* nb,
                        nbnxn_atomdata_t gmx_unused* nbatom,
                        const gmx::StepWorkload gmx_unused& stepWork,
                        gmx::AtomLocality gmx_unused aloc) GPU_FUNC_TERM;

/*! \brief Attempts to complete nonbonded GPU task.
 *
 *  This function attempts to complete the nonbonded task (both GPU and CPU auxiliary work).
 *  Success, i.e. that the tasks completed and results are ready to be consumed, is signaled
 *  by the return value (always true if blocking wait mode requested).
 *
 *  The \p completionKind parameter controls whether the behavior is non-blocking
 *  (achieved by passing GpuTaskCompletion::Check) or blocking wait until the results
 *  are ready (when GpuTaskCompletion::Wait is passed).
 *  As the "Check" mode the function will return immediately if the GPU stream
 *  still contain tasks that have not completed, it allows more flexible overlapping
 *  of work on the CPU with GPU execution.
 *
 *  Note that it is only safe to use the results, and to continue to the next MD
 *  step when this function has returned true which indicates successful completion of
 *  - All nonbonded GPU tasks: both compute and device transfer(s)
 *  - auxiliary tasks: updating the internal module state (timing accumulation, list pruning states) and
 *  - internal staging reduction of (\p fshift, \p e_el, \p e_lj).
 *
 * In GpuTaskCompletion::Check mode this function does the timing and keeps correct count
 * for the nonbonded task (incrementing only once per task), in the GpuTaskCompletion::Wait mode
 * timing is expected to be done in the caller.
 *
 *  TODO: improve the handling of outputs e.g. by ensuring that this function explcitly returns the
 *  force buffer (instead of that being passed only to nbnxn_gpu_launch_cpyback()) and by returning
 *  the energy and Fshift contributions for some external/centralized reduction.
 *
 * \param[in]  nb             The nonbonded data GPU structure
 * \param[in]  stepWork       Step schedule flags
 * \param[in]  aloc           Atom locality identifier
 * \param[out] e_lj           Pointer to the LJ energy output to accumulate into
 * \param[out] e_el           Pointer to the electrostatics energy output to accumulate into
 * \param[out] shiftForces    Shift forces buffer to accumulate into
 * \param[in]  completionKind Indicates whether nnbonded task completion should only be checked rather than waited for
 * \param[out] wcycle         Pointer to wallcycle data structure
 * \returns                   True if the nonbonded tasks associated with \p aloc locality have completed
 */
GPU_FUNC_QUALIFIER
bool gpu_try_finish_task(NbnxmGpu gmx_unused*    nb,
                         const gmx::StepWorkload gmx_unused& stepWork,
                         gmx::AtomLocality gmx_unused aloc,
                         real gmx_unused* e_lj,
                         real gmx_unused*         e_el,
                         gmx::ArrayRef<gmx::RVec> gmx_unused shiftForces,
                         GpuTaskCompletion gmx_unused completionKind,
                         gmx_wallcycle gmx_unused* wcycle) GPU_FUNC_TERM_WITH_RETURN(false);

/*! \brief  Completes the nonbonded GPU task blocking until GPU tasks and data
 * transfers to finish.
 *
 * Also does timing accounting and reduction of the internal staging buffers.
 * As this is called at the end of the step, it also resets the pair list and
 * pruning flags.
 *
 * \param[in] nb The nonbonded data GPU structure
 * \param[in]  stepWork        Step schedule flags
 * \param[in] aloc Atom locality identifier
 * \param[out] e_lj Pointer to the LJ energy output to accumulate into
 * \param[out] e_el Pointer to the electrostatics energy output to accumulate into
 * \param[out] shiftForces Shift forces buffer to accumulate into
 * \param[out] wcycle         Pointer to wallcycle data structure               */
GPU_FUNC_QUALIFIER
float gpu_wait_finish_task(NbnxmGpu gmx_unused*    nb,
                           const gmx::StepWorkload gmx_unused& stepWork,
                           gmx::AtomLocality gmx_unused aloc,
                           real gmx_unused* e_lj,
                           real gmx_unused*         e_el,
                           gmx::ArrayRef<gmx::RVec> gmx_unused shiftForces,
                           gmx_wallcycle gmx_unused* wcycle) GPU_FUNC_TERM_WITH_RETURN(0.0);

/** Returns an opaque pointer to the GPU coordinate+charge array
 *  Note: CUDA only.
 */
CUDA_FUNC_QUALIFIER
void* gpu_get_xq(NbnxmGpu gmx_unused* nb) CUDA_FUNC_TERM_WITH_RETURN(nullptr);

/*! \brief Get the pointer to the GPU nonbonded force buffer
 *
 * \param[in] nb  The nonbonded data GPU structure
 * \returns       A pointer to the force buffer in GPU memory
 */
CUDA_FUNC_QUALIFIER
DeviceBuffer<gmx::RVec> getGpuForces(NbnxmGpu gmx_unused* nb)
        CUDA_FUNC_TERM_WITH_RETURN(DeviceBuffer<gmx::RVec>{});

/** Returns an opaque pointer to the GPU shift force array
 *  Note: CUDA only.
 */
CUDA_FUNC_QUALIFIER
DeviceBuffer<gmx::RVec> gpu_get_fshift(NbnxmGpu gmx_unused* nb)
        CUDA_FUNC_TERM_WITH_RETURN(DeviceBuffer<gmx::RVec>{});


} // namespace Nbnxm
#endif
