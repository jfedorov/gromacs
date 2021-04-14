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
 *  \brief Declare GPU functions that are internal to NBNXN module.
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_NBNXM_GPU_H
#define GMX_NBNXM_NBNXM_GPU_H

#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gpu_macros.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

#include "nbnxm.h"

struct gmx_wallcycle;
class GpuEventSynchronizer;
enum class GpuTaskCompletion;
struct interaction_const_t;
struct PairlistParams;
struct NbnxmGpu;
struct NbnxnPairlistGpu;
struct nbnxn_atomdata_t;
namespace gmx
{
enum class AtomLocality : int;
class DeviceStreamManager;
enum class InteractionLocality : int;
class StepWorkload;
} // namespace gmx

namespace Nbnxm
{

class Grid;

/** Initializes the data structures related to GPU nonbonded calculations. */
GPU_FUNC_QUALIFIER
NbnxmGpu* gpu_init(const gmx::DeviceStreamManager gmx_unused& deviceStreamManager,
                   const interaction_const_t gmx_unused* ic,
                   const PairlistParams gmx_unused& listParams,
                   const nbnxn_atomdata_t gmx_unused* nbat,
                   /* true if both local and non-local are done on GPU */
                   bool gmx_unused bLocalAndNonlocal) GPU_FUNC_TERM_WITH_RETURN(nullptr);

/** Initializes pair-list data for GPU, called at every pair search step. */
GPU_FUNC_QUALIFIER
void gpu_init_pairlist(NbnxmGpu gmx_unused*          nb,
                       const struct NbnxnPairlistGpu gmx_unused* h_nblist,
                       gmx::InteractionLocality gmx_unused iloc) GPU_FUNC_TERM;

/** Frees all GPU resources used for the nonbonded calculations. */
GPU_FUNC_QUALIFIER
void gpu_free(NbnxmGpu gmx_unused* nb) GPU_FUNC_TERM;

/** Calculates the minimum size of proximity lists to improve SM load balance
 *  with GPU non-bonded kernels. */
GPU_FUNC_QUALIFIER
int gpu_min_ci_balanced(NbnxmGpu gmx_unused* nb) GPU_FUNC_TERM_WITH_RETURN(-1);

/** Returns if analytical Ewald GPU kernels are used. */
GPU_FUNC_QUALIFIER
bool gpu_is_kernel_ewald_analytical(const NbnxmGpu gmx_unused* nb) GPU_FUNC_TERM_WITH_RETURN(FALSE);

/*! \brief Returns true if there is GPU short-range work for the given interaction locality.
 *
 * Note that as, unlike nonbonded tasks, bonded tasks are not split into local/nonlocal,
 * and therefore if there are GPU offloaded bonded interactions, this function will return
 * true for both local and nonlocal atom range.
 *
 * \param[inout]  nb                   Pointer to the nonbonded GPU data structure
 * \param[in]     interactionLocality  Interaction locality identifier
 *
 * \return Whether there is short range work for a given locality.
 */
GPU_FUNC_QUALIFIER
bool haveGpuShortRangeWork(const NbnxmGpu gmx_unused* nb, gmx::InteractionLocality gmx_unused interactionLocality)
        GPU_FUNC_TERM_WITH_RETURN(FALSE);

/*! \brief
 * Launch asynchronously the nonbonded force calculations.
 *
 *  Also launches the initial pruning of a fresh list after search.
 *
 *  The local and non-local interaction calculations are launched in two
 *  separate streams. If there is no work (i.e. empty pair list), the
 *  force kernel launch is omitted.
 *
 */
GPU_FUNC_QUALIFIER
void gpu_launch_kernel(NbnxmGpu gmx_unused*    nb,
                       const gmx::StepWorkload gmx_unused& stepWork,
                       gmx::InteractionLocality gmx_unused iloc) GPU_FUNC_TERM;

/*! \brief
 * Launch asynchronously the nonbonded prune-only kernel.
 *
 *  The local and non-local list pruning are launched in their separate streams.
 *
 *  Notes for future scheduling tuning:
 *  Currently we schedule the dynamic pruning between two MD steps *after* both local and
 *  nonlocal force D2H transfers completed. We could launch already after the cpyback
 *  is launched, but we want to avoid prune kernels (especially in the non-local
 *  high prio-stream) competing with nonbonded work.
 *
 *  However, this is not ideal as this schedule does not expose the available
 *  concurrency. The dynamic pruning kernel:
 *    - should be allowed to overlap with any task other than force compute, including
 *      transfers (F D2H and the next step's x H2D as well as force clearing).
 *    - we'd prefer to avoid competition with non-bonded force kernels belonging
 *      to the same rank and ideally other ranks too.
 *
 *  In the most general case, the former would require scheduling pruning in a separate
 *  stream and adding additional event sync points to ensure that force kernels read
 *  consistent pair list data. This would lead to some overhead (due to extra
 *  cudaStreamWaitEvent calls, 3-5 us/call) which we might be able to live with.
 *  The gains from additional overlap might not be significant as long as
 *  update+constraints anyway takes longer than pruning, but there will still
 *  be use-cases where more overlap may help (e.g. multiple ranks per GPU,
 *  no/hbonds only constraints).
 *  The above second point is harder to address given that multiple ranks will often
 *  share a GPU. Ranks that complete their nonbondeds sooner can schedule pruning earlier
 *  and without a third priority level it is difficult to avoid some interference of
 *  prune kernels with force tasks (in particular preemption of low-prio local force task).
 *
 * \param [inout] nb        GPU nonbonded data.
 * \param [in]    iloc      Interaction locality flag.
 * \param [in]    numParts  Number of parts the pair list is split into in the rolling kernel.
 */
GPU_FUNC_QUALIFIER
void gpu_launch_kernel_pruneonly(NbnxmGpu gmx_unused*     nb,
                                 gmx::InteractionLocality gmx_unused iloc,
                                 int gmx_unused numParts) GPU_FUNC_TERM;

/*! \brief Initialization for X buffer operations on GPU.
 * Called on the NS step and performs (re-)allocations and memory copies. !*/
GPU_FUNC_QUALIFIER
void nbnxn_gpu_init_x_to_nbat_x(const Nbnxm::GridSet gmx_unused& gridSet,
                                NbnxmGpu gmx_unused* gpu_nbv) GPU_FUNC_TERM;

/*! \brief X buffer operations on GPU: performs conversion from rvec to nb format.
 *
 * \param[in]     grid             Grid to be converted.
 * \param[in,out] gpu_nbv          The nonbonded data GPU structure.
 * \param[in]     d_x              Device-side coordinates in plain rvec format.
 * \param[in]     xReadyOnDevice   Event synchronizer indicating that the coordinates are ready in
 * the device memory.
 * \param[in]     locality         Copy coordinates for local or non-local atoms.
 * \param[in]     gridId           Index of the grid being converted.
 * \param[in]     numColumnsMax    Maximum number of columns in the grid.
 * \param[in]     mustInsertNonLocalDependency Whether synchronization between local and non-local
 * streams should be added. Typically, true if and only if that is the last grid in gridset.
 */
CUDA_FUNC_QUALIFIER
void nbnxn_gpu_x_to_nbat_x(const Nbnxm::Grid gmx_unused& grid,
                           NbnxmGpu gmx_unused*    gpu_nbv,
                           DeviceBuffer<gmx::RVec> gmx_unused d_x,
                           GpuEventSynchronizer gmx_unused* xReadyOnDevice,
                           gmx::AtomLocality gmx_unused locality,
                           int gmx_unused gridId,
                           int gmx_unused numColumnsMax,
                           bool gmx_unused mustInsertNonLocalDependency) CUDA_FUNC_TERM;


} // namespace Nbnxm
#endif
