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

#ifndef GMX_NBNXM_NBNXM_GPU_PRIVATE_H
#define GMX_NBNXM_NBNXM_GPU_PRIVATE_H

#include "gromacs/mdtypes/locality.h"
#include "gromacs/nbnxm/gpu_types_common.h"
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

#endif // GMX_NBNXM_NBNXM_GPU_PRIVATE_H
