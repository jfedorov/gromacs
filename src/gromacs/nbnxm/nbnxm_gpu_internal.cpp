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

#include "nbnxm_gpu_internal.h"

#include "gromacs/mdtypes/locality.h"
#include "gromacs/nbnxm/nbnxm_gpu_public.h"

#if GMX_GPU_CUDA
#    include "gromacs/nbnxm/cuda/nbnxm_cuda_types.h"
#elif GMX_GPU_OPENCL
#    include "gromacs/nbnxm/opencl/nbnxm_ocl_types.h"
#elif GMX_GPU_SYCL
#    include "gromacs/nbnxm/sycl/nbnxm_sycl_types.h"
#endif

class DeviceStream;

namespace Nbnxm
{

void nbnxnInsertNonlocalGpuDependency(NbnxmGpu* nb, const InteractionLocality interactionLocality)
{
    const DeviceStream& deviceStream = *nb->deviceStreams[interactionLocality];

    /* When we get here all misc operations issued in the local stream as well as
       the local xq H2D are done,
       so we record that in the local stream and wait for it in the nonlocal one.
       This wait needs to precede any PP tasks, bonded or nonbonded, that may
       compute on interactions between local and nonlocal atoms.
     */
    if (nb->bUseTwoStreams)
    {
        if (interactionLocality == InteractionLocality::Local)
        {
            nb->misc_ops_and_local_H2D_done.markEvent(deviceStream);
            issueClFlushInStream(deviceStream);
        }
        else
        {
            nb->misc_ops_and_local_H2D_done.enqueueWaitEvent(deviceStream);
        }
    }
}

} // namespace Nbnxm
