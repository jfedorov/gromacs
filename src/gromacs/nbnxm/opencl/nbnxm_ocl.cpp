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
 *  \brief Define OpenCL implementation of nbnxm_gpu.h
 *
 *  \author Anca Hamuraru <anca@streamcomputing.eu>
 *  \author Teemu Virolainen <teemu@streamcomputing.eu>
 *  \author Dimitrios Karkoulis <dimitris.karkoulis@gmail.com>
 *  \author Szilárd Páll <pall.szilard@gmail.com>
 *  \ingroup module_nbnxm
 *
 *  TODO (psz):
 *  - Add a static const cl_uint c_pruneKernelWorkDim / c_nbnxnKernelWorkDim = 3;
 *  - Rework the copying of OCL data structures done before every invocation of both
 *    nb and prune kernels (using fillin_ocl_structures); also consider at the same
 *    time calling clSetKernelArg only on the updated parameters (if tracking changed
 *    parameters is feasible);
 *  - Consider using the event_wait_list argument to clEnqueueNDRangeKernel to mark
 *    dependencies on the kernel launched: e.g. the non-local nb kernel's dependency
 *    on the misc_ops_and_local_H2D_done event could be better expressed this way.
 *
 *  - Consider extracting common sections of the OpenCL and CUDA nbnxn logic, e.g:
 *    - in nbnxn_gpu_launch_kernel_pruneonly() the pre- and post-kernel launch logic
 *      is identical in the two implementations, so a 3-way split might allow sharing
 *      code;
 *    -
 *
 */
#include "gmxpre.h"

#include <assert.h>
#include <stdlib.h>

#if defined(_MSVC)
#    include <limits>
#endif

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/gputraits_ocl.h"
#include "gromacs/gpu_utils/oclutils.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_common.h"
#include "gromacs/nbnxm/gpu_common_utils.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/nbnxm_gpu_internal.h"
#include "gromacs/nbnxm/opencl/nbnxm_ocl.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/gmxassert.h"

#include "nbnxm_ocl_types.h"

namespace Nbnxm
{

/*! \brief Convenience constants */
//@{
static constexpr int c_clSize = c_nbnxnGpuClusterSize;
//@}

/* Constant arrays listing non-bonded kernel function names. The arrays are
 * organized in 2-dim arrays by: electrostatics and VDW type.
 *
 *  Note that the row- and column-order of function pointers has to match the
 *  order of corresponding enumerated electrostatics and vdw types, resp.,
 *  defined in nbnxm_ocl_types.h.
 */

/*! \brief Force-only kernel function names. */
static const char* nb_kfunc_noener_noprune_ptr[c_numElecTypes][c_numVdwTypes] = {
    { "nbnxn_kernel_ElecCut_VdwLJ_F_opencl",
      "nbnxn_kernel_ElecCut_VdwLJCombGeom_F_opencl",
      "nbnxn_kernel_ElecCut_VdwLJCombLB_F_opencl",
      "nbnxn_kernel_ElecCut_VdwLJFsw_F_opencl",
      "nbnxn_kernel_ElecCut_VdwLJPsw_F_opencl",
      "nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_opencl",
      "nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_opencl" },
    { "nbnxn_kernel_ElecRF_VdwLJ_F_opencl",
      "nbnxn_kernel_ElecRF_VdwLJCombGeom_F_opencl",
      "nbnxn_kernel_ElecRF_VdwLJCombLB_F_opencl",
      "nbnxn_kernel_ElecRF_VdwLJFsw_F_opencl",
      "nbnxn_kernel_ElecRF_VdwLJPsw_F_opencl",
      "nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_opencl",
      "nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_opencl" },
    { "nbnxn_kernel_ElecEwQSTab_VdwLJ_F_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_F_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_F_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJFsw_F_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_opencl" },
    { "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_F_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_F_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_opencl" },
    { "nbnxn_kernel_ElecEw_VdwLJ_F_opencl",
      "nbnxn_kernel_ElecEw_VdwLJCombGeom_F_opencl",
      "nbnxn_kernel_ElecEw_VdwLJCombLB_F_opencl",
      "nbnxn_kernel_ElecEw_VdwLJFsw_F_opencl",
      "nbnxn_kernel_ElecEw_VdwLJPsw_F_opencl",
      "nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_opencl",
      "nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_opencl" },
    { "nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_F_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_F_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_opencl" }
};

/*! \brief Force + energy kernel function pointers. */
static const char* nb_kfunc_ener_noprune_ptr[c_numElecTypes][c_numVdwTypes] = {
    { "nbnxn_kernel_ElecCut_VdwLJ_VF_opencl",
      "nbnxn_kernel_ElecCut_VdwLJCombGeom_VF_opencl",
      "nbnxn_kernel_ElecCut_VdwLJCombLB_VF_opencl",
      "nbnxn_kernel_ElecCut_VdwLJFsw_VF_opencl",
      "nbnxn_kernel_ElecCut_VdwLJPsw_VF_opencl",
      "nbnxn_kernel_ElecCut_VdwLJEwCombGeom_VF_opencl",
      "nbnxn_kernel_ElecCut_VdwLJEwCombLB_VF_opencl" },
    { "nbnxn_kernel_ElecRF_VdwLJ_VF_opencl",
      "nbnxn_kernel_ElecRF_VdwLJCombGeom_VF_opencl",
      "nbnxn_kernel_ElecRF_VdwLJCombLB_VF_opencl",
      "nbnxn_kernel_ElecRF_VdwLJFsw_VF_opencl",
      "nbnxn_kernel_ElecRF_VdwLJPsw_VF_opencl",
      "nbnxn_kernel_ElecRF_VdwLJEwCombGeom_VF_opencl",
      "nbnxn_kernel_ElecRF_VdwLJEwCombLB_VF_opencl" },
    { "nbnxn_kernel_ElecEwQSTab_VdwLJ_VF_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_VF_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_VF_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJFsw_VF_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJPsw_VF_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_VF_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_VF_opencl" },
    { "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_VF_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_VF_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_VF_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_VF_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_VF_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_VF_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_VF_opencl" },
    { "nbnxn_kernel_ElecEw_VdwLJ_VF_opencl",
      "nbnxn_kernel_ElecEw_VdwLJCombGeom_VF_opencl",
      "nbnxn_kernel_ElecEw_VdwLJCombLB_VF_opencl",
      "nbnxn_kernel_ElecEw_VdwLJFsw_VF_opencl",
      "nbnxn_kernel_ElecEw_VdwLJPsw_VF_opencl",
      "nbnxn_kernel_ElecEw_VdwLJEwCombGeom_VF_opencl",
      "nbnxn_kernel_ElecEw_VdwLJEwCombLB_VF_opencl" },
    { "nbnxn_kernel_ElecEwTwinCut_VdwLJ_VF_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_VF_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_VF_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_VF_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_VF_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_VF_opencl" }
};

/*! \brief Force + pruning kernel function pointers. */
static const char* nb_kfunc_noener_prune_ptr[c_numElecTypes][c_numVdwTypes] = {
    { "nbnxn_kernel_ElecCut_VdwLJ_F_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJCombLB_F_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJFsw_F_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJPsw_F_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_prune_opencl" },
    { "nbnxn_kernel_ElecRF_VdwLJ_F_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJCombLB_F_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJFsw_F_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJPsw_F_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_prune_opencl" },
    { "nbnxn_kernel_ElecEwQSTab_VdwLJ_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJFsw_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_prune_opencl" },
    { "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_prune_opencl" },
    { "nbnxn_kernel_ElecEw_VdwLJ_F_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJCombLB_F_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJFsw_F_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJPsw_F_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_prune_opencl" },
    { "nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_F_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_prune_opencl" }
};

/*! \brief Force + energy + pruning kernel function pointers. */
static const char* nb_kfunc_ener_prune_ptr[c_numElecTypes][c_numVdwTypes] = {
    { "nbnxn_kernel_ElecCut_VdwLJ_VF_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJCombLB_VF_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJFsw_VF_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJPsw_VF_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJEwCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecCut_VdwLJEwCombLB_VF_prune_opencl" },
    { "nbnxn_kernel_ElecRF_VdwLJ_VF_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJCombLB_VF_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJFsw_VF_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJPsw_VF_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJEwCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecRF_VdwLJEwCombLB_VF_prune_opencl" },
    { "nbnxn_kernel_ElecEwQSTab_VdwLJ_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJFsw_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJPsw_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_VF_prune_opencl" },
    { "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_VF_prune_opencl" },
    { "nbnxn_kernel_ElecEw_VdwLJ_VF_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJCombLB_VF_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJFsw_VF_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJPsw_VF_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJEwCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecEw_VdwLJEwCombLB_VF_prune_opencl" },
    { "nbnxn_kernel_ElecEwTwinCut_VdwLJ_VF_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_VF_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_VF_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_VF_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF_prune_opencl",
      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_VF_prune_opencl" }
};

/*! \brief Return a pointer to the kernel version to be executed at the current step.
 *  OpenCL kernel objects are cached in nb. If the requested kernel is not
 *  found in the cache, it will be created and the cache will be updated.
 */
static inline cl_kernel
select_nbnxn_kernel(NbnxmGpu* nb, enum ElecType elecType, enum VdwType vdwType, bool bDoEne, bool bDoPrune)
{
    const char* kernel_name_to_run;
    cl_kernel*  kernel_ptr;
    cl_int      cl_error;

    const int elecTypeIdx = static_cast<int>(elecType);
    const int vdwTypeIdx  = static_cast<int>(vdwType);

    GMX_ASSERT(elecTypeIdx < c_numElecTypes,
               "The electrostatics type requested is not implemented in the OpenCL kernels.");
    GMX_ASSERT(vdwTypeIdx < c_numVdwTypes,
               "The VdW type requested is not implemented in the OpenCL kernels.");

    if (bDoEne)
    {
        if (bDoPrune)
        {
            kernel_name_to_run = nb_kfunc_ener_prune_ptr[elecTypeIdx][vdwTypeIdx];
            kernel_ptr         = &(nb->kernel_ener_prune_ptr[elecTypeIdx][vdwTypeIdx]);
        }
        else
        {
            kernel_name_to_run = nb_kfunc_ener_noprune_ptr[elecTypeIdx][vdwTypeIdx];
            kernel_ptr         = &(nb->kernel_ener_noprune_ptr[elecTypeIdx][vdwTypeIdx]);
        }
    }
    else
    {
        if (bDoPrune)
        {
            kernel_name_to_run = nb_kfunc_noener_prune_ptr[elecTypeIdx][vdwTypeIdx];
            kernel_ptr         = &(nb->kernel_noener_prune_ptr[elecTypeIdx][vdwTypeIdx]);
        }
        else
        {
            kernel_name_to_run = nb_kfunc_noener_noprune_ptr[elecTypeIdx][vdwTypeIdx];
            kernel_ptr         = &(nb->kernel_noener_noprune_ptr[elecTypeIdx][vdwTypeIdx]);
        }
    }

    if (nullptr == kernel_ptr[0])
    {
        *kernel_ptr = clCreateKernel(nb->dev_rundata->program, kernel_name_to_run, &cl_error);
        GMX_ASSERT(cl_error == CL_SUCCESS,
                   ("clCreateKernel failed: " + ocl_get_error_string(cl_error)
                    + " for kernel named " + kernel_name_to_run)
                           .c_str());
    }

    return *kernel_ptr;
}

/*! \brief Calculates the amount of shared memory required by the nonbonded kernel in use.
 */
static inline int calc_shmem_required_nonbonded(enum VdwType vdwType, bool bPrefetchLjParam)
{
    int shmem;

    /* size of shmem (force-buffers/xq/atom type preloading) */
    /* NOTE: with the default kernel on sm3.0 we need shmem only for pre-loading */
    /* i-atom x+q in shared memory */
    shmem = c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(float) * 4; /* xqib */
    /* cj in shared memory, for both warps separately
     * TODO: in the "nowarp kernels we load cj only once  so the factor 2 is not needed.
     */
    shmem += 2 * c_nbnxnGpuJgroupSize * sizeof(int); /* cjs  */
    if (bPrefetchLjParam)
    {
        if (useLjCombRule(vdwType))
        {
            /* i-atom LJ combination parameters in shared memory */
            shmem += c_nbnxnGpuNumClusterPerSupercluster * c_clSize * 2
                     * sizeof(float); /* atib abused for ljcp, float2 */
        }
        else
        {
            /* i-atom types in shared memory */
            shmem += c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(int); /* atib */
        }
    }
    /* force reduction buffers in shared memory */
    shmem += c_clSize * c_clSize * 3 * sizeof(float); /* f_buf */
    /* Warp vote. In fact it must be * number of warps in block.. */
    shmem += sizeof(cl_uint) * 2; /* warp_any */
    return shmem;
}

/*! \brief Launch GPU kernel

   As we execute nonbonded workload in separate queues, before launching
   the kernel we need to make sure that he following operations have completed:
   - atomdata allocation and related H2D transfers (every nstlist step);
   - pair list H2D transfer (every nstlist step);
   - shift vector H2D transfer (every nstlist step);
   - force (+shift force and energy) output clearing (every step).

   These operations are issued in the local queue at the beginning of the step
   and therefore always complete before the local kernel launch. The non-local
   kernel is launched after the local on the same device/context, so this is
   inherently scheduled after the operations in the local stream (including the
   above "misc_ops").
   However, for the sake of having a future-proof implementation, we use the
   misc_ops_done event to record the point in time when the above  operations
   are finished and synchronize with this event in the non-local stream.
 */
void gpu_launch_kernel(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const Nbnxm::InteractionLocality iloc)
{
    NBAtomDataGpu*      adat         = nb->atdat;
    NBParamGpu*         nbp          = nb->nbparam;
    gpu_plist*          plist        = nb->plist[iloc];
    Nbnxm::GpuTimers*   timers       = nb->timers;
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    bool bDoTime = nb->bDoTime;

    /* Don't launch the non-local kernel if there is no work to do.
       Doing the same for the local kernel is more complicated, since the
       local part of the force array also depends on the non-local kernel.
       So to avoid complicating the code and to reduce the risk of bugs,
       we always call the local kernel and later (not in
       this function) the stream wait, local f copyback and the f buffer
       clearing. All these operations, except for the local interaction kernel,
       are needed for the non-local interactions. The skip of the local kernel
       call is taken care of later in this function. */
    if (canSkipNonbondedWork(*nb, iloc))
    {
        plist->haveFreshList = false;

        return;
    }

    if (nbp->useDynamicPruning && plist->haveFreshList)
    {
        /* Prunes for rlistOuter and rlistInner, sets plist->haveFreshList=false
           (that's the way the timing accounting can distinguish between
           separate prune kernel and combined force+prune).
         */
        Nbnxm::gpu_launch_kernel_pruneonly(nb, iloc, 1);
    }

    if (plist->nsci == 0)
    {
        /* Don't launch an empty local kernel (is not allowed with OpenCL).
         */
        return;
    }

    /* beginning of timed nonbonded calculation section */
    if (bDoTime)
    {
        timers->interaction[iloc].nb_k.openTimingRegion(deviceStream);
    }

    /* kernel launch config */

    KernelLaunchConfig config;
    config.sharedMemorySize = calc_shmem_required_nonbonded(nbp->vdwType, nb->bPrefetchLjParam);
    config.blockSize[0]     = c_clSize;
    config.blockSize[1]     = c_clSize;
    config.gridSize[0]      = plist->nsci;

    validate_global_work_size(config, 3, &nb->deviceContext_->deviceInfo());

    if (debug)
    {
        fprintf(debug,
                "Non-bonded GPU launch configuration:\n\tLocal work size: %zux%zux%zu\n\t"
                "Global work size : %zux%zu\n\t#Super-clusters/clusters: %d/%d (%d)\n",
                config.blockSize[0],
                config.blockSize[1],
                config.blockSize[2],
                config.blockSize[0] * config.gridSize[0],
                config.blockSize[1] * config.gridSize[1],
                plist->nsci * c_nbnxnGpuNumClusterPerSupercluster,
                c_nbnxnGpuNumClusterPerSupercluster,
                plist->na_c);
    }

    cl_nbparam_params_t nbparams_params;
    fillin_ocl_structures(nbp, &nbparams_params);

    auto* timingEvent = bDoTime ? timers->interaction[iloc].nb_k.fetchNextEvent() : nullptr;
    constexpr char kernelName[] = "k_calc_nb";
    const auto     kernel =
            select_nbnxn_kernel(nb,
                                nbp->elecType,
                                nbp->vdwType,
                                stepWork.computeEnergy,
                                (plist->haveFreshList && !nb->timers->interaction[iloc].didPrune));


    // The OpenCL kernel takes int as second to last argument because bool is
    // not supported as a kernel argument type (sizeof(bool) is implementation defined).
    const int computeFshift = static_cast<int>(stepWork.computeVirial);
    if (useLjCombRule(nb->nbparam->vdwType))
    {
        const auto kernelArgs = prepareGpuKernelArguments(kernel,
                                                          config,
                                                          &nbparams_params,
                                                          &adat->xq,
                                                          &adat->f,
                                                          &adat->eLJ,
                                                          &adat->eElec,
                                                          &adat->fShift,
                                                          &adat->ljComb,
                                                          &adat->shiftVec,
                                                          &nbp->nbfp,
                                                          &nbp->nbfp_comb,
                                                          &nbp->coulomb_tab,
                                                          &plist->sci,
                                                          &plist->cj4,
                                                          &plist->excl,
                                                          &computeFshift);

        launchGpuKernel(kernel, config, deviceStream, timingEvent, kernelName, kernelArgs);
    }
    else
    {
        const auto kernelArgs = prepareGpuKernelArguments(kernel,
                                                          config,
                                                          &adat->numTypes,
                                                          &nbparams_params,
                                                          &adat->xq,
                                                          &adat->f,
                                                          &adat->eLJ,
                                                          &adat->eElec,
                                                          &adat->fShift,
                                                          &adat->atomTypes,
                                                          &adat->shiftVec,
                                                          &nbp->nbfp,
                                                          &nbp->nbfp_comb,
                                                          &nbp->coulomb_tab,
                                                          &plist->sci,
                                                          &plist->cj4,
                                                          &plist->excl,
                                                          &computeFshift);
        launchGpuKernel(kernel, config, deviceStream, timingEvent, kernelName, kernelArgs);
    }

    if (bDoTime)
    {
        timers->interaction[iloc].nb_k.closeTimingRegion(deviceStream);
    }
}

} // namespace Nbnxm
