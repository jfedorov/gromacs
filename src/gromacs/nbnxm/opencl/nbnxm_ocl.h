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
 *  \brief
 *  Data types used internally in the nbnxm_ocl module.
 *
 *  \author Anca Hamuraru <anca@streamcomputing.eu>
 *  \author Szilárd Páll <pszilard@kth.se>
 *  \ingroup module_nbnxm
 */
#ifndef GMX_NBNXM_NBNXM_OPENCL_H
#define GMX_NBNXM_NBNXM_OPENCL_H

#include "gromacs/gpu_utils/gputraits_ocl.h"
#include "gromacs/gpu_utils/gmxopencl.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/nbnxm/opencl/nbnxm_ocl_types.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

/*! \brief Initializes data structures that are going to be sent to the OpenCL device.
 *
 *  The device can't use the same data structures as the host for two main reasons:
 *  - OpenCL restrictions (pointers are not accepted inside data structures)
 *  - some host side fields are not needed for the OpenCL kernels.
 *
 *  This function is called before the launch of both nbnxn and prune kernels.
 */
static void fillin_ocl_structures(NBParamGpu* nbp, cl_nbparam_params_t* nbparams_params)
{
    nbparams_params->coulomb_tab_scale = nbp->coulomb_tab_scale;
    nbparams_params->c_rf              = nbp->c_rf;
    nbparams_params->dispersion_shift  = nbp->dispersion_shift;
    nbparams_params->elecType          = nbp->elecType;
    nbparams_params->epsfac            = nbp->epsfac;
    nbparams_params->ewaldcoeff_lj     = nbp->ewaldcoeff_lj;
    nbparams_params->ewald_beta        = nbp->ewald_beta;
    nbparams_params->rcoulomb_sq       = nbp->rcoulomb_sq;
    nbparams_params->repulsion_shift   = nbp->repulsion_shift;
    nbparams_params->rlistOuter_sq     = nbp->rlistOuter_sq;
    nbparams_params->rvdw_sq           = nbp->rvdw_sq;
    nbparams_params->rlistInner_sq     = nbp->rlistInner_sq;
    nbparams_params->rvdw_switch       = nbp->rvdw_switch;
    nbparams_params->sh_ewald          = nbp->sh_ewald;
    nbparams_params->sh_lj_ewald       = nbp->sh_lj_ewald;
    nbparams_params->two_k_rf          = nbp->two_k_rf;
    nbparams_params->vdwType           = nbp->vdwType;
    nbparams_params->vdw_switch        = nbp->vdw_switch;
}

/*! \brief Validates the input global work size parameter.
 */
static inline void validate_global_work_size(const KernelLaunchConfig& config,
                                             int                       work_dim,
                                             const DeviceInformation*  dinfo)
{
    cl_uint device_size_t_size_bits;
    cl_uint host_size_t_size_bits;

    GMX_ASSERT(dinfo, "Need a valid device info object");

    size_t global_work_size[3];
    GMX_ASSERT(work_dim <= 3, "Not supporting hyper-grids just yet");
    for (int i = 0; i < work_dim; i++)
    {
        global_work_size[i] = config.blockSize[i] * config.gridSize[i];
    }

    /* Each component of a global_work_size must not exceed the range given by the
       sizeof(device size_t) for the device on which the kernel execution will
       be enqueued. See:
       https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clEnqueueNDRangeKernel.html
     */
    device_size_t_size_bits = dinfo->adress_bits;
    host_size_t_size_bits   = static_cast<cl_uint>(sizeof(size_t) * 8);

    /* If sizeof(host size_t) <= sizeof(device size_t)
            => global_work_size components will always be valid
       else
            => get device limit for global work size and
            compare it against each component of global_work_size.
     */
    if (host_size_t_size_bits > device_size_t_size_bits)
    {
        size_t device_limit;

        device_limit = (1ULL << device_size_t_size_bits) - 1;

        for (int i = 0; i < work_dim; i++)
        {
            if (global_work_size[i] > device_limit)
            {
                gmx_fatal(
                        FARGS,
                        "Watch out, the input system is too large to simulate!\n"
                        "The number of nonbonded work units (=number of super-clusters) exceeds the"
                        "device capabilities. Global work size limit exceeded (%zu > %zu)!",
                        global_work_size[i],
                        device_limit);
            }
        }
    }
}

#endif /* NBNXN_OPENCL_H */
