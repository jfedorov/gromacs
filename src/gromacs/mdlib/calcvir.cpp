/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2018 by the GROMACS development team.
 * Copyright (c) 2019,2020, by the GROMACS development team, led by
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
/* This file is completely threadsafe - keep it that way! */
#include "gmxpre.h"

#include "calcvir.h"

#include "config.h" /* for GMX_MAX_OPENMP_THREADS */

#include <algorithm>

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/gmxassert.h"

#define XXXX 0
#define XXYY 1
#define XXZZ 2
#define YYXX 3
#define YYYY 4
#define YYZZ 5
#define ZZXX 6
#define ZZYY 7
#define ZZZZ 8

static void upd_vir(rvec vir, real dvx, real dvy, real dvz)
{
    vir[XX] -= 0.5 * dvx;
    vir[YY] -= 0.5 * dvy;
    vir[ZZ] -= 0.5 * dvz;
}

static void calc_x_times_f(int nxf, const rvec x[], const rvec f[], gmx_bool bScrewPBC, const matrix box, matrix x_times_f)
{
    clear_mat(x_times_f);

    for (int i = 0; i < nxf; i++)
    {
        for (int d = 0; d < DIM; d++)
        {
            for (int n = 0; n < DIM; n++)
            {
                x_times_f[d][n] += x[i][d] * f[i][n];
            }
        }

        if (bScrewPBC)
        {
            int isx = IS2X(i);
            /* We should correct all odd x-shifts, but the range of isx is -2 to 2 */
            if (isx == 1 || isx == -1)
            {
                for (int d = 0; d < DIM; d++)
                {
                    for (int n = 0; n < DIM; n++)
                    {
                        x_times_f[d][n] += box[d][d] * f[i][n];
                    }
                }
            }
        }
    }
}

// This function can only be called (or tested) by the calculateVirial interface because it is deprecated.
// Once forcerec.shift_vec is refactored away from rvec[] this can be incorporated there.
static void calc_vir(int nxf, const rvec coordinates[], const rvec forces[], tensor vir, bool bScrewPBC, const matrix box)
{
    matrix x_times_f;

    int nthreads = gmx_omp_nthreads_get_simple_rvec_task(emntDefault, nxf * 9);

    GMX_ASSERT(nthreads >= 1, "Avoids uninitialized x_times_f (warning)");

    if (nthreads == 1)
    {
        calc_x_times_f(nxf, coordinates, forces, bScrewPBC, box, x_times_f);
    }
    else
    {
        /* Use a buffer on the stack for storing thread-local results.
         * We use 2 extra elements (=18 reals) per thread to separate thread
         * local data by at least a cache line. Element 0 is not used.
         */
        matrix xf_buf[GMX_OPENMP_MAX_THREADS * 3];

#pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int thread = 0; thread < nthreads; thread++)
        {
            int start = (nxf * thread) / nthreads;
            int end   = std::min(nxf * (thread + 1) / nthreads, nxf);

            calc_x_times_f(end - start,
                           coordinates + start,
                           forces + start,
                           bScrewPBC,
                           box,
                           thread == 0 ? x_times_f : xf_buf[thread * 3]);
        }

        for (int thread = 1; thread < nthreads; thread++)
        {
            m_add(x_times_f, xf_buf[thread * 3], x_times_f);
        }
    }

    for (int d = 0; d < DIM; d++)
    {
        upd_vir(vir[d], x_times_f[d][XX], x_times_f[d][YY], x_times_f[d][ZZ]);
    }
}

namespace gmx
{

void calculateViralParts(int                         numVirialAtoms,
                         ArrayRef<const RVec>        coordinates,
                         const ForceWithShiftForces& forceWithShiftForces,
                         tensor                      virialParts,
                         const matrix                box,
                         t_nrnb*                     nrnb,
                         const rvec                  shiftVectors[],
                         PbcType                     pbcType)
{
    /* The short-range virial from surrounding boxes */
    const rvec* fshift = as_rvec_array(forceWithShiftForces.shiftForces().data());
    calc_vir(SHIFTS, shiftVectors, fshift, virialParts, pbcType == PbcType::Screw, box);
    inc_nrnb(nrnb, eNR_VIRIAL, SHIFTS);

    /* Calculate partial virial, for local atoms only, based on short range.
     * Total virial is computed in global_stat, called from do_md
     */
    calculateVirial(numVirialAtoms, coordinates, forceWithShiftForces.force(), virialParts, false, box);
    inc_nrnb(nrnb, eNR_VIRIAL, numVirialAtoms);
}

void calculateVirial(int                  numVirialAtoms,
                     ArrayRef<const RVec> coordinates,
                     ArrayRef<const RVec> forces,
                     tensor               virial,
                     bool                 bScrewPBC,
                     const matrix         box)
{
    calc_vir(numVirialAtoms, as_rvec_array(coordinates.data()), as_rvec_array(forces.data()), virial, bScrewPBC, box);
}


} // namespace gmx
