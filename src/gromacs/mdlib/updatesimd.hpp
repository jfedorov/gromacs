/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021, by the GROMACS development team, led by
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
#ifndef GMX_MDLIB_UPDATE_HELPERS_H
#define GMX_MDLIB_UPDATE_HELPERS_H

#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"

namespace gmx
{

constexpr bool gmx_have_simd_update = GMX_SIMD && GMX_SIMD_HAVE_REAL;

static int threadBlockSize()
{
    if constexpr (gmx_have_simd_update)
    {
        return GMX_SIMD_REAL_WIDTH;
    }
    return 1;
}

/*! \brief Load (aligned) the contents of GMX_SIMD_REAL_WIDTH rvec elements sequentially into 3 SIMD registers
 *
 * The loaded output is:
 * \p r0: { r[index][XX], r[index][YY], ... }
 * \p r1: { ... }
 * \p r2: { ..., r[index+GMX_SIMD_REAL_WIDTH-1][YY], r[index+GMX_SIMD_REAL_WIDTH-1][ZZ] }
 *
 * \param[in]  r      Real to an rvec array, has to be aligned to SIMD register width
 * \param[in]  index  Index of the first rvec triplet of reals to load
 * \param[out] r0     Pointer to first SIMD register
 * \param[out] r1     Pointer to second SIMD register
 * \param[out] r2     Pointer to third SIMD register
 */
static inline void simdLoadRvecs(const rvec* r, int index, SimdReal* r0, SimdReal* r1, SimdReal* r2)
{
    const real* realPtr = r[index];

    GMX_ASSERT(isSimdAligned(realPtr), "Pointer should be SIMD aligned");

    *r0 = simdLoad(realPtr + 0 * GMX_SIMD_REAL_WIDTH);
    *r1 = simdLoad(realPtr + 1 * GMX_SIMD_REAL_WIDTH);
    *r2 = simdLoad(realPtr + 2 * GMX_SIMD_REAL_WIDTH);
}

/*! \brief Store (aligned) 3 SIMD registers sequentially to GMX_SIMD_REAL_WIDTH rvec elements
 *
 * The stored output is:
 * \p r[index] = { { r0[0], r0[1], ... }
 * ...
 * \p r[index+GMX_SIMD_REAL_WIDTH-1] =  { ... , r2[GMX_SIMD_REAL_WIDTH-2], r2[GMX_SIMD_REAL_WIDTH-1] }
 *
 * \param[out] r      Pointer to an rvec array, has to be aligned to SIMD register width
 * \param[in]  index  Index of the first rvec triplet of reals to store to
 * \param[in]  r0     First SIMD register
 * \param[in]  r1     Second SIMD register
 * \param[in]  r2     Third SIMD register
 */
static inline void simdStoreRvecs(rvec* r, int index, SimdReal r0, SimdReal r1, SimdReal r2)
{
    real* realPtr = r[index];

    GMX_ASSERT(isSimdAligned(realPtr), "Pointer should be SIMD aligned");

    store(realPtr + 0 * GMX_SIMD_REAL_WIDTH, r0);
    store(realPtr + 1 * GMX_SIMD_REAL_WIDTH, r1);
    store(realPtr + 2 * GMX_SIMD_REAL_WIDTH, r2);
}

/*! \brief Integrate using leap-frog with single group T-scaling and SIMD
 *
 * \tparam       storeUpdatedVelocities Tells whether we should store the updated velocities
 * \param[in]    start                  Index of first atom to update
 * \param[in]    nrend                  Last atom to update: \p nrend - 1
 * \param[in]    dt                     The time step
 * \param[in]    invMass                1/mass per atom
 * \param[in]    tcstat                 Temperature coupling information
 * \param[in]    x                      Input coordinates
 * \param[out]   xprime                 Updated coordinates
 * \param[inout] v                      Velocities, type either rvec* or const rvec*
 * \param[in]    f                      Forces
 */
template<StoreUpdatedVelocities storeUpdatedVelocities, typename VelocityType>
static std::enable_if_t<std::is_same<VelocityType, rvec*>::value || std::is_same<VelocityType, const rvec*>::value, void>
updateMDLeapfrogSimpleSimd(int                               start,
                           int                               nrend,
                           real                              dt,
                           gmx::ArrayRef<const real>         invMass,
                           gmx::ArrayRef<const t_grp_tcstat> tcstat,
                           const rvec* gmx_restrict x,
                           rvec* gmx_restrict xprime,
                           VelocityType gmx_restrict v,
                           const rvec* gmx_restrict f)
{
    SimdReal timestep(dt);
    SimdReal lambdaSystem(tcstat[0].lambda);

    /* We declare variables here, since code is often slower when declaring them inside the loop */

    /* Note: We should implement a proper PaddedVector, so we don't need this check */
    GMX_ASSERT(isSimdAligned(invMass.data()), "invMass should be aligned");

    for (int a = start; a < nrend; a += GMX_SIMD_REAL_WIDTH)
    {
        SimdReal invMass0, invMass1, invMass2;
        expandScalarsToTriplets(simdLoad(invMass.data() + a), &invMass0, &invMass1, &invMass2);

        SimdReal v0, v1, v2;
        SimdReal f0, f1, f2;
        simdLoadRvecs(v, a, &v0, &v1, &v2);
        simdLoadRvecs(f, a, &f0, &f1, &f2);

        v0 = fma(f0 * invMass0, timestep, lambdaSystem * v0);
        v1 = fma(f1 * invMass1, timestep, lambdaSystem * v1);
        v2 = fma(f2 * invMass2, timestep, lambdaSystem * v2);

        // TODO: Remove NOLINTs once clang-tidy is updated to v11, it should be able to handle constexpr.
        if constexpr (storeUpdatedVelocities == StoreUpdatedVelocities::yes) // NOLINT // NOLINTNEXTLINE
        {
            simdStoreRvecs(v, a, v0, v1, v2);
        }

        SimdReal x0, x1, x2; // NOLINT(readability-misleading-indentation)
        simdLoadRvecs(x, a, &x0, &x1, &x2);

        SimdReal xprime0 = fma(v0, timestep, x0);
        SimdReal xprime1 = fma(v1, timestep, x1);
        SimdReal xprime2 = fma(v2, timestep, x2);

        simdStoreRvecs(xprime, a, xprime0, xprime1, xprime2);
    }
}

} // namespace gmx

#endif
