/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
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
 *
 * \brief SYCL-specific routines for the GPU implementation of SETTLE constraints algorithm.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_mdlib
 */

#include "settle_gpu_internal.h"

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/sycl_kernel_utils.h"
#include "gromacs/pbcutil/pbc_aiuc_sycl.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/template_mp.h"

namespace gmx
{

using cl::sycl::access::fence_space;
using cl::sycl::access::mode;
using cl::sycl::access::target;

//! Number of work-items in a work-group
constexpr static int sc_workGroupSize = 256;

template<bool updateVelocities, bool computeVirial>
auto settleKernel(cl::sycl::handler&                                           cgh,
                  const int                                                    numSettles,
                  DeviceAccessor<WaterMolecule, mode::read>                    a_settles,
                  SettleParameters                                             pars,
                  DeviceAccessor<Float3, mode::read>                           a_x,
                  DeviceAccessor<Float3, mode::read_write>                     a_xp,
                  float                                                        invdt,
                  OptionalAccessor<Float3, mode::read_write, updateVelocities> a_v,
                  OptionalAccessor<float, mode_atomic, computeVirial>          a_virialScaled,
                  PbcAiuc                                                      pbcAiuc)
{
    cgh.require(a_settles);
    cgh.require(a_x);
    cgh.require(a_xp);
    if constexpr (updateVelocities)
    {
        cgh.require(a_v);
    }
    if constexpr (computeVirial)
    {
        cgh.require(a_virialScaled);
    }

    // shmem buffer for i x+q pre-loading
    auto sm_threadVirial = [&]() {
        if constexpr (computeVirial)
        {
            return cl::sycl::accessor<float, 1, mode::read_write, target::local>(
                    cl::sycl::range<1>(sc_workGroupSize * 6), cgh);
        }
        else
        {
            return nullptr;
        }
    }();

    return [=](cl::sycl::nd_item<1> itemIdx) {
        constexpr float almost_zero = real(1e-12);
        const int       settleIdx   = itemIdx.get_global_linear_id();
        const int       threadIdx = itemIdx.get_local_linear_id(); // Work-item index in work-group
        assert(itemIdx.get_local_range(0) == sc_workGroupSize);
        // These are the indexes of three atoms in a single 'water' molecule.
        // TODO Can be reduced to one integer if atoms are consecutive in memory.
        if (settleIdx < numSettles)
        {
            WaterMolecule indices = a_settles[settleIdx];

            const Float3 x_ow1 = a_x[indices.ow1];
            const Float3 x_hw2 = a_x[indices.hw2];
            const Float3 x_hw3 = a_x[indices.hw3];

            const Float3 xprime_ow1 = a_xp[indices.ow1];
            const Float3 xprime_hw2 = a_xp[indices.hw2];
            const Float3 xprime_hw3 = a_xp[indices.hw3];

            Float3 dist21;
            pbcDxAiucSycl(pbcAiuc, x_hw2, x_ow1, dist21);
            Float3 dist31;
            pbcDxAiucSycl(pbcAiuc, x_hw3, x_ow1, dist31);
            Float3 doh2;
            pbcDxAiucSycl(pbcAiuc, xprime_hw2, xprime_ow1, doh2);

            Float3 doh3;
            pbcDxAiucSycl(pbcAiuc, xprime_hw3, xprime_ow1, doh3);

            Float3 a1 = (doh2 + doh3) * (-pars.wh);

            Float3 b1 = doh2 + a1;

            Float3 c1 = doh3 + a1;

            float xakszd = dist21[YY] * dist31[ZZ] - dist21[ZZ] * dist31[YY];
            float yakszd = dist21[ZZ] * dist31[XX] - dist21[XX] * dist31[ZZ];
            float zakszd = dist21[XX] * dist31[YY] - dist21[YY] * dist31[XX];

            float xaksxd = a1[YY] * zakszd - a1[ZZ] * yakszd;
            float yaksxd = a1[ZZ] * xakszd - a1[XX] * zakszd;
            float zaksxd = a1[XX] * yakszd - a1[YY] * xakszd;

            float xaksyd = yakszd * zaksxd - zakszd * yaksxd;
            float yaksyd = zakszd * xaksxd - xakszd * zaksxd;
            float zaksyd = xakszd * yaksxd - yakszd * xaksxd;

            float axlng = cl::sycl::rsqrt(xaksxd * xaksxd + yaksxd * yaksxd + zaksxd * zaksxd);
            float aylng = cl::sycl::rsqrt(xaksyd * xaksyd + yaksyd * yaksyd + zaksyd * zaksyd);
            float azlng = cl::sycl::rsqrt(xakszd * xakszd + yakszd * yakszd + zakszd * zakszd);

            // TODO {1,2,3} indexes should be swapped with {.x, .y, .z} components.
            //      This way, we will be able to use vector ops more.
            Float3 trns1, trns2, trns3;

            trns1[XX] = xaksxd * axlng;
            trns2[XX] = yaksxd * axlng;
            trns3[XX] = zaksxd * axlng;

            trns1[YY] = xaksyd * aylng;
            trns2[YY] = yaksyd * aylng;
            trns3[YY] = zaksyd * aylng;

            trns1[ZZ] = xakszd * azlng;
            trns2[ZZ] = yakszd * azlng;
            trns3[ZZ] = zakszd * azlng;


            Float2 b0d, c0d;

            b0d[XX] = trns1[XX] * dist21[XX] + trns2[XX] * dist21[YY] + trns3[XX] * dist21[ZZ];
            b0d[YY] = trns1[YY] * dist21[XX] + trns2[YY] * dist21[YY] + trns3[YY] * dist21[ZZ];

            c0d[XX] = trns1[XX] * dist31[XX] + trns2[XX] * dist31[YY] + trns3[XX] * dist31[ZZ];
            c0d[YY] = trns1[YY] * dist31[XX] + trns2[YY] * dist31[YY] + trns3[YY] * dist31[ZZ];

            Float3 b1d, c1d;

            float a1d_z = trns1[ZZ] * a1[XX] + trns2[ZZ] * a1[YY] + trns3[ZZ] * a1[ZZ];

            b1d[XX] = trns1[XX] * b1[XX] + trns2[XX] * b1[YY] + trns3[XX] * b1[ZZ];
            b1d[YY] = trns1[YY] * b1[XX] + trns2[YY] * b1[YY] + trns3[YY] * b1[ZZ];
            b1d[ZZ] = trns1[ZZ] * b1[XX] + trns2[ZZ] * b1[YY] + trns3[ZZ] * b1[ZZ];

            c1d[XX] = trns1[XX] * c1[XX] + trns2[XX] * c1[YY] + trns3[XX] * c1[ZZ];
            c1d[YY] = trns1[YY] * c1[XX] + trns2[YY] * c1[YY] + trns3[YY] * c1[ZZ];
            c1d[ZZ] = trns1[ZZ] * c1[XX] + trns2[ZZ] * c1[YY] + trns3[ZZ] * c1[ZZ];


            const float sinphi  = a1d_z / pars.ra;
            float       cosphi2 = 1.0F - sinphi * sinphi;

            if (almost_zero > cosphi2)
            {
                cosphi2 = almost_zero;
            }

            const float rcosphi = cl::sycl::rsqrt(cosphi2);
            const float cosphi  = cosphi2 * rcosphi;
            const float sinpsi  = (b1d[ZZ] - c1d[ZZ]) * pars.irc2 * rcosphi;
            const float cospsi2 = 1.0F - sinpsi * sinpsi;

            const float cospsi = cospsi2 * cl::sycl::rsqrt(cospsi2);

            const float a2d_y = pars.ra * cosphi;
            const float b2d_x = -pars.rc * cospsi;
            const float t1    = -pars.rb * cosphi;
            const float t2    = pars.rc * sinpsi * sinphi;
            const float b2d_y = t1 - t2;
            const float c2d_y = t1 + t2;

            /*     --- Step3  al,be,ga            --- */
            const float alpha = b2d_x * (b0d[XX] - c0d[XX]) + b0d[YY] * b2d_y + c0d[YY] * c2d_y;
            const float beta  = b2d_x * (c0d[YY] - b0d[YY]) + b0d[XX] * b2d_y + c0d[XX] * c2d_y;
            const float gamma =
                    b0d[XX] * b1d[YY] - b1d[XX] * b0d[YY] + c0d[XX] * c1d[YY] - c1d[XX] * c0d[YY];
            const float al2be2 = alpha * alpha + beta * beta;
            const float tmp    = (al2be2 - gamma * gamma);
            const float sinthe = (alpha * gamma - beta * tmp * cl::sycl::rsqrt(tmp))
                                 * cl::sycl::rsqrt(al2be2 * al2be2);

            /*  --- Step4  A3' --- */
            const float costhe2 = 1.0F - sinthe * sinthe;
            const float costhe  = costhe2 * cl::sycl::rsqrt(costhe2);

            Float3 a3d, b3d, c3d;

            a3d[XX] = -a2d_y * sinthe;
            a3d[YY] = a2d_y * costhe;
            a3d[ZZ] = a1d_z;
            b3d[XX] = b2d_x * costhe - b2d_y * sinthe;
            b3d[YY] = b2d_x * sinthe + b2d_y * costhe;
            b3d[ZZ] = b1d[ZZ];
            c3d[XX] = -b2d_x * costhe - c2d_y * sinthe;
            c3d[YY] = -b2d_x * sinthe + c2d_y * costhe;
            c3d[ZZ] = c1d[ZZ];

            /*    --- Step5  A3 --- */
            Float3 a3, b3, c3;

            a3[XX] = trns1[XX] * a3d[XX] + trns1[YY] * a3d[YY] + trns1[ZZ] * a3d[ZZ];
            a3[YY] = trns2[XX] * a3d[XX] + trns2[YY] * a3d[YY] + trns2[ZZ] * a3d[ZZ];
            a3[ZZ] = trns3[XX] * a3d[XX] + trns3[YY] * a3d[YY] + trns3[ZZ] * a3d[ZZ];

            b3[XX] = trns1[XX] * b3d[XX] + trns1[YY] * b3d[YY] + trns1[ZZ] * b3d[ZZ];
            b3[YY] = trns2[XX] * b3d[XX] + trns2[YY] * b3d[YY] + trns2[ZZ] * b3d[ZZ];
            b3[ZZ] = trns3[XX] * b3d[XX] + trns3[YY] * b3d[YY] + trns3[ZZ] * b3d[ZZ];

            c3[XX] = trns1[XX] * c3d[XX] + trns1[YY] * c3d[YY] + trns1[ZZ] * c3d[ZZ];
            c3[YY] = trns2[XX] * c3d[XX] + trns2[YY] * c3d[YY] + trns2[ZZ] * c3d[ZZ];
            c3[ZZ] = trns3[XX] * c3d[XX] + trns3[YY] * c3d[YY] + trns3[ZZ] * c3d[ZZ];


            /* Compute and store the corrected new coordinate */
            const Float3 dxOw1 = a3 - a1;
            const Float3 dxHw2 = b3 - b1;
            const Float3 dxHw3 = c3 - c1;

            a_xp[indices.ow1] = xprime_ow1 + dxOw1;
            a_xp[indices.hw2] = xprime_hw2 + dxHw2;
            a_xp[indices.hw3] = xprime_hw3 + dxHw3;

            if constexpr (updateVelocities)
            {
                Float3 v_ow1 = a_v[indices.ow1];
                Float3 v_hw2 = a_v[indices.hw2];
                Float3 v_hw3 = a_v[indices.hw3];

                /* Add the position correction divided by dt to the velocity */
                v_ow1 = dxOw1 * invdt + v_ow1;
                v_hw2 = dxHw2 * invdt + v_hw2;
                v_hw3 = dxHw3 * invdt + v_hw3;

                a_v[indices.ow1] = v_ow1;
                a_v[indices.hw2] = v_hw2;
                a_v[indices.hw3] = v_hw3;
            }

            if constexpr (computeVirial)
            {
                Float3 mdb = pars.mH * dxHw2;
                Float3 mdc = pars.mH * dxHw3;
                Float3 mdo = pars.mO * dxOw1 + mdb + mdc;

                sm_threadVirial[0 * sc_workGroupSize + threadIdx] =
                        -(x_ow1[0] * mdo[0] + dist21[0] * mdb[0] + dist31[0] * mdc[0]);
                sm_threadVirial[1 * sc_workGroupSize + threadIdx] =
                        -(x_ow1[0] * mdo[1] + dist21[0] * mdb[1] + dist31[0] * mdc[1]);
                sm_threadVirial[2 * sc_workGroupSize + threadIdx] =
                        -(x_ow1[0] * mdo[2] + dist21[0] * mdb[2] + dist31[0] * mdc[2]);
                sm_threadVirial[3 * sc_workGroupSize + threadIdx] =
                        -(x_ow1[1] * mdo[1] + dist21[1] * mdb[1] + dist31[1] * mdc[1]);
                sm_threadVirial[4 * sc_workGroupSize + threadIdx] =
                        -(x_ow1[1] * mdo[2] + dist21[1] * mdb[2] + dist31[1] * mdc[2]);
                sm_threadVirial[5 * sc_workGroupSize + threadIdx] =
                        -(x_ow1[2] * mdo[2] + dist21[2] * mdb[2] + dist31[2] * mdc[2]);
            }
        }
        else // settleIdx < numSettles
        {
            // Filling data for dummy threads with zeroes
            if constexpr (computeVirial)
            {
                for (int d = 0; d < 6; d++)
                {
                    sm_threadVirial[d * sc_workGroupSize + threadIdx] = 0.0F;
                }
            }
        }

        // Basic reduction for the values inside single thread block
        // TODO what follows should be separated out as a standard virial reduction subroutine
        if constexpr (computeVirial)
        {
            // This is to ensure that all threads saved the data before reduction starts
            subGroupBarrier(itemIdx);
            constexpr int blockSize    = sc_workGroupSize;
            const int     subGroupSize = itemIdx.get_sub_group().get_max_local_range()[0];
            // Reduce up to one virial per thread block
            // All blocks are divided by half, the first half of threads sums
            // two virials. Then the first half is divided by two and the first half
            // of it sums two values... The procedure continues until only one thread left.
            // Only works if the threads per blocks is a power of two, hence the assertion.
            static_assert(gmx::isPowerOfTwo(sc_workGroupSize));
            for (int dividedAt = blockSize / 2; dividedAt >= 1; dividedAt >>= 1)
            {
                if (threadIdx < dividedAt)
                {
                    for (int d = 0; d < 6; d++)
                    {
                        sm_threadVirial[d * blockSize + threadIdx] +=
                                sm_threadVirial[d * blockSize + (threadIdx + dividedAt)];
                    }
                }
                if (dividedAt > subGroupSize / 2)
                {
                    subGroupBarrier(itemIdx);
                }
            }
            // First 6 threads in the block add the 6 components of virial to the global memory address
            if (threadIdx < 6)
            {
                atomicFetchAdd(a_virialScaled, threadIdx, sm_threadVirial[threadIdx * blockSize]);
            }
        }
    };
}

// SYCL 1.2.1 requires providing a unique type for a kernel. Should not be needed for SYCL2020.
template<bool updateVelocities, bool computeVirial>
class SettleKernelName;

template<bool updateVelocities, bool computeVirial, class... Args>
static cl::sycl::event launchSettleKernel(const DeviceStream& deviceStream, int numSettles, Args&&... args)
{
    // Should not be needed for SYCL2020.
    using kernelNameType = SettleKernelName<updateVelocities, computeVirial>;

    const int numSettlesRoundedUp =
            static_cast<int>((numSettles + sc_workGroupSize - 1) / sc_workGroupSize) * sc_workGroupSize;
    const cl::sycl::nd_range<1> rangeAllSettles(numSettlesRoundedUp, sc_workGroupSize);
    cl::sycl::queue             q = deviceStream.stream();

    cl::sycl::event e = q.submit([&](cl::sycl::handler& cgh) {
        auto kernel = settleKernel<updateVelocities, computeVirial>(
                cgh, numSettles, std::forward<Args>(args)...);
        cgh.parallel_for<kernelNameType>(rangeAllSettles, kernel);
    });

    return e;
}

/*! \brief Select templated kernel and launch it. */
template<class... Args>
static inline cl::sycl::event launchSettleKernel(bool updateVelocities, bool computeVirial, Args&&... args)
{
    return dispatchTemplatedFunction(
            [&](auto updateVelocities_, auto computeVirial_) {
                return launchSettleKernel<updateVelocities_, computeVirial_>(std::forward<Args>(args)...);
            },
            updateVelocities,
            computeVirial);
}


void launchSettleGpuKernel(const int                          numSettles,
                           const DeviceBuffer<WaterMolecule>& d_settles,
                           const SettleParameters&            settleParameters,
                           const DeviceBuffer<Float3>&        d_x,
                           DeviceBuffer<Float3>               d_xp,
                           const bool                         updateVelocities,
                           DeviceBuffer<Float3>               d_v,
                           const real                         invdt,
                           const bool                         computeVirial,
                           DeviceBuffer<float>                virialScaled,
                           const PbcAiuc&                     pbcAiuc,
                           const DeviceStream&                deviceStream)
{

    launchSettleKernel(updateVelocities,
                       computeVirial,
                       deviceStream,
                       numSettles,
                       d_settles,
                       settleParameters,
                       d_x,
                       d_xp,
                       invdt,
                       d_v,
                       virialScaled,
                       pbcAiuc);
    return;
}

} // namespace gmx
