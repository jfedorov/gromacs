/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020, by the GROMACS development team, led by
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
 * \brief Implements Leap-Frog using SYCL
 *
 * This file contains implementation of basic Leap-Frog integrator
 * using SYCL, including class initialization, data-structures management
 * and GPU kernel.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 * \author Andrey Alekseenko <al42and@gmail.com>
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include <memory>

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/leapfrog_gpu.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"

namespace gmx
{

/*! \brief Main kernel for Leap-Frog integrator.
 *
 *  The coordinates and velocities are updated on the GPU. Also saves the intermediate values of the coordinates for
 *   further use in constraints.
 *
 *  Each GPU thread works with a single particle.
 *
 * \tparam        numTempScaleValues               The number of different T-couple values.
 * \tparam        velocityScaling                  Type of the Parrinello-Rahman velocity rescaling.
 * \param         cgh                              SYCL's command group handler.
 * \param[in,out] x                                Coordinates to update upon integration.
 * \param[out]    xp                               A copy of the coordinates before the integration (for constraints).
 * \param[in,out] v                                Velocities to update.
 * \param[in]     f                                Atomic forces.
 * \param[in]     inverseMasses                    Reciprocal masses.
 * \param[in]     dt                               Timestep.
 * \param[in]     lambdas                          Temperature scaling factors (one per group).
 * \param[in]     tempScaleGroups                  Mapping of atoms into groups.
 * \param[in]     dtPressureCouple                 Time step for pressure coupling.
 * \param[in]     prVelocityScalingMatrixDiagonal  Diagonal elements of Parrinello-Rahman velocity scaling matrix
 */
template<NumTempScaleValues numTempScaleValues, VelocityScalingType velocityScaling>
auto syclLeapFrogKernel(cl::sycl::handler&            cgh,
                        DeviceBuffer<float3>&         x,
                        DeviceBuffer<float3>&         xp,
                        DeviceBuffer<float3>&         v,
                        DeviceBuffer<float3>&         f,
                        DeviceBuffer<float>&          inverseMasses,
                        float                         dt,
                        DeviceBuffer<float>&          lambdas,
                        DeviceBuffer<unsigned short>& tempScaleGroups,
                        float3                        prVelocityScalingMatrixDiagonal)
{
    using cl::sycl::access::mode;
    auto x_             = get_access<mode::read_write>(x, cgh);
    auto xp_            = get_access<mode::discard_write>(xp, cgh);
    auto v_             = get_access<mode::read_write>(v, cgh);
    auto f_             = get_access<mode::read>(f, cgh);
    auto inverseMasses_ = get_access<mode::read>(inverseMasses, cgh);
    auto lambdas_       = [&]() {
        if constexpr (numTempScaleValues != NumTempScaleValues::None)
        {
            return get_access<mode::read>(lambdas, cgh);
        }
        else
            return nullptr;
    }();
    auto tempScaleGroups_ = [&]() {
        if constexpr (numTempScaleValues == NumTempScaleValues::Multiple)
        {
            return get_access<mode::read>(tempScaleGroups, cgh);
        }
        else
            return nullptr;
    }();

    return [=](cl::sycl::id<1> itemIdx) {
        const float3 x    = x_[itemIdx];
        float3       v    = v_[itemIdx];
        const float3 f    = f_[itemIdx];
        const float  im   = inverseMasses_[itemIdx];
        const float  imdt = im * dt;

        // Swapping places for xp and x so that the x will contain the updated coordinates and xp - the
        // coordinates before update. This should be taken into account when (if) constraints are
        // applied after the update: x and xp have to be passed to constraints in the 'wrong' order.
        xp_[itemIdx] = x;

        if constexpr (numTempScaleValues != NumTempScaleValues::None
                      || velocityScaling != VelocityScalingType::None)
        {
            float3 vp = v;

            if constexpr (numTempScaleValues != NumTempScaleValues::None)
            {
                const float lambda = [=]() {
                    if constexpr (numTempScaleValues == NumTempScaleValues::Single)
                    {
                        return lambdas_[0];
                    }
                    else
                    {
                        static_assert(numTempScaleValues == NumTempScaleValues::Multiple,
                                      "Invalid value of numTempScaleValues");
                        const int tempScaleGroup = tempScaleGroups_[itemIdx];
                        return lambdas_[tempScaleGroup];
                    }
                }();
                vp *= lambda;
            }

            if constexpr (velocityScaling == VelocityScalingType::Diagonal)
            {
                vp[0] -= prVelocityScalingMatrixDiagonal[0] * v[0];
                vp[1] -= prVelocityScalingMatrixDiagonal[1] * v[1];
                vp[2] -= prVelocityScalingMatrixDiagonal[2] * v[2];
            }

            v = vp;
        }

        v += f * imdt;
        v_[itemIdx] = v;
        x_[itemIdx] = x + v * dt;
    };
}

template<NumTempScaleValues numTempScaleValues, VelocityScalingType velocityScaling>
class SyclLeapFrogKernelName;
template<NumTempScaleValues numTempScaleValues, VelocityScalingType velocityScaling>
static cl::sycl::event launchKernel(const DeviceStream&           deviceStream,
                                    int                           numAtoms,
                                    DeviceBuffer<float3>&         x,
                                    DeviceBuffer<float3>&         xp,
                                    DeviceBuffer<float3>&         v,
                                    DeviceBuffer<float3>&         f,
                                    DeviceBuffer<float>&          inverseMasses,
                                    float                         dt,
                                    DeviceBuffer<float>&          lambdas,
                                    DeviceBuffer<unsigned short>& tempScaleGroups,
                                    float3                        prVelocityScalingMatrixDiagonal)
{
    const cl::sycl::range<1> rangeAllAtoms(numAtoms);

    cl::sycl::queue q = deviceStream.stream();

    cl::sycl::event e = q.submit([&](cl::sycl::handler& cgh) {
        auto kernel = syclLeapFrogKernel<numTempScaleValues, velocityScaling>(
                cgh, x, xp, v, f, inverseMasses, dt, lambdas, tempScaleGroups,
                prVelocityScalingMatrixDiagonal);
        // QUESTION: Is it OK for us to compile with -fsycl-unnamed-kernel?
        cgh.parallel_for<SyclLeapFrogKernelName<numTempScaleValues, velocityScaling>>(rangeAllAtoms, kernel);
    });

    return e;
}

template<enum VelocityScalingType prVelocityScalingType>
static inline auto* selectLeapFrogKernelLauncher(bool doTemperatureScaling, int numTempScaleValues)
{
    if (!doTemperatureScaling)
    {
        return launchKernel<NumTempScaleValues::None, prVelocityScalingType>;
    }
    else if (numTempScaleValues == 1)
    {
        return launchKernel<NumTempScaleValues::Single, prVelocityScalingType>;
    }
    else if (numTempScaleValues > 1)
    {
        return launchKernel<NumTempScaleValues::Multiple, prVelocityScalingType>;
    }
    else
    {
        gmx_incons("Temperature coupling was requested with no temperature coupling groups.");
    }
}

/*! \brief Select templated kernel. */
static inline auto* selectLeapFrogKernelLauncher(bool                doTemperatureScaling,
                                                 int                 numTempScaleValues,
                                                 VelocityScalingType prVelocityScalingType)
{
    if (prVelocityScalingType == VelocityScalingType::None)
    {
        return selectLeapFrogKernelLauncher<VelocityScalingType::None>(doTemperatureScaling,
                                                                       numTempScaleValues);
    }
    else if (prVelocityScalingType == VelocityScalingType::Diagonal)
    {
        return selectLeapFrogKernelLauncher<VelocityScalingType::Diagonal>(doTemperatureScaling,
                                                                           numTempScaleValues);
    }
    else
    {
        gmx_incons("Only isotropic Parrinello-Rahman pressure coupling is supported.");
    }
}

void LeapFrogGpu::integrate(DeviceBuffer<float3>              d_x,
                            DeviceBuffer<float3>              d_xp,
                            DeviceBuffer<float3>              d_v,
                            DeviceBuffer<float3>              d_f,
                            const real                        dt,
                            const bool                        doTemperatureScaling,
                            gmx::ArrayRef<const t_grp_tcstat> tcstat,
                            const bool                        doParrinelloRahman,
                            const float                       dtPressureCouple,
                            const matrix                      prVelocityScalingMatrix)
{
    auto* kernelLauncher = launchKernel<NumTempScaleValues::None, VelocityScalingType::None>;
    if (doTemperatureScaling || doParrinelloRahman)
    {
        if (doTemperatureScaling)
        {
            GMX_ASSERT(numTempScaleValues_ == ssize(h_lambdas_),
                       "Number of temperature scaling factors changed since it was set for the "
                       "last time.");
            for (int i = 0; i < numTempScaleValues_; i++)
            {
                h_lambdas_[i] = tcstat[i].lambda;
            }
            copyToDeviceBuffer(&d_lambdas_, h_lambdas_.data(), 0, numTempScaleValues_,
                               deviceStream_, GpuApiCallBehavior::Async, nullptr);
        }
        VelocityScalingType prVelocityScalingType = VelocityScalingType::None;
        if (doParrinelloRahman)
        {
            prVelocityScalingType = VelocityScalingType::Diagonal;
            GMX_ASSERT(prVelocityScalingMatrix[YY][XX] == 0 && prVelocityScalingMatrix[ZZ][XX] == 0
                               && prVelocityScalingMatrix[ZZ][YY] == 0
                               && prVelocityScalingMatrix[XX][YY] == 0
                               && prVelocityScalingMatrix[XX][ZZ] == 0
                               && prVelocityScalingMatrix[YY][ZZ] == 0,
                       "Fully anisotropic Parrinello-Rahman pressure coupling is not yet supported "
                       "in GPU version of Leap-Frog integrator.");
            prVelocityScalingMatrixDiagonal_ =
                    float3{ dtPressureCouple * prVelocityScalingMatrix[XX][XX],
                            dtPressureCouple * prVelocityScalingMatrix[YY][YY],
                            dtPressureCouple * prVelocityScalingMatrix[ZZ][ZZ] };
        }
        kernelLauncher = selectLeapFrogKernelLauncher(doTemperatureScaling, numTempScaleValues_,
                                                      prVelocityScalingType);
    }

    kernelLauncher(deviceStream_, numAtoms_, d_x, d_xp, d_v, d_f, d_inverseMasses_, dt, d_lambdas_,
                   d_tempScaleGroups_, prVelocityScalingMatrixDiagonal_);
}

LeapFrogGpu::LeapFrogGpu(const DeviceContext& deviceContext, const DeviceStream& deviceStream) :
    deviceContext_(deviceContext),
    deviceStream_(deviceStream),
    numAtoms_(0)
{
}

LeapFrogGpu::~LeapFrogGpu()
{
    freeDeviceBuffer(&d_inverseMasses_);
}

void LeapFrogGpu::set(const int             numAtoms,
                      const real*           inverseMasses,
                      const int             numTempScaleValues,
                      const unsigned short* tempScaleGroups)
{
    numAtoms_           = numAtoms;
    numTempScaleValues_ = numTempScaleValues;

    reallocateDeviceBuffer(&d_inverseMasses_, numAtoms_, &numInverseMasses_,
                           &numInverseMassesAlloc_, deviceContext_);
    copyToDeviceBuffer(&d_inverseMasses_, inverseMasses, 0, numAtoms_, deviceStream_,
                       GpuApiCallBehavior::Sync, nullptr);

    // Temperature scale group map only used if there are more then one group
    if (numTempScaleValues_ > 1)
    {
        reallocateDeviceBuffer(&d_tempScaleGroups_, numAtoms_, &numTempScaleGroups_,
                               &numTempScaleGroupsAlloc_, deviceContext_);
        copyToDeviceBuffer(&d_tempScaleGroups_, tempScaleGroups, 0, numAtoms_, deviceStream_,
                           GpuApiCallBehavior::Sync, nullptr);
    }

    // If the temperature coupling is enabled, we need to make space for scaling factors
    if (numTempScaleValues_ > 0)
    {
        h_lambdas_.resize(numTempScaleValues);
        reallocateDeviceBuffer(&d_lambdas_, numTempScaleValues_, &numLambdas_, &numLambdasAlloc_,
                               deviceContext_);
    }
}

} // namespace gmx
