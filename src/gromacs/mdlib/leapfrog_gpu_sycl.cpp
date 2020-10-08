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

namespace gmx
{

/*! \brief Main kernel for Leap-Frog integrator.
 *
 *  The coordinates and velocities are updated on the GPU. Also saves the intermediate values of the coordinates for
 *   further use in constraints.
 *
 *  Each GPU thread works with a single particle. Empty declaration is needed to
 *  avoid "no previous prototype for function" clang warning.
 *
 *  \todo Check if the force should be set to zero here.
 *  \todo This kernel can also accumulate incidental temperatures for each atom.
 *
 * \tparam        numTempScaleValues               The number of different T-couple values.
 * \tparam        velocityScaling                  Type of the Parrinello-Rahman velocity rescaling.
 * \param[in]     numAtoms                         Total number of atoms.
 * \param[in,out] gm_x                             Coordinates to update upon integration.
 * \param[out]    gm_xp                            A copy of the coordinates before the integration (for constraints).
 * \param[in,out] gm_v                             Velocities to update.
 * \param[in]     gm_f                             Atomic forces.
 * \param[in]     gm_inverseMasses                 Reciprocal masses.
 * \param[in]     dt                               Timestep.
 * \param[in]     gm_lambdas                       Temperature scaling factors (one per group)
 * \param[in]     gm_tempScaleGroups               Mapping of atoms into groups.
 * \param[in]     dtPressureCouple                 Time step for pressure coupling
 * \param[in]     prVelocityScalingMatrixDiagonal  Diagonal elements of Parrinello-Rahman velocity scaling matrix
 */
template<NumTempScaleValues numTempScaleValues, VelocityScalingType velocityScaling>
class SyclLeapFrogKernelFunctor
{
public:
    SyclLeapFrogKernelFunctor(cl::sycl::handler&            cgh,
                              int                           numAtoms,
                              DeviceBuffer<float3>&         x,
                              DeviceBuffer<float3>&         xp,
                              DeviceBuffer<float3>&         v,
                              DeviceBuffer<float3>&         f,
                              DeviceBuffer<float>&          inverseMasses,
                              float                         dt,
                              DeviceBuffer<float>&          lambdas,
                              DeviceBuffer<unsigned short>& tempScaleGroups,
                              float3                        prVelocityScalingMatrixDiagonal) :
        numAtoms_(numAtoms),
        x_(x, cgh),
        xp_(xp, cgh),
        v_(v, cgh),
        f_(f, cgh),
        inverseMasses_(inverseMasses, cgh),
        dt_(dt),
        lambdas_(lambdas, cgh),
        tempScaleGroups_(tempScaleGroups, cgh),
        prVelocityScalingMatrixDiagonal_(prVelocityScalingMatrixDiagonal)
    {
    }
    void operator()(cl::sycl::id<1> itemIdx);

private:
    static constexpr bool _haveLambdas         = numTempScaleValues != NumTempScaleValues::None;
    static constexpr bool _haveTempScaleGroups = numTempScaleValues == NumTempScaleValues::Multiple;
    int                   numAtoms_;
    DeviceAccessor<float3, cl::sycl::access::mode::read_write>        x_;
    DeviceAccessor<float3, cl::sycl::access::mode::discard_write>     xp_;
    DeviceAccessor<float3, cl::sycl::access::mode::read_write>        v_;
    DeviceAccessor<float3, cl::sycl::access::mode::read>              f_;
    DeviceAccessor<float, cl::sycl::access::mode::read>               inverseMasses_;
    float                                                             dt_;
    DeviceAccessor<float, cl::sycl::access::mode::read, _haveLambdas> lambdas_;
    DeviceAccessor<unsigned short, cl::sycl::access::mode::read, _haveTempScaleGroups> tempScaleGroups_;
    float3 prVelocityScalingMatrixDiagonal_;
};

class IKernelLauncher
{
public:
    virtual ~IKernelLauncher()                                             = default;
    virtual cl::sycl::event launch(const DeviceStream&           deviceStream,
                                   int                           numAtoms,
                                   DeviceBuffer<float3>&         x,
                                   DeviceBuffer<float3>&         xp,
                                   DeviceBuffer<float3>&         v,
                                   DeviceBuffer<float3>&         f,
                                   DeviceBuffer<float>&          inverseMasses,
                                   float                         dt,
                                   DeviceBuffer<float>&          lambdas,
                                   DeviceBuffer<unsigned short>& tempScaleGroups,
                                   float3 prVelocityScalingMatrixDiagonal) = 0;
};

template<NumTempScaleValues numTempScaleValues, VelocityScalingType velocityScaling>
class KernelLauncher : public IKernelLauncher
{
public:
    virtual ~KernelLauncher() override = default;
    virtual cl::sycl::event launch(const DeviceStream&           deviceStream,
                                   int                           numAtoms,
                                   DeviceBuffer<float3>&         x,
                                   DeviceBuffer<float3>&         xp,
                                   DeviceBuffer<float3>&         v,
                                   DeviceBuffer<float3>&         f,
                                   DeviceBuffer<float>&          inverseMasses,
                                   float                         dt,
                                   DeviceBuffer<float>&          lambdas,
                                   DeviceBuffer<unsigned short>& tempScaleGroups,
                                   float3 prVelocityScalingMatrixDiagonal) final;
};

template<NumTempScaleValues numTempScaleValues, VelocityScalingType velocityScaling>
void SyclLeapFrogKernelFunctor<numTempScaleValues, velocityScaling>::operator()(cl::sycl::id<1> itemIdx)
{
    const float3 x    = x_[itemIdx];
    float3       v    = v_[itemIdx];
    const float3 f    = f_[itemIdx];
    const float  im   = inverseMasses_[itemIdx];
    const float  imdt = im * dt_;

    // Swapping places for xp and x so that the x will contain the updated coordinates and xp - the
    // coordinates before update. This should be taken into account when (if) constraints are
    // applied after the update: x and xp have to be passed to constraints in the 'wrong' order.
    xp_[itemIdx] = x;

    if (numTempScaleValues != NumTempScaleValues::None || velocityScaling != VelocityScalingType::None)
    {
        float3 vp = v;

        if (numTempScaleValues != NumTempScaleValues::None)
        {
            const float lambda = [=]() {
                if (numTempScaleValues == NumTempScaleValues::Single)
                {
                    return lambdas_[0];
                }
                else if (numTempScaleValues == NumTempScaleValues::Multiple)
                {
                    const int tempScaleGroup = tempScaleGroups_[itemIdx];
                    return lambdas_[tempScaleGroup];
                }
                else
                {
                    return 1.0F; // Should be unreachable
                }
            }();
            vp *= lambda;
        }

        if (velocityScaling == VelocityScalingType::Diagonal)
        {
            vp[0] -= prVelocityScalingMatrixDiagonal_[0] * v[0];
            vp[1] -= prVelocityScalingMatrixDiagonal_[1] * v[1];
            vp[2] -= prVelocityScalingMatrixDiagonal_[2] * v[2];
        }

        v = vp;
    }

    v += f * imdt;
    v_[itemIdx] = v;
    x_[itemIdx] = x + v * dt_;
}

template<NumTempScaleValues numTempScaleValues, VelocityScalingType velocityScaling>
cl::sycl::event
KernelLauncher<numTempScaleValues, velocityScaling>::launch(const DeviceStream&   deviceStream,
                                                            int                   numAtoms,
                                                            DeviceBuffer<float3>& x,
                                                            DeviceBuffer<float3>& xp,
                                                            DeviceBuffer<float3>& v,
                                                            DeviceBuffer<float3>& f,
                                                            DeviceBuffer<float>&  inverseMasses,
                                                            float                 dt,
                                                            DeviceBuffer<float>&  lambdas,
                                                            DeviceBuffer<unsigned short>& tempScaleGroups,
                                                            float3 prVelocityScalingMatrixDiagonal)
{
    const cl::sycl::range<1> rangeAllAtoms(numAtoms);

    cl::sycl::queue q = deviceStream.stream();

    cl::sycl::event e = q.submit([&](cl::sycl::handler& cgh) {
        auto kernel = SyclLeapFrogKernelFunctor<numTempScaleValues, velocityScaling>(
                cgh, numAtoms, x, xp, v, f, inverseMasses, dt, lambdas, tempScaleGroups,
                prVelocityScalingMatrixDiagonal);
        cgh.parallel_for(rangeAllAtoms, kernel);
    });

    return e;
}

/*! \brief Select templated kernel. */
inline IKernelLauncher* selectLeapFrogKernelLauncher(bool                doTemperatureScaling,
                                                     int                 numTempScaleValues,
                                                     VelocityScalingType prVelocityScalingType)
{
    // Check input for consistency: if there is temperature coupling, at least one coupling group should be defined.
    GMX_ASSERT(!doTemperatureScaling || (numTempScaleValues > 0),
               "Temperature coupling was requested with no temperature coupling groups.");

    if (prVelocityScalingType == VelocityScalingType::None)
    {
        if (!doTemperatureScaling)
        {
            return new KernelLauncher<NumTempScaleValues::None, VelocityScalingType::None>;
        }
        else if (numTempScaleValues == 1)
        {
            return new KernelLauncher<NumTempScaleValues::Single, VelocityScalingType::None>;
        }
        else if (numTempScaleValues > 1)
        {
            return new KernelLauncher<NumTempScaleValues::Multiple, VelocityScalingType::None>;
        }
    }
    else if (prVelocityScalingType == VelocityScalingType::Diagonal)
    {
        if (!doTemperatureScaling)
        {
            return new KernelLauncher<NumTempScaleValues::None, VelocityScalingType::Diagonal>;
        }
        else if (numTempScaleValues == 1)
        {
            return new KernelLauncher<NumTempScaleValues::Single, VelocityScalingType::Diagonal>;
        }
        else if (numTempScaleValues > 1)
        {
            return new KernelLauncher<NumTempScaleValues::Multiple, VelocityScalingType::Diagonal>;
        }
    }
    else
    {
        GMX_RELEASE_ASSERT(false,
                           "Only isotropic Parrinello-Rahman pressure coupling is supported.");
    }
    return new KernelLauncher<NumTempScaleValues::None, VelocityScalingType::None>;
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
    std::unique_ptr<IKernelLauncher> kernelLauncher;
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
        kernelLauncher.reset(selectLeapFrogKernelLauncher(doTemperatureScaling, numTempScaleValues_,
                                                          prVelocityScalingType));
    }
    else
    {
        kernelLauncher =
                std::make_unique<KernelLauncher<NumTempScaleValues::None, VelocityScalingType::None>>();
    }

    gmx_used_in_debug cl::sycl::event e =
            kernelLauncher->launch(deviceStream_, numAtoms_, d_x, d_xp, d_v, d_f, d_inverseMasses_, dt,
                                   d_lambdas_, d_tempScaleGroups_, prVelocityScalingMatrixDiagonal_);

#ifndef NDEBUG
    /* There will be synchronization when we will copy the data back to host, but for
     * debug it will be easier to wait right here */
    e.wait_and_throw();
#endif
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
