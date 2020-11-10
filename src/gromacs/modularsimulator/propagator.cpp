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
 * \brief Defines the propagator element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "propagator.h"

#include "gromacs/utility.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/fatalerror.h"

#include "modularsimulator.h"
#include "simulatoralgorithm.h"
#include "statepropagatordata.h"

namespace gmx
{
//! Update velocities
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues>
static void inline updateVelocities(int         a,
                                    real        dt,
                                    real        lambdaStart,
                                    real        lambdaEnd,
                                    const rvec* gmx_restrict invMassPerDim,
                                    rvec* gmx_restrict v,
                                    const rvec* gmx_restrict f,
                                    const rvec               diagPR,
                                    const matrix             matrixPR)
{
    for (int d = 0; d < DIM; d++)
    {
        // TODO: Extract this into policy classes
        if (numStartVelocityScalingValues != NumVelocityScalingValues::None
            && parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::No)
        {
            v[a][d] *= lambdaStart;
        }
        if (numStartVelocityScalingValues != NumVelocityScalingValues::None
            && parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::Diagonal)
        {
            v[a][d] *= (lambdaStart - diagPR[d]);
        }
        if (numStartVelocityScalingValues != NumVelocityScalingValues::None
            && parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::Full)
        {
            v[a][d] = lambdaStart * v[a][d] - iprod(matrixPR[d], v[a]);
        }
        if (numStartVelocityScalingValues == NumVelocityScalingValues::None
            && parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::Diagonal)
        {
            v[a][d] *= (1 - diagPR[d]);
        }
        if (numStartVelocityScalingValues == NumVelocityScalingValues::None
            && parrinelloRahmanVelocityScaling == ParrinelloRahmanVelocityScaling::Full)
        {
            v[a][d] -= iprod(matrixPR[d], v[a]);
        }
        v[a][d] += f[a][d] * invMassPerDim[a][d] * dt;
        if (numEndVelocityScalingValues != NumVelocityScalingValues::None)
        {
            v[a][d] *= lambdaEnd;
        }
    }
}

//! Update positions
static void inline updatePositions(int         a,
                                   real        dt,
                                   const rvec* gmx_restrict x,
                                   rvec* gmx_restrict xprime,
                                   const rvec* gmx_restrict v)
{
    for (int d = 0; d < DIM; d++)
    {
        xprime[a][d] = x[a][d] + v[a][d] * dt;
    }
}

//! Scale velocities
template<NumVelocityScalingValues numStartVelocityScalingValues>
static void inline scaleVelocities(int a, real lambda, rvec* gmx_restrict v)
{
    if (numStartVelocityScalingValues != NumVelocityScalingValues::None)
    {
        for (int d = 0; d < DIM; d++)
        {
            v[a][d] *= lambda;
        }
    }
}

//! Scale positions
template<NumPositionScalingValues numPositionScalingValues>
static void inline scalePositions(int a, real lambda, rvec* gmx_restrict x)
{
    if (numPositionScalingValues != NumPositionScalingValues::None)
    {
        for (int d = 0; d < DIM; d++)
        {
            x[a][d] *= lambda;
        }
    }
}

//! Helper function diagonalizing the PR matrix if possible
template<ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling>
static inline bool diagonalizePRMatrix(matrix matrixPR, rvec diagPR)
{
    if (parrinelloRahmanVelocityScaling != ParrinelloRahmanVelocityScaling::Full)
    {
        return false;
    }
    else
    {
        if (matrixPR[YY][XX] == 0 && matrixPR[ZZ][XX] == 0 && matrixPR[ZZ][YY] == 0)
        {
            diagPR[XX] = matrixPR[XX][XX];
            diagPR[YY] = matrixPR[YY][YY];
            diagPR[ZZ] = matrixPR[ZZ][ZZ];
            return true;
        }
        else
        {
            return false;
        }
    }
}

//! Propagation (position only)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStep::PositionsOnly>::run()
{
    wallcycle_start(wcycle_, ewcUPDATE);

    auto xp = as_rvec_array(statePropagatorData_->positionsView().paddedArrayRef().data());
    auto x  = as_rvec_array(statePropagatorData_->constPositionsView().paddedArrayRef().data());
    auto v  = as_rvec_array(statePropagatorData_->constVelocitiesView().paddedArrayRef().data());

    int nth    = gmx_omp_nthreads_get(emntUpdate);
    int homenr = mdAtoms_->mdatoms()->homenr;

#pragma omp parallel for num_threads(nth) schedule(static) default(none) shared(nth, homenr, x, xp, v)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th, end_th;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                updatePositions(a, timestep_, x, xp, v);
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, ewcUPDATE);
}

//! Propagation (scale position only)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStep::ScalePositions>::run()
{
    wallcycle_start(wcycle_, ewcUPDATE);

    auto* x = as_rvec_array(statePropagatorData_->positionsView().paddedArrayRef().data());

    const real lambda =
            (numPositionScalingValues == NumPositionScalingValues::Single) ? positionScaling_[0] : 1.0;

    int nth    = gmx_omp_nthreads_get(emntUpdate);
    int homenr = mdAtoms_->mdatoms()->homenr;

#pragma omp parallel for num_threads(nth) schedule(static) default(none) shared(nth, homenr, x) \
        firstprivate(lambda)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th, end_th;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                scalePositions<numPositionScalingValues>(
                        a,
                        (numPositionScalingValues == NumPositionScalingValues::Multiple)
                                ? positionScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                : lambda,
                        x);
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, ewcUPDATE);
}

//! Propagation (velocity only)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStep::VelocitiesOnly>::run()
{
    wallcycle_start(wcycle_, ewcUPDATE);

    auto v = as_rvec_array(statePropagatorData_->velocitiesView().paddedArrayRef().data());
    auto f = as_rvec_array(statePropagatorData_->constForcesView().force().data());
    auto invMassPerDim = mdAtoms_->mdatoms()->invMassPerDim;

    const real lambdaStart = (numStartVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? startVelocityScaling_[0]
                                     : 1.0;
    const real lambdaEnd = (numEndVelocityScalingValues == NumVelocityScalingValues::Single)
                                   ? endVelocityScaling_[0]
                                   : 1.0;

    const bool isFullScalingMatrixDiagonal =
            diagonalizePRMatrix<parrinelloRahmanVelocityScaling>(matrixPR_, diagPR_);

    const int nth    = gmx_omp_nthreads_get(emntUpdate);
    const int homenr = mdAtoms_->mdatoms()->homenr;

// const variables could be shared, but gcc-8 & gcc-9 don't agree how to write that...
// https://www.gnu.org/software/gcc/gcc-9/porting_to.html -> OpenMP data sharing
#pragma omp parallel for num_threads(nth) schedule(static) default(none) shared(v, f, invMassPerDim) \
        firstprivate(nth, homenr, lambdaStart, lambdaEnd, isFullScalingMatrixDiagonal)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th, end_th;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                if (isFullScalingMatrixDiagonal)
                {
                    updateVelocities<numStartVelocityScalingValues, ParrinelloRahmanVelocityScaling::Diagonal, numEndVelocityScalingValues>(
                            a,
                            timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim,
                            v,
                            f,
                            diagPR_,
                            matrixPR_);
                }
                else
                {
                    updateVelocities<numStartVelocityScalingValues, parrinelloRahmanVelocityScaling, numEndVelocityScalingValues>(
                            a,
                            timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim,
                            v,
                            f,
                            diagPR_,
                            matrixPR_);
                }
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, ewcUPDATE);
}

//! Propagation (leapfrog case - position and velocity)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStep::LeapFrog>::run()
{
    wallcycle_start(wcycle_, ewcUPDATE);

    auto xp = as_rvec_array(statePropagatorData_->positionsView().paddedArrayRef().data());
    auto x  = as_rvec_array(statePropagatorData_->constPositionsView().paddedArrayRef().data());
    auto v  = as_rvec_array(statePropagatorData_->velocitiesView().paddedArrayRef().data());
    auto f  = as_rvec_array(statePropagatorData_->constForcesView().force().data());
    auto invMassPerDim = mdAtoms_->mdatoms()->invMassPerDim;

    const real lambdaStart = (numStartVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? startVelocityScaling_[0]
                                     : 1.0;
    const real lambdaEnd = (numEndVelocityScalingValues == NumVelocityScalingValues::Single)
                                   ? endVelocityScaling_[0]
                                   : 1.0;

    const bool isFullScalingMatrixDiagonal =
            diagonalizePRMatrix<parrinelloRahmanVelocityScaling>(matrixPR_, diagPR_);

    const int nth    = gmx_omp_nthreads_get(emntUpdate);
    const int homenr = mdAtoms_->mdatoms()->homenr;

// const variables could be shared, but gcc-8 & gcc-9 don't agree how to write that...
// https://www.gnu.org/software/gcc/gcc-9/porting_to.html -> OpenMP data sharing
#pragma omp parallel for num_threads(nth) schedule(static) default(none) \
        shared(x, xp, v, f, invMassPerDim)                               \
                firstprivate(nth, homenr, lambdaStart, lambdaEnd, isFullScalingMatrixDiagonal)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th, end_th;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                if (isFullScalingMatrixDiagonal)
                {
                    updateVelocities<numStartVelocityScalingValues, ParrinelloRahmanVelocityScaling::Diagonal, numEndVelocityScalingValues>(
                            a,
                            timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim,
                            v,
                            f,
                            diagPR_,
                            matrixPR_);
                }
                else
                {
                    updateVelocities<numStartVelocityScalingValues, parrinelloRahmanVelocityScaling, numEndVelocityScalingValues>(
                            a,
                            timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim,
                            v,
                            f,
                            diagPR_,
                            matrixPR_);
                }
                updatePositions(a, timestep_, x, xp, v);
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, ewcUPDATE);
}

//! Propagation (velocity verlet stage 2 - velocity and position)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStep::VelocityVerletPositionsAndVelocities>::run()
{
    wallcycle_start(wcycle_, ewcUPDATE);

    auto xp = as_rvec_array(statePropagatorData_->positionsView().paddedArrayRef().data());
    auto x  = as_rvec_array(statePropagatorData_->constPositionsView().paddedArrayRef().data());
    auto v  = as_rvec_array(statePropagatorData_->velocitiesView().paddedArrayRef().data());
    auto f  = as_rvec_array(statePropagatorData_->constForcesView().force().data());
    auto invMassPerDim = mdAtoms_->mdatoms()->invMassPerDim;

    const real lambdaStart = (numStartVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? startVelocityScaling_[0]
                                     : 1.0;
    const real lambdaEnd = (numEndVelocityScalingValues == NumVelocityScalingValues::Single)
                                   ? endVelocityScaling_[0]
                                   : 1.0;

    const bool isFullScalingMatrixDiagonal =
            diagonalizePRMatrix<parrinelloRahmanVelocityScaling>(matrixPR_, diagPR_);

    const int nth    = gmx_omp_nthreads_get(emntUpdate);
    const int homenr = mdAtoms_->mdatoms()->homenr;

// const variables could be shared, but gcc-8 & gcc-9 don't agree how to write that...
// https://www.gnu.org/software/gcc/gcc-9/porting_to.html -> OpenMP data sharing
#pragma omp parallel for num_threads(nth) schedule(static) default(none) \
        shared(x, xp, v, f, invMassPerDim)                               \
                firstprivate(nth, homenr, lambdaStart, lambdaEnd, isFullScalingMatrixDiagonal)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th, end_th;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                if (isFullScalingMatrixDiagonal)
                {
                    updateVelocities<numStartVelocityScalingValues, ParrinelloRahmanVelocityScaling::Diagonal, numEndVelocityScalingValues>(
                            a,
                            0.5 * timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim,
                            v,
                            f,
                            diagPR_,
                            matrixPR_);
                }
                else
                {
                    updateVelocities<numStartVelocityScalingValues, parrinelloRahmanVelocityScaling, numEndVelocityScalingValues>(
                            a,
                            0.5 * timestep_,
                            numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaStart,
                            numEndVelocityScalingValues == NumVelocityScalingValues::Multiple
                                    ? endVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                    : lambdaEnd,
                            invMassPerDim,
                            v,
                            f,
                            diagPR_,
                            matrixPR_);
                }
                updatePositions(a, timestep_, x, xp, v);
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, ewcUPDATE);
}

//! Scaling (velocity scaling only)
template<>
template<NumVelocityScalingValues        numStartVelocityScalingValues,
         ParrinelloRahmanVelocityScaling parrinelloRahmanVelocityScaling,
         NumVelocityScalingValues        numEndVelocityScalingValues,
         NumPositionScalingValues        numPositionScalingValues>
void Propagator<IntegrationStep::ScaleVelocities>::run()
{
    if (numStartVelocityScalingValues == NumVelocityScalingValues::None)
    {
        return;
    }
    wallcycle_start(wcycle_, ewcUPDATE);

    auto* v = as_rvec_array(statePropagatorData_->velocitiesView().paddedArrayRef().data());

    const real lambdaStart = (numStartVelocityScalingValues == NumVelocityScalingValues::Single)
                                     ? startVelocityScaling_[0]
                                     : 1.0;

    const int nth    = gmx_omp_nthreads_get(emntUpdate);
    const int homenr = mdAtoms_->mdatoms()->homenr;

// const variables could be shared, but gcc-8 & gcc-9 don't agree how to write that...
// https://www.gnu.org/software/gcc/gcc-9/porting_to.html -> OpenMP data sharing
#pragma omp parallel for num_threads(nth) schedule(static) default(none) shared(v) \
        firstprivate(nth, homenr, lambdaStart)
    for (int th = 0; th < nth; th++)
    {
        try
        {
            int start_th = 0;
            int end_th   = 0;
            getThreadAtomRange(nth, th, homenr, &start_th, &end_th);

            for (int a = start_th; a < end_th; a++)
            {
                scaleVelocities<numStartVelocityScalingValues>(
                        a,
                        numStartVelocityScalingValues == NumVelocityScalingValues::Multiple
                                ? startVelocityScaling_[mdAtoms_->mdatoms()->cTC[a]]
                                : lambdaStart,
                        v);
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    wallcycle_stop(wcycle_, ewcUPDATE);
}

template<IntegrationStep algorithm>
Propagator<algorithm>::Propagator(double               timestep,
                                  StatePropagatorData* statePropagatorData,
                                  const MDAtoms*       mdAtoms,
                                  gmx_wallcycle*       wcycle) :
    timestep_(timestep),
    statePropagatorData_(statePropagatorData),
    doSingleStartVelocityScaling_(false),
    doGroupStartVelocityScaling_(false),
    doSingleEndVelocityScaling_(false),
    doGroupEndVelocityScaling_(false),
    scalingStepVelocity_(-1),
    diagPR_{ 0 },
    matrixPR_{ { 0 } },
    scalingStepPR_(-1),
    mdAtoms_(mdAtoms),
    wcycle_(wcycle)
{
}

template<IntegrationStep algorithm>
void Propagator<algorithm>::scheduleTask(Step gmx_unused step,
                                         Time gmx_unused            time,
                                         const RegisterRunFunction& registerRunFunction)
{
    const bool doSingleVScalingThisStep =
            (doSingleStartVelocityScaling_ && (step == scalingStepVelocity_));
    const bool doGroupVScalingThisStep = (doGroupStartVelocityScaling_ && (step == scalingStepVelocity_));

    if (algorithm == IntegrationStep::ScaleVelocities)
    {
        if (!doSingleVScalingThisStep && !doGroupVScalingThisStep)
        {
            return;
        }
    }

    if (algorithm == IntegrationStep::ScalePositions)
    {
        if (step != scalingStepPosition_)
        {
            return;
        }
        if (doSinglePositionScaling_)
        {
            registerRunFunction([this]() {
                run<NumVelocityScalingValues::None,
                    ParrinelloRahmanVelocityScaling::No,
                    NumVelocityScalingValues::None,
                    NumPositionScalingValues::Single>();
            });
        }
        else if (doGroupPositionScaling_)
        {
            registerRunFunction([this]() {
                run<NumVelocityScalingValues::None,
                    ParrinelloRahmanVelocityScaling::No,
                    NumVelocityScalingValues::None,
                    NumPositionScalingValues::Multiple>();
            });
        }
    }

    const bool doParrinelloRahmanThisStep = (step == scalingStepPR_);

    if (doSingleVScalingThisStep)
    {
        if (doParrinelloRahmanThisStep)
        {
            if (doSingleEndVelocityScaling_)
            {
                registerRunFunction([this]() {
                    run<NumVelocityScalingValues::Single,
                        ParrinelloRahmanVelocityScaling::Full,
                        NumVelocityScalingValues::Single,
                        NumPositionScalingValues::None>();
                });
            }
            else
            {
                registerRunFunction([this]() {
                    run<NumVelocityScalingValues::Single,
                        ParrinelloRahmanVelocityScaling::Full,
                        NumVelocityScalingValues::None,
                        NumPositionScalingValues::None>();
                });
            }
        }
        else
        {
            if (doSingleEndVelocityScaling_)
            {
                registerRunFunction([this]() {
                    run<NumVelocityScalingValues::Single,
                        ParrinelloRahmanVelocityScaling::No,
                        NumVelocityScalingValues::Single,
                        NumPositionScalingValues::None>();
                });
            }
            else
            {
                registerRunFunction([this]() {
                    run<NumVelocityScalingValues::Single,
                        ParrinelloRahmanVelocityScaling::No,
                        NumVelocityScalingValues::None,
                        NumPositionScalingValues::None>();
                });
            }
        }
    }
    else if (doGroupVScalingThisStep)
    {
        if (doParrinelloRahmanThisStep)
        {
            if (doGroupEndVelocityScaling_)
            {
                registerRunFunction([this]() {
                    run<NumVelocityScalingValues::Multiple,
                        ParrinelloRahmanVelocityScaling::Full,
                        NumVelocityScalingValues::Multiple,
                        NumPositionScalingValues::None>();
                });
            }
            else
            {
                registerRunFunction([this]() {
                    run<NumVelocityScalingValues::Multiple,
                        ParrinelloRahmanVelocityScaling::Full,
                        NumVelocityScalingValues::None,
                        NumPositionScalingValues::None>();
                });
            }
        }
        else
        {
            if (doGroupEndVelocityScaling_)
            {
                registerRunFunction([this]() {
                    run<NumVelocityScalingValues::Multiple,
                        ParrinelloRahmanVelocityScaling::No,
                        NumVelocityScalingValues::Multiple,
                        NumPositionScalingValues::None>();
                });
            }
            else
            {
                registerRunFunction([this]() {
                    run<NumVelocityScalingValues::Multiple,
                        ParrinelloRahmanVelocityScaling::No,
                        NumVelocityScalingValues::None,
                        NumPositionScalingValues::None>();
                });
            }
        }
    }
    else
    {
        if (doParrinelloRahmanThisStep)
        {
            registerRunFunction([this]() {
                run<NumVelocityScalingValues::None,
                    ParrinelloRahmanVelocityScaling::Full,
                    NumVelocityScalingValues::None,
                    NumPositionScalingValues::None>();
            });
        }
        else
        {
            registerRunFunction([this]() {
                run<NumVelocityScalingValues::None,
                    ParrinelloRahmanVelocityScaling::No,
                    NumVelocityScalingValues::None,
                    NumPositionScalingValues::None>();
            });
        }
    }
}

template<IntegrationStep algorithm>
constexpr bool hasStartVelocityScaling()
{
    return (algorithm == IntegrationStep::VelocitiesOnly || algorithm == IntegrationStep::LeapFrog
            || algorithm == IntegrationStep::VelocityVerletPositionsAndVelocities
            || algorithm == IntegrationStep::ScaleVelocities);
}

template<IntegrationStep algorithm>
constexpr bool hasEndVelocityScaling()
{
    return (hasStartVelocityScaling<algorithm>() && algorithm != IntegrationStep::ScaleVelocities);
}

template<IntegrationStep algorithm>
constexpr bool hasPositionScaling()
{
    return (algorithm == IntegrationStep::ScalePositions);
}

template<IntegrationStep algorithm>
constexpr bool hasParrinelloRahmanScaling()
{
    return (algorithm == IntegrationStep::VelocitiesOnly || algorithm == IntegrationStep::LeapFrog
            || algorithm == IntegrationStep::VelocityVerletPositionsAndVelocities);
}

template<IntegrationStep algorithm>
void Propagator<algorithm>::setNumVelocityScalingVariables(int numVelocityScalingVariables,
                                                           ScaleVelocities scaleVelocities)
{
    GMX_RELEASE_ASSERT(hasStartVelocityScaling<algorithm>() || hasEndVelocityScaling<algorithm>(),
                       "Velocity scaling not implemented.");
    GMX_RELEASE_ASSERT(startVelocityScaling_.empty(),
                       "Number of velocity scaling variables cannot be changed once set.");

    const bool scaleEndVelocities = (scaleVelocities == ScaleVelocities::PreStepAndPostStep);
    startVelocityScaling_.resize(numVelocityScalingVariables, 1.);
    if (scaleEndVelocities)
    {
        endVelocityScaling_.resize(numVelocityScalingVariables, 1.);
    }
    doSingleStartVelocityScaling_ = numVelocityScalingVariables == 1;
    doGroupStartVelocityScaling_  = numVelocityScalingVariables > 1;
    doSingleEndVelocityScaling_   = doSingleStartVelocityScaling_ && scaleEndVelocities;
    doGroupEndVelocityScaling_    = doGroupStartVelocityScaling_ && scaleEndVelocities;
}

template<IntegrationStep algorithm>
void Propagator<algorithm>::setNumPositionScalingVariables(int numPositionScalingVariables)
{
    GMX_RELEASE_ASSERT(hasPositionScaling<algorithm>(), "Position scaling not implemented.");
    GMX_RELEASE_ASSERT(positionScaling_.empty(),
                       "Number of position scaling variables cannot be changed once set.");
    positionScaling_.resize(numPositionScalingVariables, 1.);
    doSinglePositionScaling_ = (numPositionScalingVariables == 1);
    doGroupPositionScaling_  = (numPositionScalingVariables > 1);
}

template<IntegrationStep algorithm>
ArrayRef<real> Propagator<algorithm>::viewOnStartVelocityScaling()
{
    GMX_RELEASE_ASSERT(hasStartVelocityScaling<algorithm>(),
                       "Start velocity scaling not implemented.");
    GMX_RELEASE_ASSERT(!startVelocityScaling_.empty(),
                       "Number of velocity scaling variables not set.");

    return startVelocityScaling_;
}

template<IntegrationStep algorithm>
ArrayRef<real> Propagator<algorithm>::viewOnEndVelocityScaling()
{
    GMX_RELEASE_ASSERT(hasEndVelocityScaling<algorithm>(), "End velocity scaling not implemented.");
    GMX_RELEASE_ASSERT(!endVelocityScaling_.empty(),
                       "Number of velocity scaling variables not set.");

    return endVelocityScaling_;
}

template<IntegrationStep algorithm>
ArrayRef<real> Propagator<algorithm>::viewOnPositionScaling()
{
    GMX_RELEASE_ASSERT(hasPositionScaling<algorithm>(), "Position scaling not implemented.");
    GMX_RELEASE_ASSERT(!positionScaling_.empty(), "Number of position scaling variables not set.");

    return positionScaling_;
}

template<IntegrationStep algorithm>
PropagatorCallback Propagator<algorithm>::velocityScalingCallback()
{
    GMX_RELEASE_ASSERT(hasStartVelocityScaling<algorithm>() || hasEndVelocityScaling<algorithm>(),
                       "Velocity scaling not implemented.");

    return [this](Step step) { scalingStepVelocity_ = step; };
}

template<IntegrationStep algorithm>
PropagatorCallback Propagator<algorithm>::positionScalingCallback()
{
    GMX_RELEASE_ASSERT(hasPositionScaling<algorithm>(), "Position scaling not implemented.");

    return [this](Step step) { scalingStepPosition_ = step; };
}

template<IntegrationStep algorithm>
ArrayRef<rvec> Propagator<algorithm>::viewOnPRScalingMatrix()
{
    GMX_RELEASE_ASSERT(algorithm != IntegrationStep::PositionsOnly
                               && algorithm != IntegrationStep::ScaleVelocities,
                       "Parrinello-Rahman scaling not implemented for "
                       "IntegrationStep::PositionsOnly and IntegrationStep::ScaleVelocities.");

    clear_mat(matrixPR_);
    // gcc-5 needs this to be explicit (all other tested compilers would be ok
    // with simply returning matrixPR)
    return ArrayRef<rvec>(matrixPR_);
}

template<IntegrationStep algorithm>
PropagatorCallback Propagator<algorithm>::prScalingCallback()
{
    GMX_RELEASE_ASSERT(algorithm != IntegrationStep::PositionsOnly
                               && algorithm != IntegrationStep::ScaleVelocities,
                       "Parrinello-Rahman scaling not implemented for "
                       "IntegrationStep::PositionsOnly and IntegrationStep::ScaleVelocities.");

    return [this](Step step) { scalingStepPR_ = step; };
}

template<IntegrationStep algorithm>
static PropagatorConnection getConnection(Propagator<algorithm>* propagator, const PropagatorTag& propagatorTag)
{
    PropagatorConnection propagatorConnection{ propagatorTag };

    propagatorConnection.startVelocityScaling    = hasStartVelocityScaling<algorithm>();
    propagatorConnection.endVelocityScaling      = hasEndVelocityScaling<algorithm>();
    propagatorConnection.positionScaling         = hasPositionScaling<algorithm>();
    propagatorConnection.parrinelloRahmanScaling = hasParrinelloRahmanScaling<algorithm>();

    propagatorConnection.setNumVelocityScalingVariables =
            [propagator](int num, ScaleVelocities scaleVelocities) {
                propagator->setNumVelocityScalingVariables(num, scaleVelocities);
            };
    propagatorConnection.setNumPositionScalingVariables = [propagator](int num) {
        propagator->setNumPositionScalingVariables(num);
    };
    propagatorConnection.getViewOnStartVelocityScaling = [propagator]() {
        return propagator->viewOnStartVelocityScaling();
    };
    propagatorConnection.getViewOnEndVelocityScaling = [propagator]() {
        return propagator->viewOnEndVelocityScaling();
    };
    propagatorConnection.getViewOnPositionScaling = [propagator]() {
        return propagator->viewOnPositionScaling();
    };
    propagatorConnection.getVelocityScalingCallback = [propagator]() {
        return propagator->velocityScalingCallback();
    };
    propagatorConnection.getPositionScalingCallback = [propagator]() {
        return propagator->positionScalingCallback();
    };
    propagatorConnection.getViewOnPRScalingMatrix = [propagator]() {
        return propagator->viewOnPRScalingMatrix();
    };
    propagatorConnection.getPRScalingCallback = [propagator]() {
        return propagator->prScalingCallback();
    };

    return propagatorConnection;
}

template<IntegrationStep algorithm>
ISimulatorElement* Propagator<algorithm>::getElementPointerImpl(
        LegacySimulatorData*                    legacySimulatorData,
        ModularSimulatorAlgorithmBuilderHelper* builderHelper,
        StatePropagatorData*                    statePropagatorData,
        EnergyData gmx_unused*     energyData,
        FreeEnergyPerturbationData gmx_unused* freeEnergyPerturbationData,
        GlobalCommunicationHelper gmx_unused* globalCommunicationHelper,
        const PropagatorTag&                  propagatorTag,
        double                                timestep)
{
    GMX_RELEASE_ASSERT(!(algorithm == IntegrationStep::ScaleVelocities
                         || algorithm == IntegrationStep::ScalePositions)
                               || (timestep == 0.0),
                       "Scaling elements don't propagate the system.");
    auto* element    = builderHelper->storeElement(std::make_unique<Propagator<algorithm>>(
            timestep, statePropagatorData, legacySimulatorData->mdAtoms, legacySimulatorData->wcycle));
    auto* propagator = static_cast<Propagator<algorithm>*>(element);
    builderHelper->registerPropagator(getConnection<algorithm>(propagator, propagatorTag));
    return element;
}

template<IntegrationStep algorithm>
ISimulatorElement*
Propagator<algorithm>::getElementPointerImpl(LegacySimulatorData* legacySimulatorData,
                                             ModularSimulatorAlgorithmBuilderHelper* builderHelper,
                                             StatePropagatorData*        statePropagatorData,
                                             EnergyData*                 energyData,
                                             FreeEnergyPerturbationData* freeEnergyPerturbationData,
                                             GlobalCommunicationHelper*  globalCommunicationHelper,
                                             const PropagatorTag&        propagatorTag)
{
    GMX_RELEASE_ASSERT(algorithm == IntegrationStep::ScaleVelocities
                               || algorithm == IntegrationStep::ScalePositions,
                       "Adding a propagator without timestep is only allowed for scaling elements");
    return getElementPointerImpl(legacySimulatorData,
                                 builderHelper,
                                 statePropagatorData,
                                 energyData,
                                 freeEnergyPerturbationData,
                                 globalCommunicationHelper,
                                 propagatorTag,
                                 0.0);
}

// Explicit template initializations
template class Propagator<IntegrationStep::PositionsOnly>;
template class Propagator<IntegrationStep::VelocitiesOnly>;
template class Propagator<IntegrationStep::LeapFrog>;
template class Propagator<IntegrationStep::VelocityVerletPositionsAndVelocities>;
template class Propagator<IntegrationStep::ScaleVelocities>;
template class Propagator<IntegrationStep::ScalePositions>;

} // namespace gmx
