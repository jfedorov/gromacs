/*
 * This file is part of the GROMACS molecular simulation package.
 *
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
/*! \internal \file
 *
 * \brief Implements CUDA bonded functionality
 *
 * \author Jon Vincent <jvincent@nvidia.com>
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 * \author Berk Hess <hess@kth.se>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \author Alan Gray <alang@nvidia.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_listed_forces
 */

#include "gmxpre.h"

#include <cassert>

#include <math_constants.h>

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/gpu_utils/vectype_ops.cuh"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/gmxassert.h"

#include "listed_forces_gpu_impl.h"

#if defined(_MSVC)
#    include <limits>
#endif

/*-------------------------------- CUDA kernels-------------------------------- */
/*------------------------------------------------------------------------------*/

#define CUDA_DEG2RAD_F (CUDART_PI_F / 180.0F)
#define CUDA_RAD2DEG_F (180.0F / CUDART_PI_F)

namespace
{
/*! \brief Mysterious CMAP coefficient matrix */
__device__ const int cmap_coeff_matrix[] = {
    1,  0,  -3, 2,  0,  0, 0,  0,  -3, 0,  9,  -6, 2, 0,  -6, 4,  0,  0,  0, 0,  0, 0, 0,  0,
    3,  0,  -9, 6,  -2, 0, 6,  -4, 0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  9, -6, 0, 0, -6, 4,
    0,  0,  3,  -2, 0,  0, 0,  0,  0,  0,  -9, 6,  0, 0,  6,  -4, 0,  0,  0, 0,  1, 0, -3, 2,
    -2, 0,  6,  -4, 1,  0, -3, 2,  0,  0,  0,  0,  0, 0,  0,  0,  -1, 0,  3, -2, 1, 0, -3, 2,
    0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  -3, 2,  0, 0,  3,  -2, 0,  0,  0, 0,  0, 0, 3,  -2,
    0,  0,  -6, 4,  0,  0, 3,  -2, 0,  1,  -2, 1,  0, 0,  0,  0,  0,  -3, 6, -3, 0, 2, -4, 2,
    0,  0,  0,  0,  0,  0, 0,  0,  0,  3,  -6, 3,  0, -2, 4,  -2, 0,  0,  0, 0,  0, 0, 0,  0,
    0,  0,  -3, 3,  0,  0, 2,  -2, 0,  0,  -1, 1,  0, 0,  0,  0,  0,  0,  3, -3, 0, 0, -2, 2,
    0,  0,  0,  0,  0,  1, -2, 1,  0,  -2, 4,  -2, 0, 1,  -2, 1,  0,  0,  0, 0,  0, 0, 0,  0,
    0,  -1, 2,  -1, 0,  1, -2, 1,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  1, -1, 0, 0, -1, 1,
    0,  0,  0,  0,  0,  0, -1, 1,  0,  0,  2,  -2, 0, 0,  -1, 1
};

} // namespace

/*---------------- BONDED CUDA kernels--------------*/

/* Harmonic */
__device__ __forceinline__ static void
harmonic_gpu(const float kA, const float xA, const float x, float* V, float* F)
{
    constexpr float half = 0.5F;
    float           dx, dx2;

    dx  = x - xA;
    dx2 = dx * dx;

    *F = -kA * dx;
    *V = half * kA * dx2;
}

template<bool calcVir, bool calcEner>
__device__ void bonds_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          const float4    gm_xq[],
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        const int3 bondData = *(reinterpret_cast<const int3*>(d_forceatoms + 3 * i));
        int        type     = bondData.x;
        int        ai       = bondData.y;
        int        aj       = bondData.z;

        /* dx = xi - xj, corrected for periodic boundary conditions. */
        float3 dx;
        int    ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);

        float dr2 = norm2(dx);
        float dr  = sqrt(dr2);

        float vbond;
        float fbond;
        harmonic_gpu(d_forceparams[type].harmonic.krA, d_forceparams[type].harmonic.rA, dr, &vbond, &fbond);

        if (calcEner)
        {
            *vtot_loc += vbond;
        }

        if (dr2 != 0.0F)
        {
            fbond *= rsqrtf(dr2);

            float3 fij = fbond * dx;
            atomicAdd(&gm_f[ai], fij);
            atomicAdd(&gm_f[aj], -fij);
            if (calcVir && ki != gmx::c_centralShiftIndex)
            {
                atomicAdd(&sm_fShiftLoc[ki], fij);
                atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], -fij);
            }
        }
    }
}

template<bool returnShift>
__device__ __forceinline__ static float bond_angle_gpu(const float4   xi,
                                                       const float4   xj,
                                                       const float4   xk,
                                                       const PbcAiuc& pbcAiuc,
                                                       float3*        r_ij,
                                                       float3*        r_kj,
                                                       float*         costh,
                                                       int*           t1,
                                                       int*           t2)
/* Return value is the angle between the bonds i-j and j-k */
{
    *t1 = pbcDxAiuc<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xj, *r_kj);

    *costh   = cos_angle(*r_ij, *r_kj);
    float th = acosf(*costh);

    return th;
}

template<bool calcVir, bool calcEner>
__device__ void angles_gpu(const int       i,
                           float*          vtot_loc,
                           const int       numBonds,
                           const t_iatom   d_forceatoms[],
                           const t_iparams d_forceparams[],
                           const float4    gm_xq[],
                           float3          gm_f[],
                           float3          sm_fShiftLoc[],
                           const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        const int4 angleData = *(reinterpret_cast<const int4*>(d_forceatoms + 4 * i));
        int        type      = angleData.x;
        int        ai        = angleData.y;
        int        aj        = angleData.z;
        int        ak        = angleData.w;

        float3 r_ij;
        float3 r_kj;
        float  cos_theta;
        int    t1;
        int    t2;
        float  theta = bond_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_gpu(d_forceparams[type].harmonic.krA,
                     d_forceparams[type].harmonic.rA * CUDA_DEG2RAD_F,
                     theta,
                     &va,
                     &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0F)
        {
            float st    = dVdt * rsqrtf(1.0F - cos_theta2);
            float sth   = st * cos_theta;
            float nrij2 = norm2(r_ij);
            float nrkj2 = norm2(r_kj);

            float nrij_1 = rsqrtf(nrij2);
            float nrkj_1 = rsqrtf(nrkj2);

            float cik = st * nrij_1 * nrkj_1;
            float cii = sth * nrij_1 * nrij_1;
            float ckk = sth * nrkj_1 * nrkj_1;

            float3 f_i = cii * r_ij - cik * r_kj;
            float3 f_k = ckk * r_kj - cik * r_ij;
            float3 f_j = -f_i - f_k;

            atomicAdd(&gm_f[ai], f_i);
            atomicAdd(&gm_f[aj], f_j);
            atomicAdd(&gm_f[ak], f_k);

            if (calcVir)
            {
                atomicAdd(&sm_fShiftLoc[t1], f_i);
                atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], f_j);
                atomicAdd(&sm_fShiftLoc[t2], f_k);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void urey_bradley_gpu(const int       i,
                                 float*          vtot_loc,
                                 const int       numBonds,
                                 const t_iatom   d_forceatoms[],
                                 const t_iparams d_forceparams[],
                                 const float4    gm_xq[],
                                 float3          gm_f[],
                                 float3          sm_fShiftLoc[],
                                 const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        const int4 ubData = *(reinterpret_cast<const int4*>(d_forceatoms + 4 * i));
        int        type   = ubData.x;
        int        ai     = ubData.y;
        int        aj     = ubData.z;
        int        ak     = ubData.w;

        float th0A = d_forceparams[type].u_b.thetaA * CUDA_DEG2RAD_F;
        float kthA = d_forceparams[type].u_b.kthetaA;
        float r13A = d_forceparams[type].u_b.r13A;
        float kUBA = d_forceparams[type].u_b.kUBA;

        float3 r_ij;
        float3 r_kj;
        float  cos_theta;
        int    t1;
        int    t2;
        float  theta = bond_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_gpu(kthA, th0A, theta, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float3 r_ik;
        int    ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[ak], r_ik);

        float dr2 = norm2(r_ik);
        float dr  = dr2 * rsqrtf(dr2);

        float vbond;
        float fbond;
        harmonic_gpu(kUBA, r13A, dr, &vbond, &fbond);

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0F)
        {
            float st  = dVdt * rsqrtf(1.0F - cos_theta2);
            float sth = st * cos_theta;

            float nrkj2 = norm2(r_kj);
            float nrij2 = norm2(r_ij);

            float cik = st * rsqrtf(nrkj2 * nrij2);
            float cii = sth / nrij2;
            float ckk = sth / nrkj2;

            float3 f_i = cii * r_ij - cik * r_kj;
            float3 f_k = ckk * r_kj - cik * r_ij;
            float3 f_j = -f_i - f_k;

            atomicAdd(&gm_f[ai], f_i);
            atomicAdd(&gm_f[aj], f_j);
            atomicAdd(&gm_f[ak], f_k);

            if (calcVir)
            {
                atomicAdd(&sm_fShiftLoc[t1], f_i);
                atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], f_j);
                atomicAdd(&sm_fShiftLoc[t2], f_k);
            }
        }

        /* Time for the bond calculations */
        if (dr2 != 0.0F)
        {
            if (calcEner)
            {
                *vtot_loc += vbond;
            }

            fbond *= rsqrtf(dr2);

            float3 fik = fbond * r_ik;
            atomicAdd(&gm_f[ai], fik);
            atomicAdd(&gm_f[ak], -fik);

            if (calcVir && ki != gmx::c_centralShiftIndex)
            {
                atomicAdd(&sm_fShiftLoc[ki], fik);
                atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], -fik);
            }
        }
    }
}

template<bool returnShift, typename T>
__device__ __forceinline__ static float dih_angle_gpu(const T        xi,
                                                      const T        xj,
                                                      const T        xk,
                                                      const T        xl,
                                                      const PbcAiuc& pbcAiuc,
                                                      float3*        r_ij,
                                                      float3*        r_kj,
                                                      float3*        r_kl,
                                                      float3*        m,
                                                      float3*        n,
                                                      int*           t1,
                                                      int*           t2,
                                                      int*           t3)
{
    *t1 = pbcDxAiuc<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xj, *r_kj);
    *t3 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xl, *r_kl);

    *m         = cprod(*r_ij, *r_kj);
    *n         = cprod(*r_kj, *r_kl);
    float phi  = gmx_angle(*m, *n);
    float ipr  = iprod(*r_ij, *n);
    float sign = (ipr < 0.0F) ? -1.0F : 1.0F;
    phi        = sign * phi;

    return phi;
}


__device__ __forceinline__ static void
dopdihs_gpu(const float cpA, const float phiA, const int mult, const float phi, float* v, float* f)
{
    float mdphi, sdphi;

    mdphi = mult * phi - phiA * CUDA_DEG2RAD_F;
    sdphi = sinf(mdphi);
    *v    = cpA * (1.0F + cosf(mdphi));
    *f    = -cpA * mult * sdphi;
}

template<bool calcVir>
__device__ static void do_dih_fup_gpu(const int            i,
                                      const int            j,
                                      const int            k,
                                      const int            l,
                                      const float          ddphi,
                                      const float3         r_ij,
                                      const float3         r_kj,
                                      const float3         r_kl,
                                      const float3         m,
                                      const float3         n,
                                      float3               gm_f[],
                                      float3               sm_fShiftLoc[],
                                      const PbcAiuc&       pbcAiuc,
                                      const float4         gm_xq[],
                                      const int            t1,
                                      const int            t2,
                                      const int gmx_unused t3)
{
    float iprm  = norm2(m);
    float iprn  = norm2(n);
    float nrkj2 = norm2(r_kj);
    float toler = nrkj2 * GMX_REAL_EPS;
    if ((iprm > toler) && (iprn > toler))
    {
        float  nrkj_1 = rsqrtf(nrkj2); // replacing std::invsqrt call
        float  nrkj_2 = nrkj_1 * nrkj_1;
        float  nrkj   = nrkj2 * nrkj_1;
        float  a      = -ddphi * nrkj / iprm;
        float3 f_i    = a * m;
        float  b      = ddphi * nrkj / iprn;
        float3 f_l    = b * n;
        float  p      = iprod(r_ij, r_kj);
        p *= nrkj_2;
        float q = iprod(r_kl, r_kj);
        q *= nrkj_2;
        float3 uvec = p * f_i;
        float3 vvec = q * f_l;
        float3 svec = uvec - vvec;
        float3 f_j  = f_i - svec;
        float3 f_k  = f_l + svec;

        atomicAdd(&gm_f[i], f_i);
        atomicAdd(&gm_f[j], -f_j);
        atomicAdd(&gm_f[k], -f_k);
        atomicAdd(&gm_f[l], f_l);

        if (calcVir)
        {
            float3 dx_jl;
            int    t3 = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[l], gm_xq[j], dx_jl);

            atomicAdd(&sm_fShiftLoc[t1], f_i);
            atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], -f_j);
            atomicAdd(&sm_fShiftLoc[t2], -f_k);
            atomicAdd(&sm_fShiftLoc[t3], f_l);
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void pdihs_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          const float4    gm_xq[],
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        float vpd;
        float ddphi;
        dopdihs_gpu(d_forceparams[type].pdihs.cpA,
                    d_forceparams[type].pdihs.phiA,
                    d_forceparams[type].pdihs.mult,
                    phi,
                    &vpd,
                    &ddphi);

        if (calcEner)
        {
            *vtot_loc += vpd;
        }

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);
    }
}

template<bool calcVir, bool calcEner>
__device__ void rbdihs_gpu(const int       i,
                           float*          vtot_loc,
                           const int       numBonds,
                           const t_iatom   d_forceatoms[],
                           const t_iparams d_forceparams[],
                           const float4    gm_xq[],
                           float3          gm_f[],
                           float3          sm_fShiftLoc[],
                           const PbcAiuc   pbcAiuc)
{
    constexpr float c0 = 0.0F, c1 = 1.0F, c2 = 2.0F, c3 = 3.0F, c4 = 4.0F, c5 = 5.0F;

    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        /* Change to polymer convention */
        if (phi < c0)
        {
            phi += CUDART_PI_F;
        }
        else
        {
            phi -= CUDART_PI_F;
        }
        float cos_phi = cosf(phi);
        /* Beware of accuracy loss, cannot use 1-sqrt(cos^2) ! */
        float sin_phi = sinf(phi);

        float parm[NR_RBDIHS];
        for (int j = 0; j < NR_RBDIHS; j++)
        {
            parm[j] = d_forceparams[type].rbdihs.rbcA[j];
        }
        /* Calculate cosine powers */
        /* Calculate the energy */
        /* Calculate the derivative */
        float v      = parm[0];
        float ddphi  = c0;
        float cosfac = c1;

        float rbp = parm[1];
        ddphi += rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[2];
        ddphi += c2 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[3];
        ddphi += c3 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[4];
        ddphi += c4 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[5];
        ddphi += c5 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }

        ddphi = -ddphi * sin_phi;

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);
        if (calcEner)
        {
            *vtot_loc += v;
        }
    }
}

__device__ __forceinline__ static void make_dp_periodic_gpu(float* dp)
{
    /* dp cannot be outside (-pi,pi) */
    if (*dp >= CUDART_PI_F)
    {
        *dp -= 2.0F * CUDART_PI_F;
    }
    else if (*dp < -CUDART_PI_F)
    {
        *dp += 2.0F * CUDART_PI_F;
    }
}

template<bool calcVir, bool calcEner>
__device__ void idihs_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          const float4    gm_xq[],
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        /* phi can jump if phi0 is close to Pi/-Pi, which will cause huge
         * force changes if we just apply a normal harmonic.
         * Instead, we first calculate phi-phi0 and take it modulo (-Pi,Pi).
         * This means we will never have the periodicity problem, unless
         * the dihedral is Pi away from phiO, which is very unlikely due to
         * the potential.
         */
        float kA = d_forceparams[type].harmonic.krA;
        float pA = d_forceparams[type].harmonic.rA;

        float phi0 = pA * CUDA_DEG2RAD_F;

        float dp = phi - phi0;

        make_dp_periodic_gpu(&dp);

        float ddphi = -kA * dp;

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, -ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);

        if (calcEner)
        {
            *vtot_loc += -0.5F * ddphi * dp;
        }
    }
}

/*! \brief Map indices into the periodic grid */
__device__ static int cmap_setup_grid_index(int ip, int grid_spacing, int* ipm1, int* ipp1, int* ipp2)
{
    int im1, ip1, ip2;

    if (ip < 0)
    {
        ip = ip + grid_spacing - 1;
    }
    else if (ip > grid_spacing)
    {
        ip = ip - grid_spacing - 1;
    }

    im1 = ip - 1;
    ip1 = ip + 1;
    ip2 = ip + 2;

    if (ip == 0)
    {
        im1 = grid_spacing - 1;
    }
    else if (ip == grid_spacing - 2)
    {
        ip2 = 0;
    }
    else if (ip == grid_spacing - 1)
    {
        ip1 = 0;
        ip2 = 1;
    }

    *ipm1 = im1;
    *ipp1 = ip1;
    *ipp2 = ip2;

    return ip;
}

__device__ static float3 processCmapForceComponent(const float a,
                                                   const float b,
                                                   const float df,
                                                   const float gaa,
                                                   const float fga,
                                                   const float gbb,
                                                   const float hgb)
{
    float3 result{ gaa * a, fga * a - hgb * b, gbb * b }; // mapping x <-> f, y <-> g, z <-> h
    return result * df;
}

__device__ static float4 applyCmapForceComponent(const float3 forceComponent)
{
    // forceComponent mapping is x <-> f, y <-> g, z <-> h
    float4 forces;
    forces.x = forceComponent.x;
    forces.y = -forceComponent.x - forceComponent.y;
    forces.z = forceComponent.z + forceComponent.y;
    forces.w = -forceComponent.z;
    return forces;
}

template<bool calcVir>
__device__ static void accumulateCmapForces(float3        gm_f[],
                                            const float4  gm_xq[],
                                            float3        sm_fShiftLoc[],
                                            const PbcAiuc pbcAiuc,
                                            float3        r_ij,
                                            float3        r_kj,
                                            float3        r_kl,
                                            float3        a,
                                            float3        b,
                                            float3        h,
                                            float         ra2r,
                                            float         rb2r,
                                            float         rgr,
                                            float         rg,
                                            int           ai,
                                            int           aj,
                                            int           ak,
                                            int           al,
                                            float         df,
                                            int           t1,
                                            int           t2)
{
    const float fg  = iprod(r_ij, r_kj);
    const float hg  = iprod(r_kl, r_kj);
    const float fga = fg * ra2r * rgr;
    const float hgb = hg * rb2r * rgr;
    const float gaa = -ra2r * rg;
    const float gbb = rb2r * rg;

    float3 f_i, f_j, f_k, f_l;
    float4 forceX = applyCmapForceComponent(processCmapForceComponent(a.x, b.x, df, gaa, fga, gbb, hgb));
    float4 forceY = applyCmapForceComponent(processCmapForceComponent(a.y, b.y, df, gaa, fga, gbb, hgb));
    float4 forceZ = applyCmapForceComponent(processCmapForceComponent(a.z, b.z, df, gaa, fga, gbb, hgb));
    f_i.x         = forceX.x;
    f_j.x         = forceX.y;
    f_k.x         = forceX.z;
    f_l.x         = forceX.w;

    f_i.y = forceY.x;
    f_j.y = forceY.y;
    f_k.y = forceY.z;
    f_l.y = forceY.w;

    f_i.z = forceZ.x;
    f_j.z = forceZ.y;
    f_k.z = forceZ.z;
    f_l.z = forceZ.w;

    atomicAdd(&gm_f[ai], f_i);
    atomicAdd(&gm_f[aj], f_j); /* - f[i] - g[i] */
    atomicAdd(&gm_f[ak], f_k); /* h[i] + g[i] */
    atomicAdd(&gm_f[al], f_l); /* - h[i] */

    /* Shift forces */
    if (calcVir)
    {
        int t3 = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[al], gm_xq[aj], h);
        atomicAdd(&sm_fShiftLoc[t1], f_i);
        atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], f_j);
        atomicAdd(&sm_fShiftLoc[t2], f_k);
        atomicAdd(&sm_fShiftLoc[t3], f_l);
    }
}

/*! \brief
Magic lookup table for cmap indices

This maps the entries in the CMAP tables to the atoms in the
interactions
*/
__device__ static const int loopIndex[4][4] = { { 0, 4, 8, 12 },
                                                { 1, 5, 9, 13 },
                                                { 2, 6, 10, 14 },
                                                { 3, 7, 11, 15 } };

template<bool calcVir, bool calcEner>
__device__ void cmap_gpu(const int                  i,
                         float*                     vtot_loc,
                         const int                  numBonds,
                         const t_iatom              d_forceatoms[],
                         const t_iparams            d_forceparams[],
                         const int                  cmapGridSpacing,
                         const DeviceBuffer<float>& d_cmapData,
                         const int                  dc_cmapGridSize,
                         const float4               gm_xq[],
                         float3                     gm_f[],
                         float3                     sm_fShiftLoc[],
                         const PbcAiuc              pbcAiuc)
{
    if (i < numBonds)
    {
        /* Five atoms are involved in the two torsions */
        const int type = d_forceatoms[6 * i];
        const int ai   = d_forceatoms[6 * i + 1];
        const int aj   = d_forceatoms[6 * i + 2];
        const int ak   = d_forceatoms[6 * i + 3];
        const int al   = d_forceatoms[6 * i + 4];
        const int am   = d_forceatoms[6 * i + 5];

        /* Which CMAP type is this */
        const int    cmapA          = d_forceparams[type].cmap.cmapA;
        const int    cmapAGridIndex = dc_cmapGridSize * cmapA;
        const float* cmapd          = d_cmapData + cmapAGridIndex;

        int ip1m1, ip2m1, ip1p1, ip2p1, ip1p2, ip2p2;

        /* First torsion */
        const int a1i = ai;
        const int a1j = aj;
        const int a1k = ak;
        const int a1l = al;

        float3 r1_ij, r1_kj, r1_kl, r2_ij, r2_kj, r2_kl, m, n;
        float3 h1, h2;
        int    t11, t21, t31, t12, t22, t32;
        float  phi1 = dih_angle_gpu<calcVir>(
                gm_xq[a1i], gm_xq[a1j], gm_xq[a1k], gm_xq[a1l], pbcAiuc, &r1_ij, &r1_kj, &r1_kl, &m, &n, &t11, &t21, &t31);

        const float cos_phi1 = cos(phi1);

        float3 a1 = cprod(r1_ij, r1_kj);
        float3 b1 = cprod(r1_kl, r1_kj);

        const int ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[a1i], gm_xq[a1k], h1);

        const float ra21 = iprod(a1, a1);       /* 5 */
        const float rb21 = iprod(b1, b1);       /* 5 */
        const float rg21 = iprod(r1_kj, r1_kj); /* 5 */
        const float rg1  = sqrt(rg21);

        const float rgr1  = 1.0F / rg1;
        const float ra2r1 = 1.0F / ra21;
        const float rb2r1 = 1.0F / rb21;
        const float rabr1 = sqrt(ra2r1 * rb2r1);

        const float sin_phi1 = rg1 * rabr1 * iprod(a1, h1) * (-1.0F);

        if (cos_phi1 < -0.5F || cos_phi1 > 0.5F)
        {
            phi1 = asin(sin_phi1);

            if (cos_phi1 < 0.0F)
            {
                if (phi1 > 0.0F)
                {
                    phi1 = CUDART_PI_F - phi1;
                }
                else
                {
                    phi1 = -CUDART_PI_F - phi1;
                }
            }
        }
        else
        {
            phi1 = acos(cos_phi1);

            if (sin_phi1 < 0.0F)
            {
                phi1 = -phi1;
            }
        }

        float xphi1 = phi1 + CUDART_PI_F; /* 1 */

        /* Second torsion */
        const int a2i = aj;
        const int a2j = ak;
        const int a2k = al;
        const int a2l = am;

        float phi2 = dih_angle_gpu<calcVir>(
                gm_xq[a2i], gm_xq[a2j], gm_xq[a2k], gm_xq[a2l], pbcAiuc, &r2_ij, &r2_kj, &r2_kl, &m, &n, &t12, &t22, &t32);

        float cos_phi2 = cos(phi2);

        float3 a2 = cprod(r2_ij, r2_kj);
        float3 b2 = cprod(r2_kl, r2_kj);

        pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[a2i], gm_xq[a2k], h2);

        const float ra22 = iprod(a2, a2);       /* 5 */
        const float rb22 = iprod(b2, b2);       /* 5 */
        const float rg22 = iprod(r2_kj, r2_kj); /* 5 */
        const float rg2  = sqrt(rg22);

        const float rgr2  = 1.0F / rg2;
        const float ra2r2 = 1.0F / ra22;
        const float rb2r2 = 1.0F / rb22;
        const float rabr2 = sqrt(ra2r2 * rb2r2);

        const float sin_phi2 = rg2 * rabr2 * iprod(a2, h2) * (-1.0F);

        if (cos_phi2 < -0.5F || cos_phi2 > 0.5F)
        {
            phi2 = asin(sin_phi2);

            if (cos_phi2 < 0.0F)
            {
                if (phi2 > 0.0F)
                {
                    phi2 = CUDART_PI_F - phi2;
                }
                else
                {
                    phi2 = -CUDART_PI_F - phi2;
                }
            }
        }
        else
        {
            phi2 = acos(cos_phi2);

            if (sin_phi2 < 0.0F)
            {
                phi2 = -phi2;
            }
        }

        float xphi2 = phi2 + CUDART_PI_F; /* 1 */
        /* Range mangling */
        if (xphi1 < 0)
        {
            xphi1 = xphi1 + 2 * CUDART_PI_F;
        }
        else if (xphi1 >= 2 * CUDART_PI_F)
        {
            xphi1 = xphi1 - 2 * CUDART_PI_F;
        }

        if (xphi2 < 0)
        {
            xphi2 = xphi2 + 2 * CUDART_PI_F;
        }
        else if (xphi2 >= 2 * CUDART_PI_F)
        {
            xphi2 = xphi2 - 2 * CUDART_PI_F;
        }

        /* Number of grid points */
        float dx = 2 * CUDART_PI_F / cmapGridSpacing;

        /* Where on the grid are we */
        int iphi1 = static_cast<int>(xphi1 / dx);
        int iphi2 = static_cast<int>(xphi2 / dx);

        iphi1 = cmap_setup_grid_index(iphi1, cmapGridSpacing, &ip1m1, &ip1p1, &ip1p2);
        iphi2 = cmap_setup_grid_index(iphi2, cmapGridSpacing, &ip2m1, &ip2p1, &ip2p2);

        const int pos1 = iphi1 * cmapGridSpacing + iphi2;
        const int pos2 = ip1p1 * cmapGridSpacing + iphi2;
        const int pos3 = ip1p1 * cmapGridSpacing + ip2p1;
        const int pos4 = iphi1 * cmapGridSpacing + ip2p1;

        float4 ty, ty1, ty2, ty12, tx[4];
        ty.x = cmapd[pos1 * 4];
        ty.y = cmapd[pos2 * 4];
        ty.z = cmapd[pos3 * 4];
        ty.w = cmapd[pos4 * 4];

        ty1.x = cmapd[pos1 * 4 + 1];
        ty1.y = cmapd[pos2 * 4 + 1];
        ty1.z = cmapd[pos3 * 4 + 1];
        ty1.w = cmapd[pos4 * 4 + 1];

        ty2.x = cmapd[pos1 * 4 + 2];
        ty2.y = cmapd[pos2 * 4 + 2];
        ty2.z = cmapd[pos3 * 4 + 2];
        ty2.w = cmapd[pos4 * 4 + 2];

        ty12.x = cmapd[pos1 * 4 + 3];
        ty12.y = cmapd[pos2 * 4 + 3];
        ty12.z = cmapd[pos3 * 4 + 3];
        ty12.w = cmapd[pos4 * 4 + 3];

        /* Switch to degrees */
        dx    = 360.0F / cmapGridSpacing;
        xphi1 = xphi1 * CUDA_RAD2DEG_F;
        xphi2 = xphi2 * CUDA_RAD2DEG_F;


        tx[0] = ty;
        tx[1] = ty1 * dx;
        tx[2] = ty2 * dx;
        tx[3] = ty12 * dx * dx;
        // for (int i = 0; i < 4; i++) /* 16 */
        // {
        //     tx[i]      = ty[i];
        //     tx[i + 4]  = ty1[i] * dx;
        //     tx[i + 8]  = ty2[i] * dx;
        //     tx[i + 12] = ty12[i] * dx * dx;
        // }

        float tc[16] = { 0.0F };
        for (int idx = 0; idx < 16; idx++) /* 1056 */
        {
            int kIndex = 0;
            for (int k = 0; k < 4; k++)
            {
                tc[idx] += cmap_coeff_matrix[kIndex++ * 16 + idx] * tx[k].x;
                tc[idx] += cmap_coeff_matrix[kIndex++ * 16 + idx] * tx[k].y;
                tc[idx] += cmap_coeff_matrix[kIndex++ * 16 + idx] * tx[k].z;
                tc[idx] += cmap_coeff_matrix[kIndex++ * 16 + idx] * tx[k].w;
            }
        }

        const float tt = (xphi1 - iphi1 * dx) / dx;
        const float tu = (xphi2 - iphi2 * dx) / dx;

        float e   = 0.0F;
        float df1 = 0.0F;
        float df2 = 0.0F;

        for (int i = 3; i >= 0; i--)
        {
            int l1 = loopIndex[i][3];
            int l2 = loopIndex[i][2];
            int l3 = loopIndex[i][1];

            e = tt * e + ((tc[i * 4 + 3] * tu + tc[i * 4 + 2]) * tu + tc[i * 4 + 1]) * tu + tc[i * 4];
            df1 = tu * df1 + (3.0F * tc[l1] * tt + 2.0 * tc[l2]) * tt + tc[l3];
            df2 = tt * df2 + (3.0F * tc[i * 4 + 3] * tu + 2.0F * tc[i * 4 + 2]) * tu + tc[i * 4 + 1];
        }

        const float fac = CUDA_RAD2DEG_F / dx;
        df1             = df1 * fac;
        df2             = df2 * fac;

        /* CMAP energy */
        if (calcEner)
        {
            *vtot_loc += e;
        }

        /* Do forces - first torsion */
        accumulateCmapForces<calcVir>(gm_f,
                                      gm_xq,
                                      sm_fShiftLoc,
                                      pbcAiuc,
                                      r1_ij,
                                      r1_kj,
                                      r1_kl,
                                      a1,
                                      b1,
                                      h1,
                                      ra2r1,
                                      rb2r1,
                                      rgr1,
                                      rg1,
                                      a1i,
                                      a1j,
                                      a1k,
                                      a1l,
                                      df1,
                                      t11,
                                      t21);

        /* Do forces - second torsion */
        accumulateCmapForces<calcVir>(gm_f,
                                      gm_xq,
                                      sm_fShiftLoc,
                                      pbcAiuc,
                                      r2_ij,
                                      r2_kj,
                                      r2_kl,
                                      a2,
                                      b2,
                                      h2,
                                      ra2r2,
                                      rb2r2,
                                      rgr2,
                                      rg2,
                                      a2i,
                                      a2j,
                                      a2k,
                                      a2l,
                                      df2,
                                      t12,
                                      t22);
    }
}

template<bool calcVir, bool calcEner>
__device__ void pairs_gpu(const int       i,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams iparams[],
                          const float4    gm_xq[],
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc,
                          const float     scale_factor,
                          float*          vtotVdw_loc,
                          float*          vtotElec_loc)
{
    if (i < numBonds)
    {
        // TODO this should be made into a separate type, the GPU and CPU sizes should be compared
        const int3 pairData = *(reinterpret_cast<const int3*>(d_forceatoms + 3 * i));
        int        type     = pairData.x;
        int        ai       = pairData.y;
        int        aj       = pairData.z;

        float qq  = gm_xq[ai].w * gm_xq[aj].w;
        float c6  = iparams[type].lj14.c6A;
        float c12 = iparams[type].lj14.c12A;

        /* Do we need to apply full periodic boundary conditions? */
        float3 dr;
        int    fshift_index = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dr);

        float r2    = norm2(dr);
        float rinv  = rsqrtf(r2);
        float rinv2 = rinv * rinv;
        float rinv6 = rinv2 * rinv2 * rinv2;

        /* Calculate the Coulomb force * r */
        float velec = scale_factor * qq * rinv;

        /* Calculate the LJ force * r and add it to the Coulomb part */
        float fr = (12.0F * c12 * rinv6 - 6.0F * c6) * rinv6 + velec;

        float  finvr = fr * rinv2;
        float3 f     = finvr * dr;

        /* Add the forces */
        atomicAdd(&gm_f[ai], f);
        atomicAdd(&gm_f[aj], -f);
        if (calcVir && fshift_index != gmx::c_centralShiftIndex)
        {
            atomicAdd(&sm_fShiftLoc[fshift_index], f);
            atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], -f);
        }

        if (calcEner)
        {
            *vtotVdw_loc += (c12 * rinv6 - c6) * rinv6;
            *vtotElec_loc += velec;
        }
    }
}

namespace gmx
{

template<bool calcVir, bool calcEner>
__global__ void exec_kernel_gpu(BondedCudaKernelParameters kernelParams, float4* gm_xq, float3* gm_f, float3* gm_fShift)
{
    assert(blockDim.y == 1 && blockDim.z == 1);
    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    float     vtot_loc     = 0;
    float     vtotVdw_loc  = 0;
    float     vtotElec_loc = 0;

    extern __shared__ char sm_dynamicShmem[];
    char*                  sm_nextSlotPtr = sm_dynamicShmem;
    float3*                sm_fShiftLoc   = reinterpret_cast<float3*>(sm_nextSlotPtr);
    sm_nextSlotPtr += c_numShiftVectors * sizeof(float3);

    if (calcVir)
    {
        if (threadIdx.x < c_numShiftVectors)
        {
            sm_fShiftLoc[threadIdx.x] = make_float3(0.0F, 0.0F, 0.0F);
        }
        __syncthreads();
    }

    int  fType;
    bool threadComputedPotential = false;
#pragma unroll
    for (int j = 0; j < numFTypesOnGpu; j++)
    {
        if (tid >= kernelParams.fTypeRangeStart[j] && tid <= kernelParams.fTypeRangeEnd[j])
        {
            const int      numBonds        = kernelParams.numFTypeBonds[j];
            int            fTypeTid        = tid - kernelParams.fTypeRangeStart[j];
            const t_iatom* iatoms          = kernelParams.d_iatoms[j];
            const auto     cmapData        = kernelParams.d_cmapData;
            const int      cmapGridSpacing = kernelParams.d_cmapGridSpacing;
            const int      cmapGridSize    = kernelParams.dc_cmapGridSize;
            fType                          = kernelParams.fTypesOnGpu[j];
            if (calcEner)
            {
                threadComputedPotential = true;
            }

            switch (fType)
            {
                case F_BONDS:
                    bonds_gpu<calcVir, calcEner>(fTypeTid,
                                                 &vtot_loc,
                                                 numBonds,
                                                 iatoms,
                                                 kernelParams.d_forceParams,
                                                 gm_xq,
                                                 gm_f,
                                                 sm_fShiftLoc,
                                                 kernelParams.pbcAiuc);
                    break;
                case F_ANGLES:
                    angles_gpu<calcVir, calcEner>(fTypeTid,
                                                  &vtot_loc,
                                                  numBonds,
                                                  iatoms,
                                                  kernelParams.d_forceParams,
                                                  gm_xq,
                                                  gm_f,
                                                  sm_fShiftLoc,
                                                  kernelParams.pbcAiuc);
                    break;
                case F_UREY_BRADLEY:
                    urey_bradley_gpu<calcVir, calcEner>(fTypeTid,
                                                        &vtot_loc,
                                                        numBonds,
                                                        iatoms,
                                                        kernelParams.d_forceParams,
                                                        gm_xq,
                                                        gm_f,
                                                        sm_fShiftLoc,
                                                        kernelParams.pbcAiuc);
                    break;
                case F_PDIHS:
                case F_PIDIHS:
                    pdihs_gpu<calcVir, calcEner>(fTypeTid,
                                                 &vtot_loc,
                                                 numBonds,
                                                 iatoms,
                                                 kernelParams.d_forceParams,
                                                 gm_xq,
                                                 gm_f,
                                                 sm_fShiftLoc,
                                                 kernelParams.pbcAiuc);
                    break;
                case F_RBDIHS:
                    rbdihs_gpu<calcVir, calcEner>(fTypeTid,
                                                  &vtot_loc,
                                                  numBonds,
                                                  iatoms,
                                                  kernelParams.d_forceParams,
                                                  gm_xq,
                                                  gm_f,
                                                  sm_fShiftLoc,
                                                  kernelParams.pbcAiuc);
                    break;
                case F_IDIHS:
                    idihs_gpu<calcVir, calcEner>(fTypeTid,
                                                 &vtot_loc,
                                                 numBonds,
                                                 iatoms,
                                                 kernelParams.d_forceParams,
                                                 gm_xq,
                                                 gm_f,
                                                 sm_fShiftLoc,
                                                 kernelParams.pbcAiuc);
                    break;
                case F_CMAP:
                    cmap_gpu<calcVir, calcEner>(fTypeTid,
                                                &vtot_loc,
                                                numBonds,
                                                iatoms,
                                                kernelParams.d_forceParams,
                                                cmapGridSpacing,
                                                cmapData,
                                                cmapGridSize,
                                                kernelParams.d_xq,
                                                kernelParams.d_f,
                                                sm_fShiftLoc,
                                                kernelParams.pbcAiuc);
                    break;
                case F_LJ14:
                    pairs_gpu<calcVir, calcEner>(fTypeTid,
                                                 numBonds,
                                                 iatoms,
                                                 kernelParams.d_forceParams,
                                                 gm_xq,
                                                 gm_f,
                                                 sm_fShiftLoc,
                                                 kernelParams.pbcAiuc,
                                                 kernelParams.electrostaticsScaleFactor,
                                                 &vtotVdw_loc,
                                                 &vtotElec_loc);
                    break;
            }
            break;
        }
    }

    if (threadComputedPotential)
    {
        float* vtotVdw  = kernelParams.d_vTot + F_LJ14;
        float* vtotElec = kernelParams.d_vTot + F_COUL14;

        // Stage atomic accumulation through shared memory:
        // each warp will accumulate its own partial sum
        // and then a single thread per warp will accumulate this to the global sum

        int numWarps = blockDim.x / warpSize;
        int warpId   = threadIdx.x / warpSize;

        // Shared memory variables to hold block-local partial sum
        float* sm_vTot = reinterpret_cast<float*>(sm_nextSlotPtr);
        sm_nextSlotPtr += numWarps * sizeof(float);
        float* sm_vTotVdw = reinterpret_cast<float*>(sm_nextSlotPtr);
        sm_nextSlotPtr += numWarps * sizeof(float);
        float* sm_vTotElec = reinterpret_cast<float*>(sm_nextSlotPtr);

        if (threadIdx.x % warpSize == 0)
        {
            // One thread per warp initializes to zero
            sm_vTot[warpId]     = 0.;
            sm_vTotVdw[warpId]  = 0.;
            sm_vTotElec[warpId] = 0.;
        }
        __syncwarp(); // All threads in warp must wait for initialization

        // Perform warp-local accumulation in shared memory
        atomicAdd(sm_vTot + warpId, vtot_loc);
        atomicAdd(sm_vTotVdw + warpId, vtotVdw_loc);
        atomicAdd(sm_vTotElec + warpId, vtotElec_loc);

        __syncwarp(); // Ensure all threads in warp have completed
        if (threadIdx.x % warpSize == 0)
        { // One thread per warp accumulates partial sum into global sum
            atomicAdd(kernelParams.d_vTot + fType, sm_vTot[warpId]);
            atomicAdd(vtotVdw, sm_vTotVdw[warpId]);
            atomicAdd(vtotElec, sm_vTotElec[warpId]);
        }
    }
    /* Accumulate shift vectors from shared memory to global memory on the first c_numShiftVectors threads of the block. */
    if (calcVir)
    {
        __syncthreads();
        if (threadIdx.x < c_numShiftVectors)
        {
            atomicAdd(gm_fShift[threadIdx.x], sm_fShiftLoc[threadIdx.x]);
        }
    }
}


/*-------------------------------- End CUDA kernels-----------------------------*/


template<bool calcVir, bool calcEner>
void ListedForcesGpu::Impl::launchKernel()
{
    GMX_ASSERT(haveInteractions_,
               "Cannot launch bonded GPU kernels unless bonded GPU work was scheduled");

    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpu);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuBonded);

    int fTypeRangeEnd = kernelParams_.fTypeRangeEnd[numFTypesOnGpu - 1];

    if (fTypeRangeEnd < 0)
    {
        return;
    }

    auto kernelPtr = exec_kernel_gpu<calcVir, calcEner>;

    const auto kernelArgs = prepareGpuKernelArguments(
            kernelPtr, kernelLaunchConfig_, &kernelParams_, &d_xq_, &d_f_, &d_fShift_);

    launchGpuKernel(kernelPtr,
                    kernelLaunchConfig_,
                    deviceStream_,
                    nullptr,
                    "exec_kernel_gpu<calcVir, calcEner>",
                    kernelArgs);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpu);
}

void ListedForcesGpu::launchKernel(const gmx::StepWorkload& stepWork)
{
    if (stepWork.computeEnergy)
    {
        // When we need the energy, we also need the virial
        impl_->launchKernel<true, true>();
    }
    else if (stepWork.computeVirial)
    {
        impl_->launchKernel<true, false>();
    }
    else
    {
        impl_->launchKernel<false, false>();
    }
}

} // namespace gmx
