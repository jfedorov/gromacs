/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2010,2014,2015,2016,2018 by the GROMACS development team.
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
#ifndef GMX_MATH_DO_FIT_H
#define GMX_MATH_DO_FIT_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

namespace gmx
{
template<typename>
class ArrayRef;
} // namespace gmx

//! Returns RMSD or Rho (depending on \c bRho) over all atoms in \c index.
real calc_similar_ind(bool bRho, int nind, const int* index, const real mass[], const rvec x[], const rvec xp[]);

//! \copydoc calc_similar_ind
real calc_similar_ind(bool                           bRho,
                      gmx::ArrayRef<const int>       index,
                      gmx::ArrayRef<const real>      mass,
                      gmx::ArrayRef<const gmx::RVec> x,
                      gmx::ArrayRef<const gmx::RVec> xp);

//! Returns the RMS Deviation betweem \c x and \c xp over all atoms in \c index.
real rmsdev_ind(int nind, const int index[], const real mass[], const rvec x[], const rvec xp[]);

//! \copydoc rmsdev_ind
real rmsdev_ind(gmx::ArrayRef<const int>       index,
                gmx::ArrayRef<const real>      mass,
                gmx::ArrayRef<const gmx::RVec> x,
                gmx::ArrayRef<const gmx::RVec> xp);

//! Returns the RMS Deviation betweem \c x and \c xp over all atoms.
real rmsdev(int natoms, const real mass[], const rvec x[], const rvec xp[]);

//! \copydoc rmsdev
real rmsdev(gmx::ArrayRef<const real>      mass,
            gmx::ArrayRef<const gmx::RVec> x,
            gmx::ArrayRef<const gmx::RVec> xp);

/*! \brief
 * Returns size-independent Rho similarity parameter over all atoms in \c index.
 *
 * Maiorov & Crippen, PROTEINS 22, 273 (1995).
 */
real rhodev_ind(int nind, const int index[], const real mass[], const rvec x[], const rvec xp[]);

//! \copydoc rhodev_ind
real rhodev_ind(gmx::ArrayRef<const int>       index,
                gmx::ArrayRef<const real>      mass,
                gmx::ArrayRef<const gmx::RVec> x,
                gmx::ArrayRef<const gmx::RVec> xp);

/*! \brief
 * Returns size-independent Rho similarity parameter over all atoms.
 *
 * Maiorov & Crippen, PROTEINS 22, 273 (1995).
 */
real rhodev(int natoms, const real mass[], const rvec x[], const rvec xp[]);

//! \copydoc rhodev
real rhodev(gmx::ArrayRef<const real>      mass,
            gmx::ArrayRef<const gmx::RVec> x,
            gmx::ArrayRef<const gmx::RVec> xp);

/*! \brief
 * Calculates the rotation matrix R.
 *
 * Matrix will have minimal sum_i w_rls_i (xp_i - R x_i).(xp_i - R x_i).
 * \c ndim = 3 gives full fit, \c ndim = 2 gives xy fit.
 * This matrix is also used do_fit x_rotated[i] = sum R[i][j]*x[j].
 */
void calc_fit_R(int ndim, int natoms, const real* w_rls, const rvec* xp, rvec* x, matrix R);

//! \copydoc calc_fit_R
void calc_fit_R(int                            ndim,
                gmx::ArrayRef<const real>      w_rls,
                gmx::ArrayRef<const gmx::RVec> xp,
                gmx::ArrayRef<gmx::RVec>       x,
                matrix                         R);

/*! \brief
 * Do a least squares fit of \c x to \c xp.
 *
 * Atoms which have zero mass (w_rls[i]) are not taken into account in fitting.
 * This makes is possible to fit eg. on Calpha atoms and orient all atoms.
 * The routine only fits the rotational part, therefore both \c xp and \c x
 * should be centered round the origin.
 */
void do_fit_ndim(int ndim, int natoms, const real* w_rls, const rvec* xp, rvec* x);

//! \copydoc do_fit_ndim
void do_fit_ndim(int                            ndim,
                 gmx::ArrayRef<const real>      w_rls,
                 gmx::ArrayRef<const gmx::RVec> xp,
                 gmx::ArrayRef<gmx::RVec>       x);

//! Calls do_fit with fitting in 3D.
void do_fit(int natoms, const real* w_rls, const rvec* xp, rvec* x);

//! \copydoc do_fit
void do_fit(gmx::ArrayRef<const real> w_rls, gmx::ArrayRef<const gmx::RVec> xp, gmx::ArrayRef<gmx::RVec> x);

/*! \brief
 * Put the center of mass of atoms in the origin for dimensions 0 to \c ndim.
 *
 * The center of mass is computed from the index \c ind_cm.
 * When \c ind_cm!=NULL the COM is determined using \c ind_cm.
 * When \c ind_cm==NULL the COM is determined for atoms 0 to \c ncm.
 * When \c ind_reset!=NULL the coordinates indexed by \c ind_reset are reset.
 * When \c ind_reset==NULL the coordinates up to \c nreset are reset.
 */
void reset_x_ndim(int ndim, int ncm, const int* ind_cm, int nreset, const int* ind_reset, rvec x[], const real mass[]);

//! \copydoc reset_x_ndim
void reset_x_ndim(int                       ndim,
                  gmx::ArrayRef<const int>  ind_cm,
                  gmx::ArrayRef<const int>  ind_reset,
                  gmx::ArrayRef<gmx::RVec>  x,
                  gmx::ArrayRef<const real> mass);

//! Calls reset_x for resetting all dimensions.
void reset_x(int ncm, const int* ind_cm, int nreset, const int* ind_reset, rvec x[], const real mass[]);

//! \copydoc reset_x
void reset_x(gmx::ArrayRef<const int>  ind_cm,
             gmx::ArrayRef<const int>  ind_reset,
             gmx::ArrayRef<gmx::RVec>  x,
             gmx::ArrayRef<const real> mass);

#endif
