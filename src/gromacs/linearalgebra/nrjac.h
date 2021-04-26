/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2010,2014,2017,2018,2019 by the GROMACS development team.
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
#ifndef GMX_LINEARALGEBRA_NRJAC_H
#define GMX_LINEARALGEBRA_NRJAC_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

/* Diagonalizes a symmetric matrix
 *
 * \param[in,out] a           Input matrix a[0..n-1][0..n-1] must be symmetric, gets modified
 * \param[in]  numDimensions  Number of rows and columns
 * \param[out] eigenvalues    eigenvalues[0]..eigenvalues[n-1] are the eigenvalues of a
 * \param[out] eigenvectors   v[0..n-1][0..n-1] the eigenvectors: v[i][j] is component i of vector j
 * \param[out] numRotations   The number of jacobi rotations, can be nullptr
 */
void jacobi(double** a, int numDimensions, double* eigenvalues, double** eigenvectors, int* numRotations);

/* Like jacobi above, but specialized for n=3
 *
 * \param[in,out] a  The symmetric matrix to diagonalize, size 3, note that the contents gets modified
 * \param[out] eigenvalues  The eigenvalues, size 3
 * \param[out] eigenvectors The eigenvectors, size 3

 * Returns the number of jacobi rotations.
 */
int jacobi(gmx::ArrayRef<gmx::DVec> a, gmx::ArrayRef<double> eigenvalues, gmx::ArrayRef<gmx::DVec> eigenvectors);

int m_inv_gen(real* m, int n, real* minv);
/* Produces minv, a generalized inverse of m, both stored as linear arrays.
 * Inversion is done via diagonalization,
 * eigenvalues smaller than 1e-6 times the average diagonal element
 * are assumed to be zero.
 * For zero eigenvalues 1/eigenvalue is set to zero for the inverse matrix.
 * Returns the number of zero eigenvalues.
 */

#endif
