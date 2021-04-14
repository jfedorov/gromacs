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
#include "gmxpre.h"

#include "cluster_monte_carlo.h"

#include "gromacs/fileio/matio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/cmat.h"
#include "gromacs/linearalgebra/eigensolver.h"
#include "gromacs/random/threefry.h"
#include "gromacs/random/uniformintdistribution.h"
#include "gromacs/random/uniformrealdistribution.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/smalloc.h"

namespace gmx
{

ArrayRef<const int> ClusterMonteCarlo::clusterList() const
{
    if (!finished_)
    {
        GMX_THROW(InternalError("Cannot access cluster list until we have actually clustered"));
    }
    return clusters_;
}


void ClusterMonteCarlo::makeClusters()
{
    real   ecur, enext, emin, prob;
    int    nuphill = 0;
    t_mat* minimum;

    const int seed = (seed_ == -1) ? static_cast<int>(gmx::makeRandomSeed()) : seed_;

    gmx::DefaultRandomEngine rng(seed);

    if (matrix_->n1 != matrix_->nn)
    {
        GMX_THROW(
                InvalidInputError("Can not do Monte Carlo optimization with a non-square matrix."));
    }
    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted(
                    "Doing Monte Carlo optimization to find the smoothest trajectory\n"
                    "by reordering the frames to minimize the path between the two structures\n"
                    "that have the largest pairwise RMSD.\n"
                    "Using random seed %d.\n",
                    seed_);

    int  iswap = -1;
    int  jswap = -1;
    real enorm = matrix_->mat[0][0];
    for (int i = 0; (i < matrix_->n1); i++)
    {
        for (int j = 0; (j < matrix_->nn); j++)
        {
            if (matrix_->mat[i][j] > enorm)
            {
                enorm = matrix_->mat[i][j];
                iswap = i;
                jswap = j;
            }
        }
    }
    if ((iswap == -1) || (jswap == -1))
    {
        GMX_THROW(InvalidInputError("Matrix contains identical values in all fields\n"));
    }
    swap_rows(matrix_, 0, iswap);
    swap_rows(matrix_, matrix_->n1 - 1, jswap);
    emin = ecur = mat_energy(matrix_);
    GMX_LOG(logger_.info)
            .appendTextFormatted(
                    "Largest distance %g between %d and %d. Energy: %g.\n", enorm, iswap, jswap, emin);

    int nn = matrix_->nn;

    /* Initiate and store global minimum */
    minimum     = init_mat(nn, matrix_->b1D);
    minimum->nn = nn;
    copy_t_mat(minimum, matrix_);

    gmx::UniformIntDistribution<int>   intDistNN(1, nn - 2); // [1,nn-2]
    gmx::UniformRealDistribution<real> realDistOne;          // [0,1)

    for (int i = 0; (i < maxIterations_); i++)
    {
        /* Generate new swapping candidates */
        do
        {
            iswap = intDistNN(rng);
            jswap = intDistNN(rng);
        } while ((iswap == jswap) || (iswap >= nn - 1) || (jswap >= nn - 1));

        /* Apply swap and compute energy */
        swap_rows(matrix_, iswap, jswap);
        enext = mat_energy(matrix_);

        /* Compute probability */
        prob = 0;
        if ((enext < ecur) || (i < randomIterations_))
        {
            prob = 1;
            if (enext < emin)
            {
                /* Store global minimum */
                copy_t_mat(minimum, matrix_);
                emin = enext;
            }
        }
        else if (kT_ > 0)
        {
            /* Try Monte Carlo step */
            prob = std::exp(-(enext - ecur) / (enorm * kT_));
        }

        if (prob == 1 || realDistOne(rng) < prob)
        {
            if (enext > ecur)
            {
                nuphill++;
            }

            GMX_LOG(logger_.info)
                    .appendTextFormatted(
                            "Iter: %d Swapped %4d and %4d (energy: %g prob: %g)\n", i, iswap, jswap, enext, prob);
            ecur = enext;
        }
        else
        {
            swap_rows(matrix_, jswap, iswap);
        }
    }
    GMX_LOG(logger_.info).appendTextFormatted("%d uphill steps were taken during optimization\n", nuphill);

    /* Now swap the matrix to get it into global minimum mode */
    copy_t_mat(matrix_, minimum);

    GMX_LOG(logger_.info).appendTextFormatted("Global minimum energy %g\n", mat_energy(minimum));
    GMX_LOG(logger_.info).appendTextFormatted("Global minimum energy %g\n", mat_energy(matrix_));
    GMX_LOG(logger_.info)
            .appendTextFormatted("Swapped time and frame indices and RMSD to next neighbor:\n");
    for (int i = 0; (i < matrix_->nn); i++)
    {
        GMX_LOG(logger_.info)
                .appendTextFormatted("%10g  %5d  %10g\n",
                                     time_[matrix_->m_ind[i]],
                                     matrix_->m_ind[i],
                                     (i < matrix_->nn - 1)
                                             ? matrix_->mat[matrix_->m_ind[i]][matrix_->m_ind[i + 1]]
                                             : 0);
    }
}

} // namespace gmx
