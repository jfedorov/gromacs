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
#include <vector>
#include "gmxpre.h"

#include "cluster_jarvis_patrick.h"

#include "gromacs/fileio/matio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/cmat.h"
#include "gromacs/linearalgebra/eigensolver.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/smalloc.h"

namespace gmx
{

namespace
{

bool rms_dist_comp(const PairDistances& a, const PairDistances& b)
{
    return a.distance < b.distance;
}

bool clust_id_comp(const ClusterIDs& a, const ClusterIDs& b)
{
    return a.cluster < b.cluster;
}

bool jp_same(const std::vector<std::vector<int>>& nnb, int i, int j, int P)
{
    bool bIn = false;

    for (const auto& k : nnb[i])
    {
        bIn = bIn || (k == j);
        if (!bIn)
        {
            return false;
        }
    }

    bIn = false;
    for (const auto& k : nnb[j])
    {
        bIn = bIn || (k == i);
        if (!bIn)
        {
            return false;
        }
    }

    int pp = 0;
    for (const auto& ii : nnb[i])
    {
        for (const auto& jj : nnb[j])
        {
            if (ii == jj)
            {
                pp++;
            }
        }
    }

    return (pp >= P);
}

} // namespace


ArrayRef<const int> ClusterJarvisPatrick::clusterList() const
{
    if (!finished_)
    {
        GMX_THROW(InternalError("Cannot access cluster list until we have actually clustered"));
    }
    return clusters_;
}

void ClusterJarvisPatrick::makeClusters()
{
    if (finished_)
    {
        GMX_THROW(InternalError("Should not cluster again after doing it once"));
    }

    int    cid, diff;
    bool   bChange;
    real** mcpy = nullptr;

    const real rmsdcut = rmsdCutOff_ < 0 ? 10000 : rmsdCutOff_;
    const int  num     = matrix_->nn;

    /* First we sort the entries in the RMSD matrix row by row.
     * This gives us the nearest neighbor list.
     */
    std::vector<std::vector<int>> nnb(num);
    std::vector<PairDistances>    row;
    row.reserve(num);
    for (int i = 0; (i < num); i++)
    {
        for (int j = 0; (j < num); j++)
        {
            row.emplace_back(PairDistances(i, j, matrix_->mat[i][j]));
        }
        std::sort(row.begin(), row.end(), rms_dist_comp);
        if (numNearestNeighbors_ > 0)
        {
            /* Put the M nearest neighbors in the list */
            nnb[i].reserve(numNearestNeighbors_ + 1);
            for (int j = 0; (j < num) && (matrix_->mat[i][row[j].j] < rmsdcut); j++)
            {
                if (row[j].j != i)
                {
                    nnb[i].emplace_back(row[j].j);
                }
            }
        }
        else
        {
            /* Put all neighbors nearer than rmsdcut in the list */
            for (int j = 0; (j < num) && (matrix_->mat[i][row[j].j] < rmsdcut); j++)
            {
                if (row[j].j != i)
                {
                    nnb[i].emplace_back(row[j].j);
                }
            }
        }
    }
    if (debug)
    {
        fprintf(debug, "Nearest neighborlist. M = %d, P = %d\n", numNearestNeighbors_, numIdenticalNeighbors_);
        for (int i = 0; (i < num); i++)
        {
            fprintf(debug, "i:%5d nbs:", i);
            for (const auto& j : nnb[i])
            {
                fprintf(debug, "%5d[%5.3f]", j, matrix_->mat[i][j]);
            }
            fprintf(debug, "\n");
        }
    }

    std::vector<ClusterIDs> clusters(num);
    fprintf(stderr, "Linking structures ");
    /* Use mcpy for temporary storage of booleans */
    mcpy = mk_matrix(num, num, FALSE);
    for (int i = 0; i < num; i++)
    {
        for (int j = i + 1; j < num; j++)
        {
            mcpy[i][j] = static_cast<real>(jp_same(nnb, i, j, numIdenticalNeighbors_));
        }
    }
    do
    {
        fprintf(stderr, "*");
        bChange = FALSE;
        for (int i = 0; i < num; i++)
        {
            for (int j = i + 1; j < num; j++)
            {
                if (mcpy[i][j] != 0.0F)
                {
                    diff = clusters[j].cluster - clusters[i].cluster;
                    if (diff)
                    {
                        bChange = true;
                        if (diff > 0)
                        {
                            clusters[j].cluster = clusters[i].cluster;
                        }
                        else
                        {
                            clusters[i].cluster = clusters[j].cluster;
                        }
                    }
                }
            }
        }
    } while (bChange);

    fprintf(stderr, "\nSorting and renumbering clusters\n");
    /* Sort on cluster number */
    std::sort(clusters.begin(), clusters.end(), clust_id_comp);

    /* Renumber clusters */
    cid = 1;
    for (int k = 1; k < num; k++)
    {
        if (clusters[k].cluster != clusters[k - 1].cluster)
        {
            clusters[k - 1].cluster = cid;
            cid++;
        }
        else
        {
            clusters[k - 1].cluster = cid;
        }
    }
    clusters[num - 1].cluster = cid;
    clusters_.resize(cid);
    for (int k = 0; k < num; k++)
    {
        clusters_[clusters[k].configuration] = clusters[k].cluster;
    }
    if (debug)
    {
        for (int k = 0; (k < num); k++)
        {
            fprintf(debug,
                    "Cluster index for conformation %d: %d\n",
                    clusters[k].configuration,
                    clusters[k].cluster);
        }
    }

    done_matrix(num, &mcpy);
    finished_ = true;
}

} // namespace gmx
