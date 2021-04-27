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

#include <algorithm>
#include <numeric>
#include <vector>

#include "gmxpre.h"

#include "cluster_linkage.h"

#include "gromacs/gmxana/cmat.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/logger.h"

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
} // namespace

ArrayRef<const int> ClusterLinkage::clusterList() const
{
    if (!finished_)
    {
        GMX_THROW(InternalError("Cannot access cluster list until we have actually clustered"));
    }
    return clusters_;
}

void ClusterLinkage::makeClusters()
{
    if (finished_)
    {
        GMX_THROW(InternalError("Should not cluster again after doing it once"));
    }
    bool bChange = false;

    /* First we sort the entries in the RMSD matrix */
    const int                  n1 = matrix_->nn;
    const int                  nn = ((n1 - 1) * n1) / 2;
    std::vector<PairDistances> distances;
    distances.reserve(nn);
    for (int i = 0; (i < n1); i++)
    {
        for (int j = i + 1; (j < n1); j++)
        {
            distances.emplace_back(PairDistances(i, j, matrix_->mat[i][j]));
        }
    }
    if (gmx::ssize(distances) != nn)
    {
        GMX_THROW(InternalError("Need a square matrix for linkage clustering"));
    }
    std::sort(distances.begin(), distances.end(), rms_dist_comp);

    /* Now we make a cluster index for all of the conformations */
    std::vector<ClusterIDs> clusters(n1);
    for (int i = 0; i < gmx::ssize(clusters); i++)
    {
        clusters[i].cluster       = i;
        clusters[i].configuration = i;
    }

    /* Now we check the closest structures, and equalize their cluster numbers */
    GMX_LOG(logger_.info).asParagraph().appendText("Linking structures");
    do
    {
        bChange = false;
        for (int k = 0; (k < nn) && (distances[k].distance < rmsdCutOff_); k++)
        {
            int diff = clusters[distances[k].j].cluster - clusters[distances[k].i].cluster;
            if (diff)
            {
                bChange = true;
                if (diff > 0)
                {
                    clusters[distances[k].j].cluster = clusters[distances[k].i].cluster;
                }
                else
                {
                    clusters[distances[k].i].cluster = clusters[distances[k].j].cluster;
                }
            }
        }
    } while (bChange);
    GMX_LOG(logger_.info).asParagraph().appendText("Sorting and renumbering clusters");
    /* Sort on cluster number */
    std::sort(clusters.begin(), clusters.end(), clust_id_comp);

    /* Renumber clusters */
    int cid = 1;
    int k   = 1;
    for (; k < n1; k++)
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
    if (!clusters.empty())
    {
        clusters[k - 1].cluster = cid;
    }
    if (debug)
    {
        for (int k = 0; (k < n1); k++)
        {
            fprintf(debug,
                    "Cluster index for conformation %d: %d\n",
                    clusters[k].configuration,
                    clusters[k].cluster);
        }
    }
    for (int k = 0; k < n1; k++)
    {
        clusters_[clusters[k].configuration] = clusters[k].cluster;
    }
    finished_ = true;
}

} // namespace gmx
