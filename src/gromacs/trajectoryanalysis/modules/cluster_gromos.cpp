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

#include "cluster_gromos.h"

#include "gromacs/fileio/matio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/cmat.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/listoflists.h"

namespace gmx
{

namespace
{

bool nrnb_comp(ArrayRef<const int> a, ArrayRef<const int> b)
{
    /* return b<a, we want highest first */
    return b.size() < a.size();
}

void dump_nnb(FILE* fp, const char* title, const std::vector<std::vector<int>>& nnb)
{
    /* dump neighbor list */
    fprintf(fp, "%s", title);
    for (int i = 0; i < gmx::ssize(nnb); i++)
    {
        fprintf(fp, "i:%5d #:%5ld nbs:", i, nnb[i].size());
        for (int j = 0; j < gmx::ssize(nnb[i]); j++)
        {
            fprintf(fp, "%5d", nnb[i][j]);
        }
        fprintf(fp, "\n");
    }
}

} // namespace

ArrayRef<const int> ClusterGromos::clusterList() const
{
    if (!finished_)
    {
        GMX_THROW(InternalError("Cannot access cluster list until we have actually clustered"));
    }
    return clusters_;
}

void ClusterGromos::makeClusters()
{
    if (finished_)
    {
        GMX_THROW(InternalError("Should not cluster again after doing it once"));
    }
    std::vector<std::vector<int>> nnb;
    const int                     matrixSize = matrix_->nn;

    /* Put all neighbors nearer than rmsdcut in the list */
    GMX_LOG(logger_.info).asParagraph().appendText("Making list of neighbors within cutoff");
    for (int i = 0; (i < matrixSize); i++)
    {
        std::vector<int> list;
        /* put all neighbors within cut-off in list */
        for (int j = 0; j < matrixSize; j++)
        {
            if (matrix_->mat[i][j] < rmsdCutOff_)
            {
                list.emplace_back(j);
            }
        }
        nnb.push_back(list);
        if (i % (1 + matrixSize / 100) == 0)
        {
            GMX_LOG(logger_.info).appendTextFormatted("Progress %3d%%\b\b\b\b", (i * 100 + 1) / matrixSize);
        }
    }
    GMX_LOG(logger_.info).appendTextFormatted("Progress %3d%%\n", 100);

    /* sort neighbor list on number of neighbors, largest first */
    std::sort(nnb.begin(), nnb.end(), nrnb_comp);

    if (debug)
    {
        dump_nnb(debug, "Nearest neighborlist after sort.\n", nnb);
    }

    /* turn first structure with all its neighbors (largest) into cluster
       remove them from pool of structures and repeat for all remaining */
    GMX_LOG(logger_.info).appendTextFormatted("Finding clusters %4d", 0);
    /* cluster id's start at 1: */
    int k = 1;
    clusters_.resize(numInputs_);
    while (!nnb.empty() && !nnb[0].empty())
    {
        /* set cluster id (k) for first item in neighborlist */
        for (int j = 0; j < gmx::ssize(nnb[0]); j++)
        {
            clusters_[nnb[0][j]] = k;
        }
        /* mark as done */
        nnb.erase(nnb.begin());

        /* adjust number of neighbors for others, taking removals into account: */
        for (int i = 1; i < matrixSize && !nnb[i].empty(); i++)
        {
            std::vector<int> neighbors;
            for (int j = 0; j < gmx::ssize(nnb[i]); j++)
            {
                /* if this neighbor wasn't removed */
                if (clusters_[nnb[i][j]] == 0)
                {
                    neighbors.emplace_back(nnb[i][j]);
                }
            }
            nnb[i] = neighbors;
            /* now j1 is the new number of neighbors */
        }

        /* sort again on nnb[].nr, because we have new # neighbors: */
        /* but we only need to sort upto i, i.e. when nnb[].nr>0 */
        std::sort(nnb.begin(), nnb.end(), nrnb_comp);

        GMX_LOG(logger_.info).appendTextFormatted("\b\b\b\b%4d", k);
        /* new cluster id */
        k++;
    }
    GMX_LOG(logger_.info).appendText("\n");
    if (debug)
    {
        fprintf(debug, "Clusters (%d):\n", k);
        for (int i = 0; i < matrixSize; i++)
        {
            fprintf(debug, " %3d", clusters_[i]);
        }
        fprintf(debug, "\n");
    }
    finished_ = true;
}

} // namespace gmx
