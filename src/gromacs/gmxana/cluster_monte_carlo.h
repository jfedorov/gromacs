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

#ifndef GMX_GMXANA_CLUSTER_MONTE_CARLO_H
#define GMX_GMXANA_CLUSTER_MONTE_CARLO_H

#include <stdio.h>
#include <vector>

#include "gromacs/utility/arrayref.h"
#include "icluster.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct t_clusters;
struct t_mat;

namespace gmx
{

class MDLogger;

class ClusterMonteCarlo : public ICluster
{
public:
    explicit ClusterMonteCarlo(t_mat*               inputMatrix,
                               real                 rmsdCutOff,
                               real                 kT,
                               int                  numInputs,
                               int                  seed,
                               int                  maxIerations,
                               int                  randomIterations,
                               const MDLogger&      logger,
                               ArrayRef<const real> time) :
        finished_(false),
        rmsdCutOff_(rmsdCutOff),
        kT_(kT),
        numInputs_(numInputs),
        seed_(seed),
        maxIterations_(maxIerations),
        randomIterations_(randomIterations),
        matrix_(inputMatrix),
        logger_(logger),
        time_(time)
    {
        makeClusters();
    }
    ~ClusterMonteCarlo() override;

    ArrayRef<const int> clusterList() const override;

private:
    //! Perform actual clustering.
    void makeClusters();
    //! Did we perform the clustering?
    bool finished_;
    //! Value for RMSD cutoff.
    const real rmsdCutOff_;
    //! Boltzmann weigthing.
    const real kT_;
    //! Number of inputs.
    const int numInputs_;
    //! Random seed for monte carlo.
    const int seed_;
    //! Maximum number of MC steps.
    const int maxIterations_;
    //! Number of fully random steps.
    const int randomIterations_;
    //! Handle to cluster matrix.
    t_mat* matrix_;
    //! Cluster indices
    std::vector<int> clusters_;
    //! Logger handle
    const MDLogger& logger_;
    //! Time points for frames.
    ArrayRef<const real> time_;
};

} // namespace gmx

#endif
