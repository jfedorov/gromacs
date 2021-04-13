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

/*! \file
 * \brief
 * Declares gmx::ICluster for clustering methods.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inlibraryapi
 * \ingroup module_gmxana
 */

#ifndef GMX_GMXANA_ICLUSTER_H
#define GMX_GMXANA_ICLUSTER_H

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

namespace gmx
{

struct PairDistances
{
    PairDistances(int indexI, int indexJ, real dist) : i(indexI), j(indexJ), distance(dist) {}
    //! First index.
    int i;
    //! Second index.
    int j;
    //! Distance between them.
    real distance;
};

struct ClusterIDs
{
    //! Index for configuration.
    int configuration = 0;
    //! Index for cluster.
    int cluster = 0;
};

/*! \brief
 * ICluster interface for different kinds of clustering methods.
 *
 * Methods that derive from this interface can be used to cluster different kinds of inputs.
 *
 * \inlibraryapi
 * \ingroup module_gmxana
 */
class ICluster
{
public:
    ICluster() {}
    virtual ~ICluster() {}
    /*! \brief
     * Access cluster list.
     */
    virtual ArrayRef<const int> clusterList() const = 0;
};

} // namespace gmx

#endif
