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
/*! \internal \file
 * \brief
 * A collection of helper utilities that allow setting up both Nblib and
 * GROMACS fixtures for computing listed interactions given sets of parameters
 * and coordinates
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */

#ifndef NBLIB_LISTEDFORCES_LISTEDTESTHELPERS_H
#define NBLIB_LISTEDFORCES_LISTEDTESTHELPERS_H

#include "nblib/listed_forces/definitions.h"

namespace nblib
{
class Box;

//! \brief Creates a default vector of indices for two-centered interactions
template<class Interaction, std::enable_if_t<Contains<Interaction, SupportedTwoCenterTypes>{}>* = nullptr>
std::vector<InteractionIndex<Interaction>> indexVector()
{
    return { { 0, 1, 0 } };
}

//! \brief Creates a default vector of indices for three-centered interactions
template<class Interaction, std::enable_if_t<Contains<Interaction, SupportedThreeCenterTypes>{}>* = nullptr>
std::vector<InteractionIndex<Interaction>> indexVector()
{
    return { { 0, 1, 2, 0 } };
}

//! \brief Creates a default vector of indices for four-centered interactions
template<class Interaction, std::enable_if_t<Contains<Interaction, SupportedFourCenterTypes>{}>* = nullptr>
std::vector<InteractionIndex<Interaction>> indexVector()
{
    return { { 0, 1, 2, 3, 0 } };
}

} // namespace nblib

#endif // NBLIB_LISTEDFORCES_LISTEDTESTHELPERS_H
