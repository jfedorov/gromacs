/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2014,2015,2018,2019,2021, by the GROMACS development team, led by
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
#ifndef GMX_EWALD_PME_SPLINE_WORK_H
#define GMX_EWALD_PME_SPLINE_WORK_H

#include "gromacs/simd/simd.h"
#include "gromacs/utility/alignedallocator.h"

#include "pme_simd.h"

//! Masks for filtering <=5 points from (aligned) loads of grid data of 8 elements
class pme_spline_work
{
public:
    pme_spline_work(int order);

#ifdef PME_SIMD4_SPREAD_GATHER
    //! Returns the SIMD mask for the first half of 8 entries for the given offset
    const gmx::Simd4Bool& mask_S0(int offset) const { return masks_[2 * offset]; }
    //! Returns the SIMD mask for the second half of 8 entries for the given offset
    const gmx::Simd4Bool& mask_S1(int offset) const { return masks_[2 * offset + 1]; }

private:
    //! The aligned storage for the masks
    std::vector<gmx::Simd4Bool, gmx::AlignedAllocator<gmx::Simd4Bool>> masks_;
#endif
};

#endif
