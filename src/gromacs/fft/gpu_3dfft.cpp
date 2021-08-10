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
 *  \brief Implements stub GPU 3D FFT routines for CPU-only builds
 *
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \ingroup module_fft
 */

#include "gmxpre.h"

#include "gpu_3dfft.h"

#include "gromacs/utility/exceptions.h"

namespace gmx
{

// [[noreturn]] attributes must be added in the common headers, so it's easier to silence the warning here
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#endif

class Gpu3dFft::Impl
{
};

Gpu3dFft::Gpu3dFft(ivec /*realGridSize*/,
                   ivec /*realGridSizePadded*/,
                   ivec /*complexGridSizePadded*/,
                   const bool /*useDecomposition*/,
                   const bool /*performOutOfPlaceFFT*/,
                   const DeviceContext& /*context*/,
                   const DeviceStream& /*pmeStream*/,
                   DeviceBuffer<float> /*realGrid*/,
                   DeviceBuffer<float> /*complexGrid*/)
{
    GMX_THROW(InternalError("Cannot run GPU routines in a CPU-only configuration"));
}

Gpu3dFft::~Gpu3dFft() = default;

// NOLINTNEXTLINE readability-convert-member-functions-to-static
void Gpu3dFft::perform3dFft(gmx_fft_direction /*dir*/, CommandEvent* /*timingEvent*/)
{
    GMX_THROW(InternalError("Cannot run GPU routines in a CPU-only configuration"));
}

#ifdef __clang__
#    pragma clang diagnostic pop
#endif

} // namespace gmx
