/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2015,2017,2018,2019,2021, by the GROMACS development team, led by
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
 * Implements AlignedAllocator.
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "alignedallocator.h"

#include "config.h"

#include <cstdlib>

#ifdef HAVE_UNISTD_H
#    include <unistd.h>
#endif

#if GMX_NATIVE_WINDOWS
#    include <windows.h> // only for the page size query purposes
#endif

namespace gmx
{

// === AlignedAllocationPolicy

std::size_t AlignedAllocationPolicy::alignment()
{
    // For now we always use 128-byte alignment:
    // 1) IBM Power already has cache lines of 128-bytes, and needs it.
    // 2) x86 has 64 byte cache lines, but since a future AVX-1024 (rumored?)
    //    will need 1024/8=128 byte SIMD alignment, it is safer to use that
    //    already now.
    // 3) The old Pentium4 used 256-byte cache prefetching (but 64-byte lines).
    //    However, it's not worth worrying about performance for P4...
    // 4) ARM & Sparc have 64 byte lines, but will be just fine with
    //    128-byte alignment (nobody knows what the future brings)
    //
    // So, for now we're semi-lazy and just align to 128 bytes!
    //
    // TODO LINCS code is copying this assumption independently (for now)
    return 128;
}

void* AlignedAllocationPolicy::malloc(std::size_t bytes)
{
    // Adhere to the implementation requirements. Also avoids false
    // sharing.
    auto multiplesOfAlignment = (bytes / alignment() + 1) * alignment();
    return aligned_alloc(alignment(), multiplesOfAlignment);
}

void AlignedAllocationPolicy::free(void* p)
{
    std::free(p);
}

// === PageAlignedAllocationPolicy

//! Return a page size, from a sysconf/WinAPI query if available, or a default guess (4096 bytes).
//! \todo Move this function into sysinfo.cpp where other OS-specific code/includes live
static std::size_t getPageSize()
{
    long pageSize = 0;
#if GMX_NATIVE_WINDOWS
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);
    pageSize = si.dwPageSize;
#elif defined(_SC_PAGESIZE)
    /* Note that sysconf returns -1 on its error conditions, which we
       don't really need to check, nor can really handle at
       initialization time. */
    pageSize = sysconf(_SC_PAGESIZE);
#elif defined(_SC_PAGE_SIZE)
    pageSize = sysconf(_SC_PAGE_SIZE);
#else
    pageSize = -1;
#endif
    return ((pageSize == -1) ? 4096 // A useful guess
                             : static_cast<std::size_t>(pageSize));
}

/* Implements the "construct on first use" idiom to avoid the static
 * initialization order fiasco where a possible static page-aligned
 * container would be initialized before the alignment variable was.
 *
 * Note that thread-safety of the initialization is guaranteed by the
 * C++11 language standard.
 *
 * The size_t has no destructor, so there is no deinitialization
 * issue.  See https://isocpp.org/wiki/faq/ctors for discussion of
 * alternatives and trade-offs. */
std::size_t PageAlignedAllocationPolicy::alignment()
{
    static size_t thePageSize = getPageSize();
    return thePageSize;
}

void* PageAlignedAllocationPolicy::malloc(std::size_t bytes)
{
    // Adhere to the implementation requirements. Also avoids false
    // sharing.
    auto multiplesOfAlignment = (bytes / alignment() + 1) * alignment();
    return aligned_alloc(alignment(), multiplesOfAlignment);
}

void PageAlignedAllocationPolicy::free(void* p)
{
    std::free(p);
}

} // namespace gmx
