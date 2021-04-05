#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2021, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.

if(GMX_CP2K)
    set(CP2K_DIR "" CACHE STRING "Path to the directory with libcp2k.a library")
    set(CP2K_LIBS "" CACHE STRING "List of libraries for linking with CP2K. Typically this should be compination of LDFLAGS and LIBS variables from ARCH file used to compile CP2K")
    if ((CP2K_DIR STREQUAL "") OR (CP2K_LIBS STREQUAL ""))
        message(FATAL_ERROR "To build GROMACS with CP2K Interface both CP2K_DIR and CP2K_LIBS should be defined")
    endif()

    include_directories(SYSTEM "${CP2K_DIR}/../../../src/start")

    set(CMAKE_CXX_STANDARD_LIBRARIES " ${CMAKE_CXX_STANDARD_LIBRARIES} -Wl,--allow-multiple-definition -L${CP2K_DIR} -lcp2k -L${CP2K_DIR}/exts/dbcsr -ldbcsr ${CP2K_LIBS} -lgfortran")
    if (GMX_MPI)
        set(CMAKE_CXX_STANDARD_LIBRARIES " ${CMAKE_CXX_STANDARD_LIBRARIES} -lmpi_mpifh")
    endif()
endif()

