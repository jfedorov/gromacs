/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2008, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2017,2018 by the GROMACS development team.
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
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
#ifndef GMX_TIMING_WALLCYCLE_H
#define GMX_TIMING_WALLCYCLE_H

/* NOTE: None of the routines here are safe to call within an OpenMP
 * region */

#include <stdio.h>

#include "gromacs/utility/basedefinitions.h"

typedef struct gmx_wallcycle* gmx_wallcycle_t;
struct t_commrec;

enum class WallCycleCounter : int
{
    RUN,
    STEP,
    PPDURINGPME,
    DOMDEC,
    DDCOMMLOAD,
    DDCOMMBOUND,
    VSITECONSTR,
    PP_PMESENDX,
    NS,
    LAUNCH_GPU,
    MOVEX,
    FORCE,
    MOVEF,
    PMEMESH,
    PME_REDISTXF,
    PME_SPREAD,
    PME_GATHER,
    PME_FFT,
    PME_FFTCOMM,
    LJPME,
    PME_SOLVE,
    PMEWAITCOMM,
    PP_PMEWAITRECVF,
    WAIT_GPU_PME_SPREAD,
    PME_FFT_MIXED_MODE,
    PME_SOLVE_MIXED_MODE,
    WAIT_GPU_PME_GATHER,
    WAIT_GPU_BONDED,
    PME_GPU_F_REDUCTION,
    WAIT_GPU_NB_NL,
    WAIT_GPU_NB_L,
    WAIT_GPU_STATE_PROPAGATOR_DATA,
    NB_XF_BUF_OPS,
    VSITESPREAD,
    PULLPOT,
    AWH,
    TRAJ,
    UPDATE,
    CONSTR,
    MoveE,
    ROT,
    ROTadd,
    SWAP,
    IMD,
    TEST,
    Count
};

enum class WallCycleSubCounter : int
{
    DD_REDIST,
    DD_GRID,
    DD_SETUPCOMM,
    DD_MAKETOP,
    DD_MAKECONSTR,
    DD_TOPOTHER,
    DD_GPU,
    NBS_GRID_LOCAL,
    NBS_GRID_NONLOCAL,
    NBS_SEARCH_LOCAL,
    NBS_SEARCH_NONLOCAL,
    LISTED,
    LISTED_FEP,
    RESTRAINTS,
    LISTED_BUF_OPS,
    NONBONDED_PRUNING,
    NONBONDED_KERNEL,
    NONBONDED_CLEAR,
    NONBONDED_FEP,
    LAUNCH_GPU_NONBONDED,
    LAUNCH_GPU_BONDED,
    LAUNCH_GPU_PME,
    LAUNCH_STATE_PROPAGATOR_DATA,
    EWALD_CORRECTION,
    NB_X_BUF_OPS,
    NB_F_BUF_OPS,
    CLEAR_FORCE_BUFFER,
    LAUNCH_GPU_NB_X_BUF_OPS,
    LAUNCH_GPU_NB_F_BUF_OPS,
    LAUNCH_GPU_MOVEX,
    LAUNCH_GPU_MOVEF,
    LAUNCH_GPU_UPDATE_CONSTRAIN,
    TEST,
    Count
};

static constexpr const int sc_numWallCycleCounters    = static_cast<int>(WallCycleCounter::Count);
static constexpr const int sc_numWallCycleSubCounters = static_cast<int>(WallCycleSubCounter::Count);
static constexpr const int sc_numWallCycleCountersSquared =
        sc_numWallCycleCounters * sc_numWallCycleCounters;

bool wallcycle_have_counter();
/* Returns if cycle counting is supported */

gmx_wallcycle_t wallcycle_init(FILE* fplog, int resetstep, struct t_commrec* cr);
/* Returns the wall cycle structure.
 * Returns NULL when cycle counting is not supported.
 */

/* cleans up wallcycle structure */
void wallcycle_destroy(gmx_wallcycle_t wc);

void wallcycle_start(gmx_wallcycle* wc, WallCycleCounter ewc);
/* Starts the cycle counter (and increases the call count) */

void wallcycle_start_nocount(gmx_wallcycle* wc, WallCycleCounter ewc);
/* Starts the cycle counter without increasing the call count */

double wallcycle_stop(gmx_wallcycle* wc, WallCycleCounter ewc);
/* Stop the cycle count for ewc , returns the last cycle count */

void wallcycle_increment_event_count(gmx_wallcycle* wc, WallCycleCounter ewc);
/* Only increment call count for ewc  by one */

void wallcycle_get(gmx_wallcycle* wc, WallCycleCounter ewc, int* n, double* c);
/* Returns the cumulative count and cycle count for ewc  */

void wallcycle_reset_all(gmx_wallcycle* wc);
/* Resets all cycle counters to zero */

void wallcycle_scale_by_num_threads(gmx_wallcycle* wc, bool isPmeRank, int nthreads_pp, int nthreads_pme);
/* Scale the cycle counts to reflect how many threads run for that number of cycles */

int64_t wcycle_get_reset_counters(gmx_wallcycle* wc);
/* Return reset_counters from wc struct */

void wcycle_set_reset_counters(gmx_wallcycle* wc, int64_t reset_counters);
/* Set reset_counters */

void wallcycle_sub_start(gmx_wallcycle* wc, WallCycleSubCounter ewcs);
/* Set the start sub cycle count for ewcs  */

void wallcycle_sub_start_nocount(gmx_wallcycle* wc, WallCycleSubCounter ewcs);
/* Set the start sub cycle count for ewcs  without increasing the call count */

void wallcycle_sub_stop(gmx_wallcycle* wc, WallCycleSubCounter ewcs);
/* Stop the sub cycle count for ewcs */

#endif
