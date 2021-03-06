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

#include <array>
#include <memory>
#include <vector>

#include "gromacs/timing/cyclecounter.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/enumerationhelpers.h"


#ifdef ITT_INSTRUMENT 

#include <assert.h>
#include <string>
#include <unordered_map>
#include <ittnotify.h>

#endif 

#ifndef DEBUG_WCYCLE
/*! \brief Enables consistency checking for the counters.
 *
 * If the macro is set to 1, code checks if you stop a counter different from the last
 * one that was opened and if you do nest too deep.
 */
#    define DEBUG_WCYCLE 0
#endif

struct t_commrec;

#ifndef DEBUG_WCYCLE
/*! \brief Enables consistency checking for the counters.
 *
 * If the macro is set to 1, code checks if you stop a counter different from the last
 * one that was opened and if you do nest too deep.
 */
#    define DEBUG_WCYCLE 0
#endif

enum class WallCycleCounter : int
{
    Run,
    Step,
    PpDuringPme,
    Domdec,
    DDCommLoad,
    DDCommBound,
    VsiteConstr,
    PpPmeSendX,
    NS,
    LaunchGpu,
    MoveX,
    Force,
    MoveF,
    PmeMesh,
    PmeRedistXF,
    PmeSpread,
    PmeGather,
    PmeFft,
    PmeFftComm,
    LJPme,
    PmeSolve,
    PmeWaitComm,
    PpPmeWaitRecvF,
    WaitGpuPmeSpread,
    PmeFftMixedMode,
    PmeSolveMixedMode,
    WaitGpuPmeGather,
    WaitGpuBonded,
    PmeGpuFReduction,
    WaitGpuNbNL,
    WaitGpuNbL,
    WaitGpuStatePropagatorData,
    NbXFBufOps,
    VsiteSpread,
    PullPot,
    Awh,
    Traj,
    Update,
    Constr,
    MoveE,
    Rot,
    RotAdd,
    Swap,
    Imd,
    Test,
    Count
};

enum class WallCycleSubCounter : int
{
    DDRedist,
    DDGrid,
    DDSetupComm,
    DDMakeTop,
    DDMakeConstr,
    DDTopOther,
    DDGpu,
    NBSGridLocal,
    NBSGridNonLocal,
    NBSSearchLocal,
    NBSSearchNonLocal,
    Listed,
    ListedFep,
    Restraints,
    ListedBufOps,
    NonbondedPruning,
    NonbondedKernel,
    NonbondedClear,
    NonbondedFep,
    NonbondedFepReduction,
    LaunchGpuNonBonded,
    LaunchGpuBonded,
    LaunchGpuPme,
    LaunchStatePropagatorData,
    EwaldCorrection,
    NBXBufOps,
    NBFBufOps,
    ClearForceBuffer,
    LaunchGpuNBXBufOps,
    LaunchGpuNBFBufOps,
    LaunchGpuMoveX,
    LaunchGpuMoveF,
    LaunchGpuUpdateConstrain,
    Test,
    Count
};

static constexpr int sc_numWallCycleCounters        = static_cast<int>(WallCycleCounter::Count);
static constexpr int sc_numWallCycleSubCounters     = static_cast<int>(WallCycleSubCounter::Count);
static constexpr int sc_numWallCycleCountersSquared = sc_numWallCycleCounters * sc_numWallCycleCounters;
static constexpr bool sc_useCycleSubcounters        = GMX_CYCLE_SUBCOUNTERS;

struct wallcc_t
{
    int          n;
    gmx_cycles_t c;
    gmx_cycles_t start;
};

#if DEBUG_WCYCLE
static constexpr int c_MaxWallCycleDepth = 6;
#endif


struct gmx_wallcycle
{
    gmx::EnumerationArray<WallCycleCounter, wallcc_t> wcc;
    /* did we detect one or more invalid cycle counts */
    bool haveInvalidCount;
    /* variables for testing/debugging */
    bool                  wc_barrier;
    std::vector<wallcc_t> wcc_all;
    int                   wc_depth;
#if DEBUG_WCYCLE
    std::array<WallCycleCounter, c_MaxWallCycleDepth> counterlist;
    int                                               count_depth;
    bool                                              isMasterRank;
#endif
    WallCycleCounter                                     ewc_prev;
    gmx_cycles_t                                         cycle_prev;
    int64_t                                              reset_counters;
    const t_commrec*                                     cr;
    gmx::EnumerationArray<WallCycleSubCounter, wallcc_t> wcsc;
#ifdef ITT_INSTRUMENT
     __itt_pt_region pt_region;
     bool            in_pt_region;
    std::unordered_map<std::string, __itt_pt_region> pt_region_map;
    
    unsigned long                                    invoke_idx;

    __itt_domain*                                    task_domain;
    __itt_string_handle*                             task_string_handle;

    std::unordered_map<std::string, __itt_string_handle*> task_map;

#endif    
};

//! Returns if cycle counting is supported
bool wallcycle_have_counter();

//! Returns the wall cycle structure.
std::unique_ptr<gmx_wallcycle> wallcycle_init(FILE* fplog, int resetstep, const t_commrec* cr);

//! Adds custom barrier for wallcycle counting.
void wallcycleBarrier(gmx_wallcycle* wc);

void wallcycle_sub_get(gmx_wallcycle* wc, WallCycleSubCounter ewcs, int* n, double* c);
/* Returns the cumulative count and sub cycle count for ewcs */

inline void wallcycle_all_start(gmx_wallcycle* wc, WallCycleCounter ewc, gmx_cycles_t cycle)
{
    wc->ewc_prev   = ewc;
    wc->cycle_prev = cycle;
}

inline void wallcycle_all_stop(gmx_wallcycle* wc, WallCycleCounter ewc, gmx_cycles_t cycle)
{
    const int prev    = static_cast<int>(wc->ewc_prev);
    const int current = static_cast<int>(ewc);
    wc->wcc_all[prev * sc_numWallCycleCounters + current].n += 1;
    wc->wcc_all[prev * sc_numWallCycleCounters + current].c += cycle - wc->cycle_prev;
}

#ifdef ITT_INSTRUMENT

//#define ITT_GENERAL

const static std::string s_lookup_pattern("1652");

#ifndef ITT_START_FRAME
#define ITT_START_FRAME       (102+(4000*5))
//#define ITT_START_FRAME       (102)
#define ITT_MAX_PT_REGION_IDS 16
#endif 


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define wallcycle_start(p0, p1)          _wallcycle_start(p0, p1,  __FILE__ ":" TOSTRING(__LINE__))
#define wallcycle_start_nocount(p0, p1)  _wallcycle_start_nocount(p0, p1,  __FILE__ ":" TOSTRING(__LINE__))

static inline void get_pt_region(gmx_wallcycle* wc, const char* location)
{
    //printf("--------------- at: %s, from: %s\n", __FUNCTION__, location);
    std::string key(location);
    std::string key_generic("LaunchGpu Other");

    auto search = wc->pt_region_map.find(key);
    if (search == wc->pt_region_map.end() ) { 
        if ( wc->pt_region_map.size() < (ITT_MAX_PT_REGION_IDS-1) ) {
#ifdef ITT_INSTRUMENT_DEBUG
            printf("=========> Making Pt_region for location: %s, size: %lu\n", 
                    location, wc->pt_region_map.size());
#endif            
            wc->pt_region = __itt_pt_region_create(key.c_str());
            wc->pt_region_map.insert({key, wc->pt_region});
        } else if ( wc->pt_region_map.size() == ITT_MAX_PT_REGION_IDS  ) {
#ifdef ITT_INSTRUMENT_DEBUG
            printf("=========> Making Pt_region for generic location: %s, size: %lu, real location: %s\n", 
                    key_generic.c_str(), wc->pt_region_map.size(), location);
#endif            
            wc->pt_region = __itt_pt_region_create(key_generic.c_str());
            wc->pt_region_map.insert({key_generic, wc->pt_region});
        } else {
#ifdef ITT_INSTRUMENT_DEBUG
            printf("=========> Taking Pt_region of generic location: %s, size: %lu, real location: %s\n", 
                    key_generic.c_str(), wc->pt_region_map.size(), location);
#endif            
            auto _search = wc->pt_region_map.find(key_generic);
            assert( _search != wc->pt_region_map.end() );
            wc->pt_region = _search->second;
        } 
    } else {
#ifdef ITT_INSTRUMENT_DEBUG
        printf("=========> Found Pt_region for location: %s, size: %lu\n", 
            location, wc->pt_region_map.size());
#endif            
        wc->pt_region = search->second;
    }
}

static inline void get_task_string_handle(gmx_wallcycle* wc, const char* location)
{
    std::string key(location);

    auto search = wc->task_map.find(key);
    if (search == wc->task_map.end() ) { 
#ifdef ITT_INSTRUMENT_DEBUG
        printf("=========> Making Task for location: %s, size: %lu\n", 
                  location, wc->task_map.size());
#endif            
        wc->task_string_handle = __itt_string_handle_create(key.c_str());
        wc->task_map.insert({key, wc->task_string_handle});
    } else {
#ifdef ITT_INSTRUMENT_DEBUG
        printf("=========> Found Task for location: %s, size: %lu\n", 
            location, wc->task_map.size());
#endif            
        wc->task_string_handle = search->second;
    }
}

static inline bool is_filtered_location(const std::string& loc) {
    if ( loc.find(s_lookup_pattern, 3 ) != std::string::npos) {
#ifdef ITT_INSTRUMENT_DEBUG
        printf("=========> Filtered location: %s\n", loc.c_str());
#endif        
        return true;
    }
    return false;
}

#endif // ITT_INSTRUMENT

//! Starts the cycle counter (and increases the call count)
#ifdef ITT_INSTRUMENT
inline void _wallcycle_start(gmx_wallcycle* wc, WallCycleCounter ewc, const char* location)
#else 
inline void wallcycle_start(gmx_wallcycle* wc, WallCycleCounter ewc)
#endif
{
    //printf("--------------- at: %s, from: %s\n", __FUNCTION__, location);
    if (wc == nullptr)
    {
        return;
    }

    wallcycleBarrier(wc);
#ifdef ITT_INSTRUMENT
    wc->in_pt_region = false;
    std::string loc(location);

    if ( WallCycleCounter::LaunchGpu == ewc )  {
        if ( wc->invoke_idx == ITT_START_FRAME ) {
            __itt_resume();
            wc->task_domain = __itt_domain_create("Intra-step tasks");
#ifdef ITT_INSTRUMENT_DEBUG
            printf("=========> Resuming VTune collection at frame: %d\n", ITT_START_FRAME);
#endif            
        }

        if ( (wc->invoke_idx >= ITT_START_FRAME) && is_filtered_location(loc) ) {
            get_task_string_handle(wc, location);
            __itt_task_begin(wc->task_domain, __itt_null, __itt_null, wc->task_string_handle );
            get_pt_region(wc, location);
            wc->in_pt_region = true;
#ifdef ITT_INSTRUMENT_DEBUG
            printf("=========> Region Start at frame: %lu\n", (wc->invoke_idx));
#endif            
            __itt_mark_pt_region_begin(wc->pt_region);
        }
    }
#endif


#if DEBUG_WCYCLE
    debug_start_check(wc, ewc);
#endif
    gmx_cycles_t cycle = gmx_cycles_read();
    wc->wcc[ewc].start = cycle;
    if (!wc->wcc_all.empty())
    {
        wc->wc_depth++;
        if (ewc == WallCycleCounter::Run)
        {
            wallcycle_all_start(wc, ewc, cycle);
        }
        else if (wc->wc_depth == 3)
        {
            wallcycle_all_stop(wc, ewc, cycle);
        }
    }
}

//! Starts the cycle counter without increasing the call count
#ifdef ITT_INSTRUMENT
inline void _wallcycle_start_nocount(gmx_wallcycle* wc, WallCycleCounter ewc, const char* location)
#else
inline void wallcycle_start_nocount(gmx_wallcycle* wc, WallCycleCounter ewc)
#endif    
{
    if (wc == nullptr)
    {
        return;
    }
#ifdef ITT_INSTRUMENT
    _wallcycle_start(wc, ewc, location);
#else    
    wallcycle_start(wc, ewc);
#endif    
    wc->wcc[ewc].n--;
}

//! Stop the cycle count for ewc , returns the last cycle count
inline double wallcycle_stop(gmx_wallcycle* wc, WallCycleCounter ewc)
{
    gmx_cycles_t cycle, last;

    if (wc == nullptr)
    {
        return 0;
    }

    wallcycleBarrier(wc);
#ifdef ITT_INSTRUMENT

    if ( WallCycleCounter::LaunchGpu == ewc )  {
//            printf("=========>At function %s location  %s at frame: %lu\n", __FUNCTION__, loc.c_str(), (wc->invoke_idx));
        if ( (wc->invoke_idx >= ITT_START_FRAME) && wc->in_pt_region ){
           __itt_task_end(wc->task_domain);
           __itt_mark_pt_region_end(wc->pt_region);
#ifdef ITT_INSTRUMENT_DEBUG
            printf("=========> Region End at frame: %lu\n", (wc->invoke_idx));
#endif            
           wc->in_pt_region = false;
        }
        (wc->invoke_idx)++;
    }
#endif    

#if DEBUG_WCYCLE
    debug_stop_check(wc, ewc);
#endif

    /* When processes or threads migrate between cores, the cycle counting
     * can get messed up if the cycle counter on different cores are not
     * synchronized. When this happens we expect both large negative and
     * positive cycle differences. We can detect negative cycle differences.
     * Detecting too large positive counts if difficult, since count can be
     * large, especially for ewcRUN. If we detect a negative count,
     * we will not print the cycle accounting table.
     */
    cycle = gmx_cycles_read();
    if (cycle >= wc->wcc[ewc].start)
    {
        last = cycle - wc->wcc[ewc].start;
    }
    else
    {
        last                 = 0;
        wc->haveInvalidCount = true;
    }
    wc->wcc[ewc].c += last;
    wc->wcc[ewc].n++;
    if (!wc->wcc_all.empty())
    {
        wc->wc_depth--;
        if (ewc == WallCycleCounter::Run)
        {
            wallcycle_all_stop(wc, ewc, cycle);
        }
        else if (wc->wc_depth == 2)
        {
            wallcycle_all_start(wc, ewc, cycle);
        }
    }

    return last;
}

//! Only increment call count for ewc by one
inline void wallcycle_increment_event_count(gmx_wallcycle* wc, WallCycleCounter ewc)
{
    if (wc == nullptr)
    {
        return;
    }
    wc->wcc[ewc].n++;
}

//! Returns the cumulative count and cycle count for ewc
void wallcycle_get(gmx_wallcycle* wc, WallCycleCounter ewc, int* n, double* c);

//! Resets all cycle counters to zero
void wallcycle_reset_all(gmx_wallcycle* wc);

//! Scale the cycle counts to reflect how many threads run for that number of cycles
void wallcycle_scale_by_num_threads(gmx_wallcycle* wc, bool isPmeRank, int nthreads_pp, int nthreads_pme);

//! Return reset_counters from wc struct
int64_t wcycle_get_reset_counters(gmx_wallcycle* wc);

//! Set reset_counters
void wcycle_set_reset_counters(gmx_wallcycle* wc, int64_t reset_counters);

//! Set the start sub cycle count for ewcs
inline void wallcycle_sub_start(gmx_wallcycle* wc, WallCycleSubCounter ewcs)
{
    if (sc_useCycleSubcounters && wc != nullptr)
    {
        wc->wcsc[ewcs].start = gmx_cycles_read();
    }
}

//! Set the start sub cycle count for ewcs without increasing the call count
inline void wallcycle_sub_start_nocount(gmx_wallcycle* wc, WallCycleSubCounter ewcs)
{
    if (sc_useCycleSubcounters && wc != nullptr)
    {
        wallcycle_sub_start(wc, ewcs);
        wc->wcsc[ewcs].n--;
    }
}

//! Stop the sub cycle count for ewcs
inline void wallcycle_sub_stop(gmx_wallcycle* wc, WallCycleSubCounter ewcs)
{
    if (sc_useCycleSubcounters && wc != nullptr)
    {
        wc->wcsc[ewcs].c += gmx_cycles_read() - wc->wcsc[ewcs].start;
        wc->wcsc[ewcs].n++;
    }
}

#endif
