/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017 by the GROMACS development team.
 * Copyright (c) 2018,2019,2020,2021, by the GROMACS development team, led by
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
#include "gmxpre.h"

#include "stat.h"

#include <cstdio>
#include <cstring>

#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/localtopologychecker.h"
#include "gromacs/fileio/checkpoint.h"
#include "gromacs/fileio/xtcio.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/math/utilities.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/rbin.h"
#include "gromacs/mdlib/tgroup.h"
#include "gromacs/mdlib/vcm.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/observablesreducer.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/smalloc.h"

typedef struct gmx_global_stat
{
    t_bin* rb;
    int*   itc0;
    int*   itc1;
} t_gmx_global_stat;

gmx_global_stat_t global_stat_init(const t_inputrec* ir)
{
    gmx_global_stat_t gs;

    snew(gs, 1);

    gs->rb = mk_bin();
    snew(gs->itc0, ir->opts.ngtc);
    snew(gs->itc1, ir->opts.ngtc);

    return gs;
}

void global_stat_destroy(gmx_global_stat_t gs)
{
    destroy_bin(gs->rb);
    sfree(gs->itc0);
    sfree(gs->itc1);
    sfree(gs);
}

static int filter_enerdterm(const real* afrom, gmx_bool bToBuffer, real* ato, gmx_bool bTemp, gmx_bool bPres, gmx_bool bEner)
{
    int i, to, from;

    from = 0;
    to   = 0;
    for (i = 0; i < F_NRE; i++)
    {
        if (bToBuffer)
        {
            from = i;
        }
        else
        {
            to = i;
        }
        switch (i)
        {
            case F_EKIN:
            case F_TEMP:
            case F_DKDL:
                if (bTemp)
                {
                    ato[to++] = afrom[from++];
                }
                break;
            case F_PRES:
            case F_PDISPCORR:
                if (bPres)
                {
                    ato[to++] = afrom[from++];
                }
                break;
            case F_ETOT:
            case F_ECONSERVED:
                // Don't reduce total and conserved energy
                // because they are computed later (see #4301)
                break;
            default:
                if (bEner)
                {
                    ato[to++] = afrom[from++];
                }
                break;
        }
    }

    return to;
}

void global_stat(const gmx_global_stat&   gs,
                 const t_commrec*         cr,
                 gmx_enerdata_t*          enerd,
                 tensor                   fvir,
                 tensor                   svir,
                 const t_inputrec&        inputrec,
                 gmx_ekindata_t*          ekind,
                 gmx::ArrayRef<real>      constraintsRmsdData,
                 t_vcm*                   vcm,
                 gmx::ArrayRef<real>      sig,
                 bool                     bSumEkinhOld,
                 int                      flags,
                 int64_t                  step,
                 gmx::ObservablesReducer* observablesReducer)
/* instead of current system, gmx_booleans for summing virial, kinetic energy, and other terms */
{
    int ie = 0, ifv = 0, isv = 0, irmsd = 0;
    int idedl = 0, idedlo = 0, idvdll = 0, idvdlnl = 0, iepl = 0, icm = 0, imass = 0, ica = 0, inb = 0;
    int isig = -1;
    int icj = -1, ici = -1, icx = -1;

    bool checkNumberOfBondedInteractions = (flags & CGLO_CHECK_NUMBER_OF_BONDED_INTERACTIONS) != 0;
    bool bVV                             = EI_VV(inputrec.eI);
    bool bTemp                           = ((flags & CGLO_TEMPERATURE) != 0);
    bool bEner                           = ((flags & CGLO_ENERGY) != 0);
    bool bPres                           = ((flags & CGLO_PRESSURE) != 0);
    bool bConstrVir                      = ((flags & CGLO_CONSTRAINT) != 0);
    bool bEkinAveVel                     = (inputrec.eI == IntegrationAlgorithm::VV
                        || (inputrec.eI == IntegrationAlgorithm::VVAK && bPres));
    bool bReadEkin                       = ((flags & CGLO_READEKIN) != 0);

    t_bin* rb   = gs.rb;
    int*   itc0 = gs.itc0;
    int*   itc1 = gs.itc1;


    reset_bin(rb);
    /* This routine copies all the data to be summed to one big buffer
     * using the t_bin struct.
     */

    /* First, we neeed to identify which enerd->term should be
       communicated.  Temperature and pressure terms should only be
       communicated and summed when they need to be, to avoid repeating
       the sums and overcounting. */

    std::array<real, F_NRE> copyenerd;
    int nener = filter_enerdterm(enerd->term.data(), TRUE, copyenerd.data(), bTemp, bPres, bEner);

    /* First, the data that needs to be communicated with velocity verlet every time
       This is just the constraint virial.*/
    if (bConstrVir)
    {
        isv = add_binr(rb, DIM * DIM, svir[0]);
    }

    /* We need the force virial and the kinetic energy for the first time through with velocity verlet */
    if (bTemp || !bVV)
    {
        if (ekind)
        {
            for (int j = 0; (j < inputrec.opts.ngtc); j++)
            {
                if (bSumEkinhOld)
                {
                    itc0[j] = add_binr(rb, DIM * DIM, ekind->tcstat[j].ekinh_old[0]);
                }
                if (bEkinAveVel && !bReadEkin)
                {
                    itc1[j] = add_binr(rb, DIM * DIM, ekind->tcstat[j].ekinf[0]);
                }
                else if (!bReadEkin)
                {
                    itc1[j] = add_binr(rb, DIM * DIM, ekind->tcstat[j].ekinh[0]);
                }
            }
            /* these probably need to be put into one of these categories */
            idedl = add_binr(rb, 1, &(ekind->dekindl));
            if (bSumEkinhOld)
            {
                idedlo = add_binr(rb, 1, &(ekind->dekindl_old));
            }
            if (ekind->cosacc.cos_accel != 0)
            {
                ica = add_binr(rb, 1, &(ekind->cosacc.mvcos));
            }
        }
    }

    if (bPres)
    {
        ifv = add_binr(rb, DIM * DIM, fvir[0]);
    }

    gmx::EnumerationArray<NonBondedEnergyTerms, int> inn;
    if (bEner)
    {
        ie = add_binr(rb, nener, copyenerd.data());
        if (!constraintsRmsdData.empty())
        {
            irmsd = add_binr(rb, 2, constraintsRmsdData.data());
        }
        for (auto key : gmx::keysOf(inn))
        {
            inn[key] = add_binr(rb, enerd->grpp.nener, enerd->grpp.energyGroupPairTerms[key].data());
        }
        if (inputrec.efep != FreeEnergyPerturbationType::No)
        {
            idvdll  = add_bind(rb, enerd->dvdl_lin);
            idvdlnl = add_bind(rb, enerd->dvdl_nonlin);
            if (enerd->foreignLambdaTerms.numLambdas() > 0)
            {
                iepl = add_bind(rb,
                                enerd->foreignLambdaTerms.energies().size(),
                                enerd->foreignLambdaTerms.energies().data());
            }
        }
    }

    if (vcm)
    {
        icm   = add_binr(rb, DIM * vcm->nr, vcm->group_p[0]);
        imass = add_binr(rb, vcm->nr, vcm->group_mass.data());
        if (vcm->mode == ComRemovalAlgorithm::Angular)
        {
            icj = add_binr(rb, DIM * vcm->nr, vcm->group_j[0]);
            icx = add_binr(rb, DIM * vcm->nr, vcm->group_x[0]);
            ici = add_binr(rb, DIM * DIM * vcm->nr, vcm->group_i[0][0]);
        }
    }

    double nb;
    if (checkNumberOfBondedInteractions)
    {
        GMX_RELEASE_ASSERT(DOMAINDECOMP(cr),
                           "No need to check number of bonded interactions when not using domain "
                           "decomposition");
        nb  = cr->dd->localTopologyChecker->numBondedInteractions();
        inb = add_bind(rb, 1, &nb);
    }
    if (!sig.empty())
    {
        isig = add_binr(rb, sig);
    }

    gmx::ArrayRef<double> observablesReducerBuffer = observablesReducer->communicationBuffer();
    int                   ior                      = 0;
    if (!observablesReducerBuffer.empty())
    {
        ior = add_bind(rb, observablesReducerBuffer.ssize(), observablesReducerBuffer.data());
    }

    sum_bin(rb, cr);

    /* Extract all the data locally */

    if (bConstrVir)
    {
        extract_binr(rb, isv, DIM * DIM, svir[0]);
    }

    /* We need the force virial and the kinetic energy for the first time through with velocity verlet */
    if (bTemp || !bVV)
    {
        if (ekind)
        {
            for (int j = 0; (j < inputrec.opts.ngtc); j++)
            {
                if (bSumEkinhOld)
                {
                    extract_binr(rb, itc0[j], DIM * DIM, ekind->tcstat[j].ekinh_old[0]);
                }
                if (bEkinAveVel && !bReadEkin)
                {
                    extract_binr(rb, itc1[j], DIM * DIM, ekind->tcstat[j].ekinf[0]);
                }
                else if (!bReadEkin)
                {
                    extract_binr(rb, itc1[j], DIM * DIM, ekind->tcstat[j].ekinh[0]);
                }
            }
            extract_binr(rb, idedl, 1, &(ekind->dekindl));
            if (bSumEkinhOld)
            {
                extract_binr(rb, idedlo, 1, &(ekind->dekindl_old));
            }
            if (ekind->cosacc.cos_accel != 0)
            {
                extract_binr(rb, ica, 1, &(ekind->cosacc.mvcos));
            }
        }
    }
    if (bPres)
    {
        extract_binr(rb, ifv, DIM * DIM, fvir[0]);
    }

    if (bEner)
    {
        extract_binr(rb, ie, nener, copyenerd.data());
        if (!constraintsRmsdData.empty())
        {
            extract_binr(rb, irmsd, constraintsRmsdData);
        }
        for (auto key : gmx::keysOf(inn))
        {
            extract_binr(rb, inn[key], enerd->grpp.nener, enerd->grpp.energyGroupPairTerms[key].data());
        }
        if (inputrec.efep != FreeEnergyPerturbationType::No)
        {
            extract_bind(rb, idvdll, enerd->dvdl_lin);
            extract_bind(rb, idvdlnl, enerd->dvdl_nonlin);
            if (enerd->foreignLambdaTerms.numLambdas() > 0)
            {
                extract_bind(rb,
                             iepl,
                             enerd->foreignLambdaTerms.energies().size(),
                             enerd->foreignLambdaTerms.energies().data());
            }
        }

        filter_enerdterm(copyenerd.data(), FALSE, enerd->term.data(), bTemp, bPres, bEner);
    }

    if (vcm)
    {
        extract_binr(rb, icm, DIM * vcm->nr, vcm->group_p[0]);
        extract_binr(rb, imass, vcm->nr, vcm->group_mass.data());
        if (vcm->mode == ComRemovalAlgorithm::Angular)
        {
            extract_binr(rb, icj, DIM * vcm->nr, vcm->group_j[0]);
            extract_binr(rb, icx, DIM * vcm->nr, vcm->group_x[0]);
            extract_binr(rb, ici, DIM * DIM * vcm->nr, vcm->group_i[0][0]);
        }
    }

    if (checkNumberOfBondedInteractions)
    {
        extract_bind(rb, inb, 1, &nb);
        GMX_RELEASE_ASSERT(DOMAINDECOMP(cr),
                           "No need to check number of bonded interactions when not using domain "
                           "decomposition");
        cr->dd->localTopologyChecker->setNumberOfBondedInteractionsOverAllDomains(gmx::roundToInt(nb));
    }

    if (!sig.empty())
    {
        extract_binr(rb, isig, sig);
    }

    if (!observablesReducerBuffer.empty())
    {
        extract_bind(rb, ior, observablesReducerBuffer.ssize(), observablesReducerBuffer.data());
        observablesReducer->reductionComplete(step);
    }
}
