/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2006 - 2014, The GROMACS development team.
 * Copyright (c) 2015,2016,2017,2018,2019,2020,2021, by the GROMACS development team, led by
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
 *
 * \brief This file defines functions used by the domdec module
 * while managing the construction, use and error checking for
 * topologies local to a DD rank.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */

#include "gmxpre.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>

#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_network.h"
#include "gromacs/domdec/ga2la.h"
#include "gromacs/domdec/localtopologychecker.h"
#include "gromacs/domdec/options.h"
#include "gromacs/domdec/reversetopology.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/pbcutil/mshift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topsort.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"

#include "domdec_constraints.h"
#include "domdec_internal.h"
#include "domdec_vsite.h"
#include "dump.h"

using gmx::ArrayRef;
using gmx::DDBondedChecking;
using gmx::RVec;

/*! \brief The number of integer item in the local state, used for broadcasting of the state */
#define NITEM_DD_INIT_LOCAL_STATE 5

void dd_init_local_state(const gmx_domdec_t& dd, const t_state* state_global, t_state* state_local)
{
    int buf[NITEM_DD_INIT_LOCAL_STATE];

    if (DDMASTER(dd))
    {
        buf[0] = state_global->flags;
        buf[1] = state_global->ngtc;
        buf[2] = state_global->nnhpres;
        buf[3] = state_global->nhchainlength;
        buf[4] = state_global->dfhist ? state_global->dfhist->nlambda : 0;
    }
    dd_bcast(&dd, NITEM_DD_INIT_LOCAL_STATE * sizeof(int), buf);

    init_gtc_state(state_local, buf[1], buf[2], buf[3]);
    init_dfhist_state(state_local, buf[4]);
    state_local->flags = buf[0];
}

/*! \brief Check if a link is stored in \p link between charge groups \p cg_gl and \p cg_gl_j and if not so, store a link */
static void check_link(t_blocka* link, int cg_gl, int cg_gl_j)
{
    bool bFound = false;
    for (int k = link->index[cg_gl]; k < link->index[cg_gl + 1]; k++)
    {
        GMX_RELEASE_ASSERT(link->a, "Inconsistent NULL pointer while making charge-group links");
        if (link->a[k] == cg_gl_j)
        {
            bFound = TRUE;
        }
    }
    if (!bFound)
    {
        GMX_RELEASE_ASSERT(link->a || link->index[cg_gl + 1] + 1 > link->nalloc_a,
                           "Inconsistent allocation of link");
        /* Add this charge group link */
        if (link->index[cg_gl + 1] + 1 > link->nalloc_a)
        {
            link->nalloc_a = over_alloc_large(link->index[cg_gl + 1] + 1);
            srenew(link->a, link->nalloc_a);
        }
        link->a[link->index[cg_gl + 1]] = cg_gl_j;
        link->index[cg_gl + 1]++;
    }
}

void makeBondedLinks(gmx_domdec_t*                              dd,
                     const gmx_mtop_t&                          mtop,
                     gmx::ArrayRef<AtomInfoWithinMoleculeBlock> atomInfoForEachMoleculeBlock)
{

    if (!dd->comm->systemInfo.filterBondedCommunication)
    {
        /* Only communicate atoms based on cut-off */
        dd->comm->bondedLinks = nullptr;
        return;
    }

    t_blocka* link = nullptr;

    /* For each atom make a list of other atoms in the system
     * that a linked to it via bonded interactions
     * which are also stored in reverse_top.
     */

    reverse_ilist_t ril_intermol;
    if (mtop.bIntermolecularInteractions)
    {
        t_atoms atoms;

        atoms.nr   = mtop.natoms;
        atoms.atom = nullptr;

        GMX_RELEASE_ASSERT(mtop.intermolecular_ilist,
                           "We should have an ilist when intermolecular interactions are on");

        ReverseTopOptions rtOptions(DDBondedChecking::ExcludeZeroLimit);
        make_reverse_ilist(
                *mtop.intermolecular_ilist, &atoms, rtOptions, AtomLinkRule::AllAtomsInBondeds, &ril_intermol);
    }

    snew(link, 1);
    snew(link->index, mtop.natoms + 1);
    link->nalloc_a = 0;
    link->a        = nullptr;

    link->index[0]                 = 0;
    int indexOfFirstAtomInMolecule = 0;
    int numLinkedAtoms             = 0;
    for (size_t mb = 0; mb < mtop.molblock.size(); mb++)
    {
        const gmx_molblock_t& molb = mtop.molblock[mb];
        if (molb.nmol == 0)
        {
            continue;
        }
        const gmx_moltype_t& molt = mtop.moltype[molb.type];
        /* Make a reverse ilist in which the interactions are linked
         * to all atoms, not only the first atom as in gmx_reverse_top.
         * The constraints are discarded here.
         */
        ReverseTopOptions rtOptions(DDBondedChecking::ExcludeZeroLimit);
        reverse_ilist_t   ril;
        make_reverse_ilist(molt.ilist, &molt.atoms, rtOptions, AtomLinkRule::AllAtomsInBondeds, &ril);

        AtomInfoWithinMoleculeBlock* atomInfoOfMoleculeBlock = &atomInfoForEachMoleculeBlock[mb];

        int mol = 0;
        for (mol = 0; mol < (mtop.bIntermolecularInteractions ? molb.nmol : 1); mol++)
        {
            for (int a = 0; a < molt.atoms.nr; a++)
            {
                int atomIndex              = indexOfFirstAtomInMolecule + a;
                link->index[atomIndex + 1] = link->index[atomIndex];
                int i                      = ril.index[a];
                while (i < ril.index[a + 1])
                {
                    int ftype = ril.il[i++];
                    int nral  = NRAL(ftype);
                    /* Skip the ifunc index */
                    i++;
                    for (int j = 0; j < nral; j++)
                    {
                        int aj = ril.il[i + j];
                        if (aj != a)
                        {
                            check_link(link, atomIndex, indexOfFirstAtomInMolecule + aj);
                        }
                    }
                    i += nral_rt(ftype);
                }

                if (mtop.bIntermolecularInteractions)
                {
                    int i = ril_intermol.index[atomIndex];
                    while (i < ril_intermol.index[atomIndex + 1])
                    {
                        int ftype = ril_intermol.il[i++];
                        int nral  = NRAL(ftype);
                        /* Skip the ifunc index */
                        i++;
                        for (int j = 0; j < nral; j++)
                        {
                            /* Here we assume we have no charge groups;
                             * this has been checked above.
                             */
                            int aj = ril_intermol.il[i + j];
                            check_link(link, atomIndex, aj);
                        }
                        i += nral_rt(ftype);
                    }
                }
                if (link->index[atomIndex + 1] - link->index[atomIndex] > 0)
                {
                    SET_CGINFO_BOND_INTER(atomInfoOfMoleculeBlock->atomInfo[a]);
                    numLinkedAtoms++;
                }
            }

            indexOfFirstAtomInMolecule += molt.atoms.nr;
        }
        int nlink_mol = link->index[indexOfFirstAtomInMolecule]
                        - link->index[indexOfFirstAtomInMolecule - molt.atoms.nr];

        if (debug)
        {
            fprintf(debug,
                    "molecule type '%s' %d atoms has %d atom links through bonded interac.\n",
                    *molt.name,
                    molt.atoms.nr,
                    nlink_mol);
        }

        if (molb.nmol > mol)
        {
            /* Copy the data for the rest of the molecules in this block */
            link->nalloc_a += (molb.nmol - mol) * nlink_mol;
            srenew(link->a, link->nalloc_a);
            for (; mol < molb.nmol; mol++)
            {
                for (int a = 0; a < molt.atoms.nr; a++)
                {
                    int atomIndex              = indexOfFirstAtomInMolecule + a;
                    link->index[atomIndex + 1] = link->index[atomIndex + 1 - molt.atoms.nr] + nlink_mol;
                    for (int j = link->index[atomIndex]; j < link->index[atomIndex + 1]; j++)
                    {
                        link->a[j] = link->a[j - nlink_mol] + molt.atoms.nr;
                    }
                    if (link->index[atomIndex + 1] - link->index[atomIndex] > 0
                        && atomIndex - atomInfoOfMoleculeBlock->indexOfFirstAtomInMoleculeBlock
                                   < gmx::ssize(atomInfoOfMoleculeBlock->atomInfo))
                    {
                        SET_CGINFO_BOND_INTER(
                                atomInfoOfMoleculeBlock
                                        ->atomInfo[atomIndex - atomInfoOfMoleculeBlock->indexOfFirstAtomInMoleculeBlock]);
                        numLinkedAtoms++;
                    }
                }
                indexOfFirstAtomInMolecule += molt.atoms.nr;
            }
        }
    }

    if (debug)
    {
        fprintf(debug, "Of the %d atoms %d are linked via bonded interactions\n", mtop.natoms, numLinkedAtoms);
    }

    dd->comm->bondedLinks = link;
}

typedef struct
{
    real r2;
    int  ftype;
    int  a1;
    int  a2;
} bonded_distance_t;

/*! \brief Compare distance^2 \p r2 against the distance in \p bd and if larger store it along with \p ftype and atom indices \p a1 and \p a2 */
static void update_max_bonded_distance(real r2, int ftype, int a1, int a2, bonded_distance_t* bd)
{
    if (r2 > bd->r2)
    {
        bd->r2    = r2;
        bd->ftype = ftype;
        bd->a1    = a1;
        bd->a2    = a2;
    }
}

/*! \brief Set the distance, function type and atom indices for the longest distance between atoms of molecule type \p molt for two-body and multi-body bonded interactions */
static void bonded_cg_distance_mol(const gmx_moltype_t*   molt,
                                   const DDBondedChecking ddBondedChecking,
                                   gmx_bool               bExcl,
                                   ArrayRef<const RVec>   x,
                                   bonded_distance_t*     bd_2b,
                                   bonded_distance_t*     bd_mb)
{
    const ReverseTopOptions rtOptions(ddBondedChecking);

    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        if (dd_check_ftype(ftype, rtOptions))
        {
            const auto& il   = molt->ilist[ftype];
            int         nral = NRAL(ftype);
            if (nral > 1)
            {
                for (int i = 0; i < il.size(); i += 1 + nral)
                {
                    for (int ai = 0; ai < nral; ai++)
                    {
                        int atomI = il.iatoms[i + 1 + ai];
                        for (int aj = ai + 1; aj < nral; aj++)
                        {
                            int atomJ = il.iatoms[i + 1 + aj];
                            if (atomI != atomJ)
                            {
                                real rij2 = distance2(x[atomI], x[atomJ]);

                                update_max_bonded_distance(
                                        rij2, ftype, atomI, atomJ, (nral == 2) ? bd_2b : bd_mb);
                            }
                        }
                    }
                }
            }
        }
    }
    if (bExcl)
    {
        const auto& excls = molt->excls;
        for (gmx::index ai = 0; ai < excls.ssize(); ai++)
        {
            for (const int aj : excls[ai])
            {
                if (ai != aj)
                {
                    real rij2 = distance2(x[ai], x[aj]);

                    /* There is no function type for exclusions, use -1 */
                    update_max_bonded_distance(rij2, -1, ai, aj, bd_2b);
                }
            }
        }
    }
}

/*! \brief Set the distance, function type and atom indices for the longest atom distance involved in intermolecular interactions for two-body and multi-body bonded interactions */
static void bonded_distance_intermol(const InteractionLists& ilists_intermol,
                                     const DDBondedChecking  ddBondedChecking,
                                     ArrayRef<const RVec>    x,
                                     PbcType                 pbcType,
                                     const matrix            box,
                                     bonded_distance_t*      bd_2b,
                                     bonded_distance_t*      bd_mb)
{
    t_pbc pbc;

    set_pbc(&pbc, pbcType, box);

    const ReverseTopOptions rtOptions(ddBondedChecking);

    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        if (dd_check_ftype(ftype, rtOptions))
        {
            const auto& il   = ilists_intermol[ftype];
            int         nral = NRAL(ftype);

            /* No nral>1 check here, since intermol interactions always
             * have nral>=2 (and the code is also correct for nral=1).
             */
            for (int i = 0; i < il.size(); i += 1 + nral)
            {
                for (int ai = 0; ai < nral; ai++)
                {
                    int atom_i = il.iatoms[i + 1 + ai];

                    for (int aj = ai + 1; aj < nral; aj++)
                    {
                        rvec dx;

                        int atom_j = il.iatoms[i + 1 + aj];

                        pbc_dx(&pbc, x[atom_i], x[atom_j], dx);

                        const real rij2 = norm2(dx);

                        update_max_bonded_distance(rij2, ftype, atom_i, atom_j, (nral == 2) ? bd_2b : bd_mb);
                    }
                }
            }
        }
    }
}

//! Returns whether \p molt has at least one virtual site
static bool moltypeHasVsite(const gmx_moltype_t& molt)
{
    bool hasVsite = false;
    for (int i = 0; i < F_NRE; i++)
    {
        if ((interaction_function[i].flags & IF_VSITE) && !molt.ilist[i].empty())
        {
            hasVsite = true;
        }
    }

    return hasVsite;
}

//! Returns coordinates not broken over PBC for a molecule
static void getWholeMoleculeCoordinates(const gmx_moltype_t*  molt,
                                        const gmx_ffparams_t* ffparams,
                                        PbcType               pbcType,
                                        t_graph*              graph,
                                        const matrix          box,
                                        ArrayRef<const RVec>  x,
                                        ArrayRef<RVec>        xs)
{
    if (pbcType != PbcType::No)
    {
        mk_mshift(nullptr, graph, pbcType, box, as_rvec_array(x.data()));

        shift_x(graph, box, as_rvec_array(x.data()), as_rvec_array(xs.data()));
        /* By doing an extra mk_mshift the molecules that are broken
         * because they were e.g. imported from another software
         * will be made whole again. Such are the healing powers
         * of GROMACS.
         */
        mk_mshift(nullptr, graph, pbcType, box, as_rvec_array(xs.data()));
    }
    else
    {
        /* We copy the coordinates so the original coordinates remain
         * unchanged, just to be 100% sure that we do not affect
         * binary reproducibility of simulations.
         */
        for (int i = 0; i < molt->atoms.nr; i++)
        {
            copy_rvec(x[i], xs[i]);
        }
    }

    if (moltypeHasVsite(*molt))
    {
        gmx::constructVirtualSites(xs, ffparams->iparams, molt->ilist);
    }
}

void dd_bonded_cg_distance(const gmx::MDLogger&   mdlog,
                           const gmx_mtop_t&      mtop,
                           const t_inputrec&      inputrec,
                           ArrayRef<const RVec>   x,
                           const matrix           box,
                           const DDBondedChecking ddBondedChecking,
                           real*                  r_2b,
                           real*                  r_mb)
{
    bonded_distance_t bd_2b = { 0, -1, -1, -1 };
    bonded_distance_t bd_mb = { 0, -1, -1, -1 };

    bool bExclRequired = inputrecExclForces(&inputrec);

    *r_2b         = 0;
    *r_mb         = 0;
    int at_offset = 0;
    for (const gmx_molblock_t& molb : mtop.molblock)
    {
        const gmx_moltype_t& molt = mtop.moltype[molb.type];
        if (molt.atoms.nr == 1 || molb.nmol == 0)
        {
            at_offset += molb.nmol * molt.atoms.nr;
        }
        else
        {
            t_graph graph;
            if (inputrec.pbcType != PbcType::No)
            {
                graph = mk_graph_moltype(molt);
            }

            std::vector<RVec> xs(molt.atoms.nr);
            for (int mol = 0; mol < molb.nmol; mol++)
            {
                getWholeMoleculeCoordinates(&molt,
                                            &mtop.ffparams,
                                            inputrec.pbcType,
                                            &graph,
                                            box,
                                            x.subArray(at_offset, molt.atoms.nr),
                                            xs);

                bonded_distance_t bd_mol_2b = { 0, -1, -1, -1 };
                bonded_distance_t bd_mol_mb = { 0, -1, -1, -1 };

                bonded_cg_distance_mol(&molt, ddBondedChecking, bExclRequired, xs, &bd_mol_2b, &bd_mol_mb);

                /* Process the mol data adding the atom index offset */
                update_max_bonded_distance(bd_mol_2b.r2,
                                           bd_mol_2b.ftype,
                                           at_offset + bd_mol_2b.a1,
                                           at_offset + bd_mol_2b.a2,
                                           &bd_2b);
                update_max_bonded_distance(bd_mol_mb.r2,
                                           bd_mol_mb.ftype,
                                           at_offset + bd_mol_mb.a1,
                                           at_offset + bd_mol_mb.a2,
                                           &bd_mb);

                at_offset += molt.atoms.nr;
            }
        }
    }

    if (mtop.bIntermolecularInteractions)
    {
        GMX_RELEASE_ASSERT(mtop.intermolecular_ilist,
                           "We should have an ilist when intermolecular interactions are on");

        bonded_distance_intermol(
                *mtop.intermolecular_ilist, ddBondedChecking, x, inputrec.pbcType, box, &bd_2b, &bd_mb);
    }

    *r_2b = sqrt(bd_2b.r2);
    *r_mb = sqrt(bd_mb.r2);

    if (*r_2b > 0 || *r_mb > 0)
    {
        GMX_LOG(mdlog.info).appendText("Initial maximum distances in bonded interactions:");
        if (*r_2b > 0)
        {
            GMX_LOG(mdlog.info)
                    .appendTextFormatted(
                            "    two-body bonded interactions: %5.3f nm, %s, atoms %d %d",
                            *r_2b,
                            (bd_2b.ftype >= 0) ? interaction_function[bd_2b.ftype].longname : "Exclusion",
                            bd_2b.a1 + 1,
                            bd_2b.a2 + 1);
        }
        if (*r_mb > 0)
        {
            GMX_LOG(mdlog.info)
                    .appendTextFormatted(
                            "  multi-body bonded interactions: %5.3f nm, %s, atoms %d %d",
                            *r_mb,
                            interaction_function[bd_mb.ftype].longname,
                            bd_mb.a1 + 1,
                            bd_mb.a2 + 1);
        }
    }
}
