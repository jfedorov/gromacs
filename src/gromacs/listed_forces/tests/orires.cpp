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
 * \brief Implements test of orientation restraints
 *
 * \author Joe Jordan <ejjordan@kth.se>
 * \ingroup module_listed_forces
 */
#include "gmxpre.h"

#include "gromacs/listed_forces/orires.h"

#include <gtest/gtest.h>

#include <numeric>
#include <memory>

#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/topology/topology.h"
#include "gromacs/topology/mtop_util.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

void inputrecSetup(t_inputrec* inputrec)
{
    inputrec->orires_fc     = 100;
    inputrec->orires_tau    = 1;
    inputrec->bPeriodicMols = false;
    inputrec->nsteps        = 0;
    inputrec->delta_t       = 0.002;
}

gmx_moltype_t getMoltype(const int oriresTypeIndex, const int interactionType)
{
    gmx_moltype_t moltype;
    moltype.atoms.nr         = NRAL(interactionType);
    std::vector<int>& iatoms = moltype.ilist[interactionType].iatoms;
    iatoms.push_back(oriresTypeIndex);
    iatoms.push_back(0);
    iatoms.push_back(1);
    return moltype;
}

t_iparams getOriresParam(int ex, int power, int label, real c, real obs, real kfac)
{
    t_iparams oriresParams;
    oriresParams.orires = { ex, power, label, c, obs, kfac };
    return oriresParams;
}

t_iparams getBondParam(real rA, real kA)
{
    t_iparams bondParams;
    bondParams.harmonic = { rA, kA };
    return bondParams;
}

gmx_molblock_t getMolblock(const int typeIndex, const int nmol)
{
    gmx_molblock_t molblock;
    molblock.type = typeIndex;
    molblock.nmol = nmol;
    return molblock;
}

/*! \brief Initializes a basic topology with orientation restraints*/
void mtopSetup(gmx_mtop_t* mtop)
{
    const int bondType   = 0;
    const int oriresType = 1;
    const int numMols    = 1;

    gmx_moltype_t bondMoltype   = getMoltype(bondType, F_BONDS);
    gmx_moltype_t oriresMoltype = getMoltype(oriresType, F_ORIRES);

    gmx_molblock_t bondMolblock   = getMolblock(bondType, numMols);
    gmx_molblock_t oriresMolblock = getMolblock(oriresType, numMols);

    mtop->ffparams.iparams.push_back(getBondParam(1, 1));
    mtop->ffparams.iparams.push_back(getOriresParam(1, 1, 1, 1, 1, 1));

    mtop->moltype.push_back(bondMoltype);
    mtop->moltype.push_back(oriresMoltype);

    const int oriresMolblockSize = 6;
    for (int i = 0; i < oriresMolblockSize; i++)
    {
        mtop->molblock.push_back(oriresMolblock);
    }
    mtop->molblock.push_back(bondMolblock);

    mtop->natoms = oriresMoltype.atoms.nr * oriresMolblock.nmol * oriresMolblockSize
                   + bondMoltype.atoms.nr * bondMolblock.nmol;
    mtop->finalize();
}

TEST(OriresTest, CanConstructOrires)
{
    t_inputrec inputrec;
    inputrecSetup(&inputrec);
    gmx_mtop_t mtop;
    mtopSetup(&mtop);
    t_state state;

    EXPECT_ANY_THROW(t_oriresdata oriresdata(nullptr, mtop, inputrec, nullptr, nullptr, &state));
}

} // namespace

} // namespace test

} // namespace gmx
