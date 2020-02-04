/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020, by the GROMACS development team, led by
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
 * Implements QMMMTopologyPreprocessor
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "qmmmtopologypreprocessor.h"

#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/mtop_lookup.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"

namespace gmx
{

QMMMTopologyPreprocessor::QMMMTopologyPreprocessor(gmx_mtop_t* mtop, const std::vector<index> qmIndices)
{
    qmIndices_ = qmIndices;
    topInfo_   = QMMMTopologyInfo();

    // 1) Split QM-containing molecules from other molecules in blocks
    splitQMblocks(mtop);

    // 1.5) Nullify charges on all virtual sites consisting of QM only atoms
    modifyQMMMVirtualSites(mtop);

    // 2) Nullify charges on all QM atoms
    removeQMClassicalCharges(mtop);

    // 3) Exclude LJ interactions between QM atoms
    addQMLJExclusions(mtop);

    // 4) Build atomNumbers vector with atomic numbers of all atoms
    buildQMMMAtomNumbers(mtop);

    // 5) Make F_CONNBOND between atoms within QM region
    modifyQMMMBonds(mtop);

    /*
     * 6) Remove angles containing 2 or more QM atoms
     * 7) Remove all settles containing any number of QM atoms
     */
    modifyQMMMAngles(mtop);

    // 8) Remove dihedrals containing 3 or more QM atoms
    modifyQMMMDihedrals(mtop);

    // 9) Build vector containing pairs of bonded QM - MM atoms (Link frontier)
    buildQMMMLink(mtop);

    // finalize topology
    mtop->finalize();
}

const QMMMTopologyInfo& QMMMTopologyPreprocessor::topInfo() const
{
    return topInfo_;
}

const std::vector<int>& QMMMTopologyPreprocessor::atomNumbers() const
{
    return atomNumbers_;
}

const std::vector<real>& QMMMTopologyPreprocessor::atomCharges() const
{
    return atomCharges_;
}

const std::vector<LinkFrontier>& QMMMTopologyPreprocessor::link() const
{
    return link_;
}

bool QMMMTopologyPreprocessor::isQMAtom(index globalAtomIndex)
{
    return std::find(qmIndices_.begin(), qmIndices_.end(), globalAtomIndex) != qmIndices_.end();
}

void QMMMTopologyPreprocessor::splitQMblocks(gmx_mtop_t* mtop)
{

    // Global counter of atoms
    index iAt = 0;

    // Counter of molecules point to the specific moltype
    std::vector<int> mbCount(mtop->moltype.size());
    for (size_t mb = 0; mb < mtop->molblock.size(); mb++)
    {
        mbCount[mtop->molblock[mb].type] += mtop->molblock[mb].nmol;
    }

    // Loop over all blocks in topology
    // mb - current block in mtop
    for (size_t mb = 0; mb < mtop->molblock.size(); mb++)
    {
        // Initialize block as non-QM first
        bQMBlock_.push_back(false);

        // Pointer to current block
        gmx_molblock_t* molb = &mtop->molblock[mb];

        // Number of atoms in all molecules of current block
        int nat_mol = mtop->moltype[molb->type].atoms.nr;

        // Loop over all molecules in molb block, total number of them mb
        // mol - current molecule in block mb (molb)
        for (int mol = 0; mol < molb->nmol; mol++)
        {

            // search for QM atoms in current molecule
            bool bQMMM = false;
            for (int i = 0; i < nat_mol; i++)
            {
                if (isQMAtom(iAt))
                {
                    bQMMM = true;
                }
                iAt++;
            }

            // Apparently current molecule (mb,mol) contains QM atoms
            // We should split it from the current block and create new blocks
            // For that molecule and all molecules after it new block will be created
            if (bQMMM)
            {
                // if this block contains only 1 molecule, then splitting not needed
                if (molb->nmol > 1)
                {
                    // If current molecule is not the first in the block,
                    // then we need to first create block for it and all molecule before it
                    if (mol > 0)
                    {
                        // Split the molblock at this molecule
                        auto pos = mtop->molblock.begin() + mb + 1;
                        mtop->molblock.insert(pos, mtop->molblock[mb]);
                        mtop->molblock[mb].nmol = mol;
                        mtop->molblock[mb + 1].nmol -= mol;
                        bQMBlock_[mb] = false;
                        bQMBlock_.push_back(true);
                        mb++;
                        molb = &mtop->molblock[mb];
                    }

                    // If current molecule is not the only one in new block,
                    // Then we split new block after that molecule
                    if (molb->nmol > 1)
                    {
                        auto pos = mtop->molblock.begin() + mb + 1;
                        mtop->molblock.insert(pos, mtop->molblock[mb]);
                        molb                    = &mtop->molblock[mb];
                        mtop->molblock[mb].nmol = 1;
                        mtop->molblock[mb + 1].nmol -= 1;
                        bQMBlock_[mb] = true;
                    }
                }
                else
                {
                    bQMBlock_[mb] = true;
                }

                // Create a copy of a moltype for a molecule
                // containing QM atoms and append it in the end of the moltype vector
                // if there is only 1 molecule pointing to that type the skip that step
                if (mbCount[molb->type] > 1)
                {
                    // Here comes a huge piece of "not so good" code, because someone deleted operator= from gmx_moltype_t
                    std::vector<gmx_moltype_t> temp(mtop->moltype.size());
                    for (size_t i = 0; i < mtop->moltype.size(); ++i)
                    {
                        copy_moltype(&mtop->moltype[i], &temp[i]);
                    }
                    mtop->moltype.resize(mtop->moltype.size() + 1);
                    for (size_t i = 0; i < temp.size(); ++i)
                    {
                        copy_moltype(&temp[i], &mtop->moltype[i]);
                    }
                    copy_moltype(&mtop->moltype[molb->type], &mtop->moltype.back());

                    // Set the molecule type for the QMMM molblock
                    molb->type = mtop->moltype.size() - 1;
                }
            }
        }
    }

    // Call finalize() to rebuild Block Indicies or else atoms lookup will be screwed
    mtop->finalize();
}

void QMMMTopologyPreprocessor::removeQMClassicalCharges(gmx_mtop_t* mtop)
{
    // Loop over all atoms and remove charge if they are QM atoms.
    // Sum-up total removed charge and remaning charge on MM atoms
    // Build atomCharges_ vector
    int molb = 0;
    for (int i = 0; i < mtop->natoms; i++)
    {
        int indexInMolecule;
        mtopGetMolblockIndex(mtop, i, &molb, nullptr, &indexInMolecule);
        t_atom* atom = &mtop->moltype[mtop->molblock[molb].type].atoms.atom[indexInMolecule];
        if (isQMAtom(i))
        {
            topInfo_.qmQTot += atom->q;
            atom->q  = 0.0;
            atom->qB = 0.0;
        }
        else
        {
            topInfo_.mmQTot += atom->q;
        }

        atomCharges_.push_back(atom->q);
    }
}

void QMMMTopologyPreprocessor::addQMLJExclusions(gmx_mtop_t* mtop)
{
    // Add all QM atoms to the mtop->intermolecularExclusionGroup
    mtop->intermolecularExclusionGroup.reserve(mtop->intermolecularExclusionGroup.size()
                                               + qmIndices_.size());
    for (size_t i = 0; i < qmIndices_.size(); i++)
    {
        mtop->intermolecularExclusionGroup.push_back(qmIndices_[i]);
        topInfo_.numNb++;
    }
}

void QMMMTopologyPreprocessor::buildQMMMAtomNumbers(gmx_mtop_t* mtop)
{
    // Save to parameters_.atomNumbers_ atom numbers of all atoms
    AtomIterator atoms(*mtop);
    while (atoms->globalAtomNumber() < mtop->natoms)
    {
        // Check if we have valid atomnumbers
        if (atoms->atom().atomnumber < 0)
        {
            gmx_fatal(FARGS,
                      "Atoms %d does not have atomic number needed for QMMM. Check atomtypes "
                      "section in you topology or forcefield.",
                      atoms->globalAtomNumber());
        }

        atomNumbers_.push_back(atoms->atom().atomnumber);
        atoms++;
    }

    // Save in topInfo_ number of QM and MM atoms
    topInfo_.qmNum += static_cast<int>(qmIndices_.size());
    topInfo_.mmNum += static_cast<int>(mtop->natoms - qmIndices_.size());
}

void QMMMTopologyPreprocessor::modifyQMMMBonds(gmx_mtop_t* mtop)
{
    // Loop over all blocks in topology
    // mb - current block in mtop
    for (size_t mb = 0; mb < mtop->molblock.size(); mb++)
    {
        // check if current block contains QM atoms
        if (bQMBlock_[mb])
        {
            // molb - strucutre with current block
            gmx_molblock_t mlb = mtop->molblock[mb];
            // molt - strucutre with current block type
            gmx_moltype_t* mlt = &mtop->moltype[mlb.type];
            // start - first atom in current block
            int start = mtop->moleculeBlockIndices[mb].globalAtomStart;
            // loop over all interaction types
            for (int ftype = 0; ftype < F_NRE; ftype++)
            {
                // If not bonded interaction or F_CONNBONDS or some form of Restraints
                // or not pair interaction then go the next type
                if (!(interaction_function[ftype].flags & IF_BOND) || ftype == F_CONNBONDS
                    || ftype == F_RESTRBONDS || ftype == F_HARMONIC || ftype == F_DISRES || ftype == F_ORIRES
                    || ftype == F_ANGRESZ || interaction_function[ftype].nratoms != 2)
                {
                    continue;
                }

                // Loop over all interactions
                int j = 0;
                while (j < mlt->ilist[ftype].size())
                {
                    // Global indexes of atoms involved into the interaction
                    int a1 = mlt->ilist[ftype].iatoms[j + 1] + start;
                    int a2 = mlt->ilist[ftype].iatoms[j + 2] + start;

                    // If both atoms are QM then remove interaction
                    // if it was IF_CHEMBOND then also convert it to F_CONNBONDS
                    if (isQMAtom(a1) && isQMAtom(a2))
                    {
                        // Remove interaction
                        for (size_t k = j; k < mlt->ilist[ftype].iatoms.size() - 3; k++)
                        {
                            mlt->ilist[ftype].iatoms[k] = mlt->ilist[ftype].iatoms[k + 3];
                        }
                        mlt->ilist[ftype].iatoms.resize(mlt->ilist[ftype].iatoms.size() - 3);
                        topInfo_.rBonds++;

                        // Add chemical bond to the F_CONNBONDS (bond type 5)
                        if (IS_CHEMBOND(ftype))
                        {
                            mlt->ilist[F_CONNBONDS].iatoms.resize(mlt->ilist[F_CONNBONDS].iatoms.size() + 3);
                            mlt->ilist[F_CONNBONDS].iatoms[mlt->ilist[F_CONNBONDS].iatoms.size() - 3] = 5;
                            mlt->ilist[F_CONNBONDS].iatoms[mlt->ilist[F_CONNBONDS].iatoms.size() - 2] =
                                    a1 - start;
                            mlt->ilist[F_CONNBONDS].iatoms[mlt->ilist[F_CONNBONDS].iatoms.size() - 1] =
                                    a2 - start;
                            topInfo_.convBonds++;
                        }
                    }
                    else
                    {
                        j += 3;
                    }
                }
            }
        }
    }
}

void QMMMTopologyPreprocessor::buildQMMMLink(gmx_mtop_t* mtop)
{
    // Loop over all blocks in topology
    // mb - current block in mtop
    for (size_t mb = 0; mb < mtop->molblock.size(); mb++)
    {
        // check if current block contains QM atoms
        if (bQMBlock_[mb])
        {
            // molb - strucutre with current block
            gmx_molblock_t mlb = mtop->molblock[mb];
            // molt - strucutre with current block type
            gmx_moltype_t* mlt = &mtop->moltype[mlb.type];
            // start - first atom in current block
            int start = mtop->moleculeBlockIndices[mb].globalAtomStart;
            // loop over all interaction types
            for (int ftype = 0; ftype < F_NRE; ftype++)
            {
                // If not chemical bond interaction or not pair interaction then skip
                if (!(interaction_function[ftype].flags & IF_CHEMBOND)
                    || interaction_function[ftype].nratoms != 2)
                {
                    continue;
                }

                // Loop over all interactions in the current molblock
                int j = 0;
                while (j < mlt->ilist[ftype].size())
                {
                    // Global indexes of atoms involved into the interaction
                    int a1 = mlt->ilist[ftype].iatoms[j + 1] + start;
                    int a2 = mlt->ilist[ftype].iatoms[j + 2] + start;

                    // Update Link Frontier List if one of the atoms QM and one MM
                    if (isQMAtom(a1) && !isQMAtom(a2))
                    {
                        link_.push_back({ a1, a2 });
                        topInfo_.linkNum++;
                    }
                    if (isQMAtom(a2) && !isQMAtom(a1))
                    {
                        link_.push_back({ a2, a1 });
                        topInfo_.linkNum++;
                    }

                    // Check if it is constrained bond within QM subsystem
                    if (isQMAtom(a2) && isQMAtom(a1) && (interaction_function[ftype].flags & IF_CONSTRAINT))
                    {
                        topInfo_.numQMConstr++;
                    }

                    j += 3;
                }
            }
        }
    }
}

void QMMMTopologyPreprocessor::modifyQMMMAngles(gmx_mtop_t* mtop)
{
    // Loop over all blocks in topology
    // mb - current block in mtop
    for (size_t mb = 0; mb < mtop->molblock.size(); mb++)
    {
        // check if current block contains QM atoms
        if (bQMBlock_[mb])
        {
            // molb - strucutre with current block
            gmx_molblock_t mlb = mtop->molblock[mb];
            // molt - strucutre with current block type
            gmx_moltype_t* mlt = &mtop->moltype[mlb.type];
            // start - first atom in current block
            int start = mtop->moleculeBlockIndices[mb].globalAtomStart;
            // loop over all interaction types
            for (int ftype = 0; ftype < F_NRE; ftype++)
            {
                // If not bonded interaction or Restraints
                // or not three-particle interaction then go the next type
                if ((!(interaction_function[ftype].flags & IF_BOND) || ftype == F_RESTRANGLES
                     || interaction_function[ftype].nratoms != 3)
                    && (ftype != F_SETTLE))
                {
                    continue;
                }

                // Loop over all interactions
                int j = 0;
                while (j < mlt->ilist[ftype].size())
                {
                    // Calculate number of qm atoms in the interaction
                    int numQm = 0;
                    for (int k = 1; k <= 3; k++)
                    {
                        if (isQMAtom(mlt->ilist[ftype].iatoms[j + k] + start))
                        {
                            numQm++;
                        }
                    }

                    // If at least 2 atoms are QM then remove interaction
                    if (numQm >= 2)
                    {
                        // Add chemical bond to the F_CONNBONDS (bond type 5)
                        if (ftype == F_SETTLE)
                        {
                            mlt->ilist[F_CONNBONDS].iatoms.resize(mlt->ilist[F_CONNBONDS].iatoms.size() + 6);
                            mlt->ilist[F_CONNBONDS].iatoms[mlt->ilist[F_CONNBONDS].iatoms.size() - 6] = 5;
                            mlt->ilist[F_CONNBONDS].iatoms[mlt->ilist[F_CONNBONDS].iatoms.size() - 5] =
                                    mlt->ilist[ftype].iatoms[j + 1];
                            mlt->ilist[F_CONNBONDS].iatoms[mlt->ilist[F_CONNBONDS].iatoms.size() - 4] =
                                    mlt->ilist[ftype].iatoms[j + 2];
                            mlt->ilist[F_CONNBONDS].iatoms[mlt->ilist[F_CONNBONDS].iatoms.size() - 3] = 5;
                            mlt->ilist[F_CONNBONDS].iatoms[mlt->ilist[F_CONNBONDS].iatoms.size() - 2] =
                                    mlt->ilist[ftype].iatoms[j + 1];
                            mlt->ilist[F_CONNBONDS].iatoms[mlt->ilist[F_CONNBONDS].iatoms.size() - 1] =
                                    mlt->ilist[ftype].iatoms[j + 3];
                            topInfo_.convBonds += 2;
                            topInfo_.rSettle++;
                        }
                        else
                        {
                            topInfo_.rAngles++;
                        }

                        // Remove interaction
                        for (size_t k = j; k < mlt->ilist[ftype].iatoms.size() - 4; k++)
                        {
                            mlt->ilist[ftype].iatoms[k] = mlt->ilist[ftype].iatoms[k + 4];
                        }
                        mlt->ilist[ftype].iatoms.resize(mlt->ilist[ftype].iatoms.size() - 4);
                    }
                    else
                    {
                        j += 4;
                    }
                }
            }
        }
    }
}


void QMMMTopologyPreprocessor::modifyQMMMDihedrals(gmx_mtop_t* mtop)
{
    // Loop over all blocks in topology
    // mb - current block in mtop
    for (size_t mb = 0; mb < mtop->molblock.size(); mb++)
    {
        // check if current block contains QM atoms
        if (bQMBlock_[mb])
        {
            // molb - strucutre with current block
            gmx_molblock_t mlb = mtop->molblock[mb];
            // molt - strucutre with current block type
            gmx_moltype_t* mlt = &mtop->moltype[mlb.type];
            // start - first atom in current block
            int start = mtop->moleculeBlockIndices[mb].globalAtomStart;
            // loop over all interaction types
            for (int ftype = 0; ftype < F_NRE; ftype++)
            {
                // If not bonded interaction or Restraints
                // or not four-particle interaction then go the next type
                if (!(interaction_function[ftype].flags & IF_BOND) || ftype == F_RESTRDIHS
                    || interaction_function[ftype].nratoms != 4)
                {
                    continue;
                }

                // Loop over all interactions
                int j = 0;
                while (j < mlt->ilist[ftype].size())
                {
                    // Calculate number of qm atoms in the interaction
                    int numQm = 0;
                    for (int k = 1; k <= 4; k++)
                    {
                        if (isQMAtom(mlt->ilist[ftype].iatoms[j + k] + start))
                        {
                            numQm++;
                        }
                    }

                    // If at least 3 atoms are QM then remove interaction
                    if (numQm >= 3)
                    {
                        topInfo_.rDihedrals++;

                        // Remove interaction
                        for (size_t k = j; k < mlt->ilist[ftype].iatoms.size() - 5; k++)
                        {
                            mlt->ilist[ftype].iatoms[k] = mlt->ilist[ftype].iatoms[k + 5];
                        }
                        mlt->ilist[ftype].iatoms.resize(mlt->ilist[ftype].iatoms.size() - 5);
                    }
                    else
                    {
                        j += 5;
                    }
                }
            }
        }
    }
}

void QMMMTopologyPreprocessor::modifyQMMMVirtualSites(gmx_mtop_t* mtop)
{
    // Loop over all blocks in topology
    // mb - current block in mtop
    for (size_t mb = 0; mb < mtop->molblock.size(); mb++)
    {
        // check if current block contains QM atoms
        if (bQMBlock_[mb])
        {
            // molb - strucutre with current block
            gmx_molblock_t mlb = mtop->molblock[mb];
            // molt - strucutre with current block type
            gmx_moltype_t* mlt = &mtop->moltype[mlb.type];
            // start - first atom in current block
            int start = mtop->moleculeBlockIndices[mb].globalAtomStart;
            // loop over all interaction types
            for (int ftype = 0; ftype < F_NRE; ftype++)
            {
                if (IS_VSITE(ftype))
                {
                    // Loop over all interactions
                    int j = 0;
                    while (j < mlt->ilist[ftype].size())
                    {
                        // Calculate number of qm atoms in the interaction
                        int numQm = 0;
                        for (int k = 2; k <= interaction_function[ftype].nratoms; k++)
                        {
                            if (isQMAtom(mlt->ilist[ftype].iatoms[j + k] + start))
                            {
                                numQm++;
                            }
                        }

                        // If all atoms froming that virtual site are QM atoms
                        // then remove classical charge from that virtual site
                        if (numQm == (interaction_function[ftype].nratoms - 1))
                        {
                            topInfo_.rVSites++;
                            topInfo_.qmQTot += mlt->atoms.atom[mlt->ilist[ftype].iatoms[j + 1]].q;
                            mlt->atoms.atom[mlt->ilist[ftype].iatoms[j + 1]].q  = 0.0;
                            mlt->atoms.atom[mlt->ilist[ftype].iatoms[j + 1]].qB = 0.0;
                        }

                        j += interaction_function[ftype].nratoms + 1;
                    }
                }
            }
        }
    }
}

} // namespace gmx
