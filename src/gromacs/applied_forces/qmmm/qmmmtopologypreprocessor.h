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
 * QMMMTopologyPrepocessor class responsible for
 * all modificatios of the topology during input pre-processing
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_QMMMTOPOLOGYPREPROCESSOR_H
#define GMX_APPLIED_FORCES_QMMMTOPOLOGYPREPROCESSOR_H

#include <string>

#include "qmmmtypes.h"

struct gmx_mtop_t;

namespace gmx
{

/*! \internal
 * \brief Contains various information about topology modifications
 * Used for statistics during topology pre-processing within QMMMTopologyPreprocessor class
 */
struct QMMMTopologyInfo
{
    //! Total number of MM atoms
    int mmNum = 0;
    //! Total number of QM atoms
    int qmNum = 0;
    //! Total remaining charge of MM part
    real mmQTot = 0.0;
    //! Total classical charge removed from QM atoms
    real qmQTot = 0.0;
    //! Total number of Non-bonded (LJ) exclusions made for QM-QM interactions
    int numNb = 0;
    //! Total number of removed classical Bonds between QM-QM atoms
    int rBonds = 0;
    //! Total number of removed classical Angles between QM-QM atoms
    int rAngles = 0;
    //! Total number of removed classical Dihedrals between QM-QM atoms
    int rDihedrals = 0;
    //! Total number of removed F_SETTLE between QM-QM atoms
    int rSettle = 0;
    //! Total number of empty chemical bonds (F_CONNBONDS) added between QM-QM atoms
    int convBonds = 0;
    //! Total number of virtual sites, that consisting of QM atoms only 
    int rVSites = 0;
    //! Total number of constrained bonds within QM subsystem
    int numQMConstr = 0;
    //! Total number of broken bonds between QM and MM atoms (Link Frontier)
    int linkNum = 0;
};

/*! \internal
 * \brief Class implementing gmx_mtop_t QMMM modifications during preprocessing
 * 1) Split QM-containing molecules from other molecules in blocks
 * 1.5) Nullify charges on all virtual sites consisting of QM only atoms
 * 2) Nullifies charges on all QM atoms
 * 3) Excludes LJ interactions between QM atoms
 * 4) Builds vector with atomic numbers of all atoms
 * 5) Makes F_CONNBOND between atoms within QM region
 * 6) Removes angles containing 2 or more QM atoms
 * 7) Removes all settles containing any number of QM atoms
 * 8) Removes dihedrals containing 3 or more QM atoms
 * 9) Builds vector containing pairs of bonded QM - MM atoms (Link frontier)
 */
class QMMMTopologyPreprocessor
{
public:
    /*! \brief Constructor for QMMMTopologyPreprocessor from its parameters
     * Pocesses mtop topology and prepares atomNumbers_ and link_ vectors
     * Builds topInfo_ containing information about topology modifications
     */
    QMMMTopologyPreprocessor(gmx_mtop_t* mtop, const std::vector<index> qmIndices);

    /*! \brief Returns data about modifications made via QMMMTopologyInfo
     */
    const QMMMTopologyInfo& topInfo() const;

    /*! \brief Returns vector with All Atoms Numbers for the processed topology
     */
    const std::vector<int>& atomNumbers() const;

    /*! \brief Returns vector with All Atoms point charges for the processed topology
     */
    const std::vector<real>& atomCharges() const;

    /*! \brief Returns vector with Link Frontier fot the processed topology
     */
    const std::vector<LinkFrontier>& link() const;

private:
    //! Retruns true is globalAtomIndex bewlongs to QM region
    bool isQMAtom(index globalAtomIndex);

    /*! \brief Splits QM containing molecules out of MM blocks in topology
     * Modifies blocks in topology
     * Updates bQMBlock vector containing QM flags of all blocks in modified mtop
     */
    void splitQMblocks(gmx_mtop_t* mtop);

    /*! \brief Removes classical charges from QM atoms
     * Provides data about removed charge via topInfo_
     */
    void removeQMClassicalCharges(gmx_mtop_t* mtop);

    //! \brief Build exlusion list for LJ interactions between QM atoms
    void addQMLJExclusions(gmx_mtop_t* mtop);

    /*! \brief Builds atomNumbers_ vector
     * Provides data about total number of QM and MM atoms via topInfo_
     */
    void buildQMMMAtomNumbers(gmx_mtop_t* mtop);

    /*! \brief Modifies pairwise bonded interactions
     * Creates F_CONNBOND between QM atoms
     * Removes any other pairwise bonded interactions between QM-QM atoms
     * Any restraints and constraints will be kept
     * Provides some data about Cleaning via topInfo_
     */
    void modifyQMMMBonds(gmx_mtop_t* mtop);

    /*! \brief Builds link_ vector with pairs of atoms indicting broken QM - MM chemical bonds.
    * Also performs search of constrained bonds within QM subsystem.
    */
    void buildQMMMLink(gmx_mtop_t* mtop);

    /*! \brief Modifies three-particle interactions
     * Removes any other three-particle bonded interactions including 2 or more QM atoms
     * Any restraints and constraints will be kept
     * Any F_SETTLE containing QM atoms will be converted to the pair of F_CONNBONDS
     * Provides some data about Cleaning via topInfo_
     */
    void modifyQMMMAngles(gmx_mtop_t* mtop);

    /*! \brief Modifies four-particle interactions
     * Removes any other four-particle bonded interactions including 3 or more QM atoms
     * Any restraints and constraints will be kept
     * Provides some data about Cleaning via topInfo_
     */
    void modifyQMMMDihedrals(gmx_mtop_t* mtop);

    /*! \brief Removes charge from all virtual sites which are consists of only QM atoms
     */
    void modifyQMMMVirtualSites(gmx_mtop_t* mtop);

    //! Vector indicating which molblocks are QM
    std::vector<bool> bQMBlock_;
    //! Global indices of QM atoms;
    std::vector<index> qmIndices_;
    //! Vector with atom numbers for the whole system
    std::vector<int> atomNumbers_;
    //! Vector with atom point charges for the whole system
    std::vector<real> atomCharges_;
    //! Vector with pairs of indicies defining broken bonds in QMMM
    std::vector<LinkFrontier> link_;
    //! Structure with information about modifications made
    QMMMTopologyInfo topInfo_;
};

} // namespace gmx

#endif
