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
 * Declares input generator class for CP2K QMMM
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_QMMMINPUTGENERATOR_H
#define GMX_APPLIED_FORCES_QMMMINPUTGENERATOR_H

#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/pbcutil/pbc.h"

#include "qmmmtypes.h"

namespace gmx
{

/*! \internal \brief
 * Class that takes QMMMParameters, Coordinates, Point charges, Box dimensions, pbcType.
 * Generates QM/MM sample input parameters and pdb-style coordinates for CP2K.
 * Input are generated as std::string objects which can be stored in tpr KVT
 * and/or flushed into the files.
 */
class QMMMInputGenerator
{
public:
    //! Construct QMMMInputGenerator from its parameters
    QMMMInputGenerator(const QMMMParameters        parameters,
                       const PbcType               pbcType,
                       const matrix                box,
                       const std::vector<real>&    q,
                       const ArrayRef<const RVec>& x);

    /*!\brief Generates sample CP2K input file
     *
     */
    std::string generateCP2KInput();

    /*!\brief Generates PDB file suitable for usage with CP2K.
     *  In that PDB file Point Charges of MM atoms are provided with Extended Beta field
     */
    std::string generateCP2KPdb();

    /*! \brief Returns computed QM box dimensions
     */
    const matrix& qmBox() const;

    /*! \brief Returns computed translation vector in order to center QM atoms inside QM box
     */
    const RVec& qmTrans() const;

private:
    /*!\brief Check if atom belongs to the global index of qmAtoms_
     */
    bool isQMAtom(index globalAtomIndex);

    /*!\brief Calculates dimensions, center of the QM box.
     *  Also evaluates translation for the system in order to center QM atoms inside QM box
     *  scale - how much QM box would be bigger than radius of QM system
     *  minL  - minimum norm of the QM box vector, default 1 nm (10 A)
     */
    void computeQMBox(const real scale = 1.5, const real minL = 1.0);

    //! QMMM Parameters structure
    const QMMMParameters parameters_;
    //! Simulation PbcType
    PbcType pbc_;
    //! Simulation Box
    matrix box_;
    //! QM box
    matrix qmBox_;
    //! PBC-aware center of QM subsystem
    RVec qmCenter_;
    //! Translation that shifts qmCenter_ to the center of qmBox_
    RVec qmTrans_;
    //! Atoms point charges
    const std::vector<real> q_;
    //! Atoms coordinates
    const ArrayRef<const RVec> x_;
};

} // namespace gmx

#endif // GMX_APPLIED_FORCES_QMMMINPUTGENERATOR_H
