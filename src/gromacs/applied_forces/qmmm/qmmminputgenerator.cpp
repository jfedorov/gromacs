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
 * Implements input generator class for CP2K QMMM
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include "qmmminputgenerator.h"

#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

QMMMInputGenerator::QMMMInputGenerator(const QMMMParameters        parameters,
                                       const PbcType               pbcType,
                                       const matrix                box,
                                       const std::vector<real>&    q,
                                       const ArrayRef<const RVec>& x) :
    parameters_(parameters),
    pbc_(pbcType),
    qmBox_{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } },
    qmCenter_{ 0.0, 0.0, 0.0 },
    qmTrans_{ 0.0, 0.0, 0.0 },
    q_(q),
    x_(x)
{
    copy_mat(box, box_);
    computeQMBox();
}

bool QMMMInputGenerator::isQMAtom(index globalAtomIndex)
{
    return std::find(parameters_.qmIndices_.begin(), parameters_.qmIndices_.end(), globalAtomIndex)
           != parameters_.qmIndices_.end();
}

void QMMMInputGenerator::computeQMBox(const real scale, const real minL)
{
    // Init atom numbers
    size_t nQm = parameters_.qmIndices_.size();

    // If there is only one QM atom, then just copy the box_
    if (nQm < 2)
    {
        copy_mat(box_, qmBox_);
        qmCenter_ = x_[parameters_.qmIndices_[0]];
        qmTrans_  = RVec(qmBox_[0]) / 2 + RVec(qmBox_[1]) / 2 + RVec(qmBox_[2]) / 2 - qmCenter_;
        return;
    }

    // Initialize pbc
    t_pbc pbc;
    set_pbc(&pbc, pbc_, box_);

    /* To compute qmBox_:
     * 1) Compute maximum dimension - maxDist betweeen QM atoms within PBC
     * 2) Make projection of the each box_ vector onto the cross prod of other two vectors
     * 3) Calculate scales so that norm will be maxDist for each box_ vector
     * 4) Apply scales to the box_ to get qmBox_
     */
    RVec dx(0.0, 0.0, 0.0);
    real maxDist = 0.0;

    // Search for the maxDist - maximum distance within QM system
    for (size_t i = 0; i < nQm - 1; i++)
    {
        for (size_t j = i + 1; j < nQm; j++)
        {
            pbc_dx(&pbc, x_[parameters_.qmIndices_[i]], x_[parameters_.qmIndices_[j]], dx);
            maxDist = dx.norm() > maxDist ? dx.norm() : maxDist;
        }
    }

    // Apply scale factor: qmBox_ should be *scale times bigger than maxDist
    maxDist *= scale;

    // Box vectors
    RVec vec0(box_[0]);
    RVec vec1(box_[1]);
    RVec vec2(box_[2]);

    // Compute scales sc0, sc1 and sc2 that will transform box_ into qmBox_
    dx = vec0.cross(vec1);
    dx /= dx.norm();
    real sc2 = maxDist / fabs(vec2.dot(dx));

    dx = vec1.cross(vec2);
    dx /= dx.norm();
    real sc0 = maxDist / fabs(vec0.dot(dx));

    dx = vec0.cross(vec2);
    dx /= dx.norm();
    real sc1 = maxDist / fabs(vec1.dot(dx));

    // Transform box_ into qmBox_ using computed sc0, sc1 and sc2
    svmul(sc0, box_[0], qmBox_[0]);
    svmul(sc1, box_[1], qmBox_[1]);
    svmul(sc2, box_[2], qmBox_[2]);

    // If some vector of the QM box is smaller than minL then scale it up
    if (norm(qmBox_[0]) < minL)
    {
        svmul(minL / norm(qmBox_[0]), qmBox_[0], qmBox_[0]);
    }
    if (norm(qmBox_[1]) < minL)
    {
        svmul(minL / norm(qmBox_[1]), qmBox_[1], qmBox_[1]);
    }
    if (norm(qmBox_[2]) < minL)
    {
        svmul(minL / norm(qmBox_[2]), qmBox_[2], qmBox_[2]);
    }

    // If now some vector of qmBox_ is longer than respective box_ vector then reset it
    if (norm(qmBox_[0]) > norm(box_[0]))
    {
        copy_rvec(box_[0], qmBox_[0]);
    }
    if (norm(qmBox_[1]) > norm(box_[1]))
    {
        copy_rvec(box_[1], qmBox_[1]);
    }
    if (norm(qmBox_[2]) > norm(box_[2]))
    {
        copy_rvec(box_[2], qmBox_[2]);
    }

    /* Now we need to also compute translation vector.
     * In order to center QM atoms in the computed qmBox_
     *
     * First compute center of QM system by averaging PBC-aware distance vectors
     * with respect to the first QM atom.
     */
    for (size_t i = 1; i < nQm; i++)
    {
        // compute pbc-aware distance vector between QM atom 0 and QM atom i
        pbc_dx(&pbc, x_[parameters_.qmIndices_[i]], x_[parameters_.qmIndices_[0]], dx);

        // add that to the qmCenter_
        qmCenter_ += dx;
    }

    // Average over all nQm atoms and add first atom coordiantes
    qmCenter_ = qmCenter_ / nQm + x_[parameters_.qmIndices_[0]];

    // Translation vector will be center of qmBox_ - qmCenter_
    qmTrans_ = RVec(qmBox_[0]) / 2 + RVec(qmBox_[1]) / 2 + RVec(qmBox_[2]) / 2 - qmCenter_;
}

std::string QMMMInputGenerator::generateCP2KInput()
{

    std::string inp = "";

    // Init some numbers
    size_t nQm = parameters_.qmIndices_.size();
    size_t nMm = parameters_.mmIndices_.size();
    size_t nAt = nQm + nMm;

    // Count the numbers of individual QM atoms per type
    std::vector<int> num_atoms(periodic_system.size(), 0);
    for (size_t i = 0; i < nQm; i++)
    {
        num_atoms[parameters_.atomNumbers_[parameters_.qmIndices_[i]]]++;
    }

    // Begin CP2K input generation

    inp += "&GLOBAL\n";
    inp += "  PRINT_LEVEL LOW\n";
    inp += "  PROJECT GROMACS\n";
    inp += "  RUN_TYPE ENERGY_FORCE\n";
    inp += "&END GLOBAL\n";
    inp += "&FORCE_EVAL\n";
    inp += "  METHOD QMMM\n";
    inp += "  &DFT\n";

    // write charge and multiplicity
    inp += formatString("    CHARGE %d\n", parameters_.qmCharge_);
    inp += formatString("    MULTIPLICITY %d\n", parameters_.qmMult_);

    // If multiplicity is not 1 then we should use unrestricted Kohn-Sham
    if (parameters_.qmMult_ > 1)
    {
        inp += "    UKS\n";
    }

    // Basis files, Grid setup and SCF parameters
    inp += "    BASIS_SET_FILE_NAME  BASIS_MOLOPT\n";
    inp += "    POTENTIAL_FILE_NAME  POTENTIAL\n";
    inp += "    &MGRID\n";
    inp += "      NGRIDS 5\n";
    inp += "      CUTOFF 450\n";
    inp += "      REL_CUTOFF 50\n";
    inp += "      COMMENSURATE\n";
    inp += "    &END MGRID\n";
    inp += "    &SCF\n";
    inp += "      SCF_GUESS RESTART\n";
    inp += "      EPS_SCF 5.0E-8\n";
    inp += "      MAX_SCF 20\n";
    inp += "      &OT  T\n";
    inp += "        MINIMIZER  DIIS\n";
    inp += "        STEPSIZE   0.15\n";
    inp += "        PRECONDITIONER FULL_ALL\n";
    inp += "      &END OT\n";
    inp += "      &OUTER_SCF  T\n";
    inp += "        MAX_SCF 20\n";
    inp += "        EPS_SCF 5.0E-8\n";
    inp += "      &END OUTER_SCF\n";
    inp += "    &END SCF\n";

    // DFT functional parameters
    inp += "    &XC\n";
    inp += "      DENSITY_CUTOFF     1.0E-12\n";
    inp += "      GRADIENT_CUTOFF    1.0E-12\n";
    inp += "      TAU_CUTOFF         1.0E-12\n";
    inp += formatString("      &XC_FUNCTIONAL %s\n", c_qmmmQMMethodNames[parameters_.qmMethod_]);
    inp += "      &END XC_FUNCTIONAL\n";
    inp += "    &END XC\n";
    inp += "    &QS\n";
    inp += "     METHOD GPW\n";
    inp += "     EPS_DEFAULT 1.0E-10\n";
    inp += "     EXTRAPOLATION ASPC\n";
    inp += "     EXTRAPOLATION_ORDER  4\n";
    inp += "    &END QS\n";
    inp += "  &END DFT\n";

    // QMMM parameters
    inp += "  &QMMM\n";

    // QM cell
    inp += "    &CELL\n";
    inp += formatString("      A %.3lf %.3lf %.3lf\n", qmBox_[0][0] * 10, qmBox_[0][1] * 10,
                        qmBox_[0][2] * 10);
    inp += formatString("      B %.3lf %.3lf %.3lf\n", qmBox_[1][0] * 10, qmBox_[1][1] * 10,
                        qmBox_[1][2] * 10);
    inp += formatString("      C %.3lf %.3lf %.3lf\n", qmBox_[2][0] * 10, qmBox_[2][1] * 10,
                        qmBox_[2][2] * 10);
    inp += "      PERIODIC XYZ\n";
    inp += "    &END CELL\n";

    inp += "    CENTER EVERY_STEP\n";
    inp += "    CENTER_GRID TRUE\n";
    // inp += "    CENTER NEVER\n";
    inp += "    &WALLS\n";
    inp += "      TYPE REFLECTIVE\n";
    inp += "    &END WALLS\n";

    inp += "    ECOUPL GAUSS\n";
    inp += "    USE_GEEP_LIB 12\n";
    inp += "    &PERIODIC\n";
    inp += "      GMAX     1.0E+00\n";
    inp += "      &MULTIPOLE ON\n";
    inp += "         RCUT     1.0E+01\n";
    inp += "         EWALD_PRECISION     1.0E-06\n";
    inp += "      &END\n";
    inp += "    &END PERIODIC\n";

    // Print indicies of QM atoms
    // Loop over counter of QM atom types
    for (size_t i = 0; i < num_atoms.size(); i++)
    {
        if (num_atoms[i] > 0)
        {
            inp += formatString("    &QM_KIND %3s\n", periodic_system[i].c_str());
            inp += "      MM_INDEX";
            // Loop over all QM atoms indexes
            for (size_t j = 0; j < nQm; j++)
            {
                if (parameters_.atomNumbers_[parameters_.qmIndices_[j]] == static_cast<index>(i))
                {
                    inp += formatString(" %d", static_cast<int>(parameters_.qmIndices_[j] + 1));
                }
            }

            inp += "\n";
            inp += "    &END QM_KIND\n";
        }
    }

    // Print &LINK groups
    // Loop over parameters_.link_
    for (size_t i = 0; i < parameters_.link_.size(); i++)
    {
        inp += "    &LINK\n";
        inp += formatString("      QM_INDEX %d\n", static_cast<int>(parameters_.link_[i].qm) + 1);
        inp += formatString("      MM_INDEX %d\n", static_cast<int>(parameters_.link_[i].mm) + 1);
        inp += "    &END LINK\n";
    }

    inp += "  &END QMMM\n";
    inp += "  &MM\n";
    inp += "    &FORCEFIELD\n";
    inp += "      DO_NONBONDED FALSE\n";
    inp += "    &END FORCEFIELD\n";
    inp += "    &POISSON\n";
    inp += "      &EWALD\n";
    inp += "        EWALD_TYPE NONE\n";
    inp += "      &END EWALD\n";
    inp += "    &END POISSON\n";
    inp += "  &END MM\n";

    inp += "  &SUBSYS\n";

    // Print cell parameters
    inp += "    &CELL\n";
    inp += formatString("      A %.3lf %.3lf %.3lf\n", box_[0][0] * 10, box_[0][1] * 10, box_[0][2] * 10);
    inp += formatString("      B %.3lf %.3lf %.3lf\n", box_[1][0] * 10, box_[1][1] * 10, box_[1][2] * 10);
    inp += formatString("      C %.3lf %.3lf %.3lf\n", box_[2][0] * 10, box_[2][1] * 10, box_[2][2] * 10);
    inp += "      PERIODIC XYZ\n";
    inp += "    &END CELL\n";

    // Print topology section
    inp += "    &TOPOLOGY\n";

    // pdb file name
    inp += "      COORD_FILE_NAME "
           + parameters_.qmInputFileName_.substr(0, parameters_.qmInputFileName_.find_last_of("."))
           + ".pdb" + "\n";

    inp += "      COORD_FILE_FORMAT PDB\n";
    inp += "      CHARGE_EXTENDED TRUE\n";
    inp += "      CONNECTIVITY OFF\n";
    inp += "      &GENERATE\n";
    inp += "         &ISOLATED_ATOMS\n";
    inp += formatString("            LIST %d..%d\n", 1, static_cast<int>(nAt));
    inp += "         &END\n";
    inp += "      &END GENERATE\n";
    inp += "    &END TOPOLOGY\n";

    // Now we will print basises for all types of QM atoms
    // Loop over counter of QM atom types
    for (size_t i = 0; i < num_atoms.size(); i++)
    {
        if (num_atoms[i] > 0)
        {
            inp += "    &KIND " + periodic_system[i] + "\n";
            inp += "      ELEMENT " + periodic_system[i] + "\n";
            inp += "      BASIS_SET DZVP-MOLOPT-GTH\n";
            inp += "      POTENTIAL GTH-" + std::string(c_qmmmQMMethodNames[parameters_.qmMethod_]);
            inp += "\n";
            inp += "    &END KIND\n";
        }
    }

    // Add element kind X - they are represents virtual sites
    inp += "    &KIND X\n";
    inp += "      ELEMENT H\n";
    inp += "    &END KIND\n";
    inp += "  &END SUBSYS\n";
    inp += "&END FORCE_EVAL\n";

    return inp;
}

std::string QMMMInputGenerator::generateCP2KPdb()
{

    std::string pdb = "";

    /* Generate *.pdb formatted lines
     * and append to std::string pdb
     */
    for (size_t i = 0; i < x_.size(); i++)
    {
        pdb += "ATOM  ";

        // Here we need to print i % 100000 because atom counter in *.pdb has only 5 digits
        pdb += formatString("%5d ", static_cast<int>(i % 100000));

        // Atom name
        pdb += formatString(" %3s ", periodic_system[parameters_.atomNumbers_[i]].c_str());

        // Lable atom as QM or MM residue
        if (isQMAtom(i))
        {
            pdb += " QM     1     ";
        }
        else
        {
            pdb += " MM     2     ";
        }

        // Coordinates
        pdb += formatString("%7.3lf %7.3lf %7.3lf  1.00  0.00         ", (x_[i][XX] + qmTrans_[XX]) * 10,
                            (x_[i][YY] + qmTrans_[YY]) * 10, (x_[i][ZZ] + qmTrans_[ZZ]) * 10);

        // Atom symbol
        pdb += formatString(" %3s ", periodic_system[parameters_.atomNumbers_[i]].c_str());

        // Point charge for MM atoms or 0 for QM atoms
        pdb += formatString("%lf\n", q_[i]);
    }

    return pdb;
}

const RVec& QMMMInputGenerator::qmTrans() const
{
    return qmTrans_;
}

const matrix& QMMMInputGenerator::qmBox() const
{
    return qmBox_;
}

} // namespace gmx
