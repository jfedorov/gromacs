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
 * Implements force provider for QMMM
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include "qmmmforceprovider.h"

#include "config.h"

#include <fstream>
#include <sstream>

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/math/multidimarray.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/exceptions.h"

namespace gmx
{

QMMMForceProvider::QMMMForceProvider(const QMMMParameters& parameters,
                                     const LocalAtomSet&   localQMAtomSet,
                                     const LocalAtomSet&   localMMAtomSet,
                                     PbcType               pbcType,
                                     const MDLogger&       logger) :
    parameters_(parameters),
    qmAtoms_(localQMAtomSet),
    mmAtoms_(localMMAtomSet),
    pbcType_(pbcType),
    logger_(logger),
    box_{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } }
{
    // If CP2K is not linked but QMMM simulation is requested then throw an Error.
    if (parameters_.active_ && GMX_CP2K == 0)
    {
        GMX_THROW(InternalError(
                "CP2K has not been linked into GROMACS, QMMM simulation is not possible.\n"));
    }
}

QMMMForceProvider::~QMMMForceProvider()
{
    if constexpr (GMX_CP2K == 1)
    {
        cp2k_destroy_force_env(force_env_);
        if constexpr (GMX_LIB_MPI == 1)
        {
            cp2k_finalize_without_mpi();
        }
        else
        {
            cp2k_finalize();
        }
    }
}

bool QMMMForceProvider::isQMAtom(index globalAtomIndex)
{
    return std::find(qmAtoms_.globalIndex().begin(), qmAtoms_.globalIndex().end(), globalAtomIndex)
           != qmAtoms_.globalIndex().end();
}

void QMMMForceProvider::appendLog(const std::string msg)
{
    GMX_LOG(logger_.info).asParagraph().appendText(msg);
}

void QMMMForceProvider::initQMMMforce(const t_commrec& cr)
{

    // write CP2K input if we are Master
    if (MASTER(&cr))
    {
        // check existence of parameters_.qmInputFileName_ file
        std::string   fName = parameters_.qmInputFileName_;
        std::ifstream inpFile(fName);
        bool          inpFileExists = inpFile.good();
        inpFile.close();

        if (parameters_.qmMethod_ == QMMMQMMethod::INPUT)
        {
            // CP2K input should be provided by user, throw an error if it is not found
            if (!inpFileExists)
            {
                std::string msg = "Requested qmmm-qmmethod = INPUT, but qmmm-qminputfile = " 
                                  + parameters_.qmInputFileName_ + " not found. " 
                                  + "Please make sure that QM input file was copied to the simulation working directory\n";
                GMX_THROW(FileIOError(msg.c_str()));
            }
        }
        else
        {
            // CP2K input should been generated during grompp and stored into file and/or into *.tpr
            if (!inpFileExists)
            {
                // Input file not found inform user about that
                std::string msg = "\nQMMM Note: CP2K input file " + fName
                                  + " not exists. It will be regenerated from the saved data.\n\n";
                appendLog(msg);

                // And regenerate *.inp and *.pdb from the stored in parameters_
                std::ofstream fInp(parameters_.qmInputFileName_);
                fInp << parameters_.qmInput_;
                fInp.close();
                std::ofstream fPdb(fName.substr(0, fName.find_last_of(".")) + ".pdb");
                fPdb << parameters_.qmPdb_;
                fPdb.close();
            }
            else
            {
                // Check if existing file is the same as saved version and inform user
                std::ifstream     inpFile(fName);
                std::stringstream strStream;
                strStream << inpFile.rdbuf();
                inpFile.close();

                if (strStream.str() != parameters_.qmInput_)
                {
                    // Print message if they are not the same
                    std::string msg =
                            "\nQMMM Warning: CP2K input file " + fName
                            + " has been changed after grompp.\n"
                            + "This could affect reproducibility of the simulation.\n"
                            + "Consider to regenerate tpr file using qmmm-qmmethod = INPUT "
                            + "and providing your custom CP2K input file.\n\n";
                    appendLog(msg);
                }
            }
        }
    }

    // Attempt to init CP2K and create force_env
    std::string fInp = parameters_.qmInputFileName_;
    std::string fOut = fInp.substr(0, fInp.find_last_of(".")) + ".out";
#if GMX_LIB_MPI
    if constexpr (GMX_CP2K == 1)
    {
        cp2k_init_without_mpi();
        cp2k_create_force_env_comm(&force_env_, fInp.c_str(), fOut.c_str(),
                                   MPI_Comm_c2f(cr.mpi_comm_mysim));
    }
#else
    // CP2K could not work with thread-MPI in case it is used throw an error
    if (cr.nnodes > 1)
    {
        std::string msg =
                "CP2K could not use thread-MPI. Use OpenMP parallelization for single-node "
                "version "
                "(\"mdrun -ntomp\" option) or compile GROMACS with external MPI library.\n";
        GMX_THROW(NotImplementedError(msg.c_str()));
    }

    if constexpr (GMX_CP2K == 1)
    {
        cp2k_init();
        cp2k_create_force_env(&force_env_, fInp.c_str(), fOut.c_str());
    }
#endif

} // namespace gmx

void QMMMForceProvider::calculateForces(const ForceProviderInput& fInput, ForceProviderOutput* fOutput)
{
    // Save number of atoms into the nat
    size_t nAt = qmAtoms_.numAtomsGlobal() + mmAtoms_.numAtomsGlobal();

    // Save box
    copy_mat(fInput.box_, box_);

    // Initialize PBC
    t_pbc pbc;
    set_pbc(&pbc, pbcType_, box_);

    /*
     * 1) We need to gather fInput.x_ in case of MPI / DD setup
     */

    // x - coordinates (gathered across nodes in case of DD)
    std::vector<RVec> x(nAt, RVec({ 0.0, 0.0, 0.0 }));
    // Fill in local cordinates of QM atoms
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {
        x[qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]] = fInput.x_[qmAtoms_.localIndex()[i]];
    }
    // Fill in local cordinates of MM atoms
    for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
    {
        x[mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]] = fInput.x_[mmAtoms_.localIndex()[i]];
    }
    // If we are in MPI / DD conditions then gather coordinates over nodes
    if (havePPDomainDecomposition(&fInput.cr_))
    {
        gmx_sum(3 * nAt, &x[0][XX], &fInput.cr_);
    }

    /*
     * 2) If calculateForce called first time, then we need to init some stuff
     */
    if (!step_)
    {
        try
        {
            initQMMMforce(fInput.cr_);
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
    }

    // Wrap all atoms back to the cell 0
    // ArrayRef<RVec> x_wrap(&x.front(),&x.back());
    // put_atoms_in_box(pbcType_, fInput.box_, x_wrap);

    /*
     * 3) Cast data to double format of libcp2k
     * update coordinates and box in CP2K and perform QM calculation
     */
    // x_d - coordinates casted to linear dobule vector for cp2k with parameters_.qmTrans_ added
    std::vector<double> x_d(3 * nAt, 0.0);
    for (size_t i = 0; i < nAt; i++)
    {
        x_d[3 * i]     = static_cast<double>((x[i][XX] + parameters_.qmTrans_[XX]) / BOHR2NM);
        x_d[3 * i + 1] = static_cast<double>((x[i][YY] + parameters_.qmTrans_[YY]) / BOHR2NM);
        x_d[3 * i + 2] = static_cast<double>((x[i][ZZ] + parameters_.qmTrans_[ZZ]) / BOHR2NM);
    }

    // box_d - box_ casted to linear dobule vector for CP2K
    std::vector<double> box_d(9);
    for (size_t i = 0; i < DIM; i++)
    {
        box_d[3 * i]     = static_cast<double>(box_[0][i] / BOHR2NM);
        box_d[3 * i + 1] = static_cast<double>(box_[1][i] / BOHR2NM);
        box_d[3 * i + 2] = static_cast<double>(box_[2][i] / BOHR2NM);
    }

    // Update coordinates and box in CP2K
    if constexpr (GMX_CP2K == 1)
    {
        cp2k_set_positions(force_env_, &x_d[0], 3 * nAt);
        cp2k_set_cell(force_env_, &box_d[0]);
    }
    if (MASTER(&fInput.cr_))
        fprintf(stderr, "cp2k_set_positions & cp2k_set_cell - DONE\n");

    // Run cp2k
    if constexpr (GMX_CP2K == 1)
    {
        cp2k_calc_energy_force(force_env_);
    }
    if (MASTER(&fInput.cr_))
        fprintf(stderr, "cp2k_calc_energy_force - DONE\n");

    /*
     * 4) Get output data
     * We need to fill only local part into fOutput!
     */

    // Get QM + QM-MM Energy
    double qmEner = 0.0;
    if constexpr (GMX_CP2K == 1)
    {
        cp2k_get_potential_energy(force_env_, &qmEner);
    }
    if (MASTER(&fInput.cr_))
        fprintf(stderr, "QMener=%.12lf\n", qmEner);

    // Get Gradient
    std::vector<double> grd(3 * nAt, 0.0);
    if constexpr (GMX_CP2K == 1)
    {
        cp2k_get_forces(force_env_, &grd[0], 3 * nAt);
    }

    // convert energy and gradients into the Gromacs units
    if (MASTER(&fInput.cr_))
    {
        fOutput->enerd_.term[F_EQM] += qmEner * HARTREE2KJ * AVOGADRO;
    }

    // forces on QM atoms first
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {
        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][XX] +=
                static_cast<real>(grd[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]])
                * HARTREE_BOHR2MD;

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][YY] +=
                static_cast<real>(grd[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1])
                * HARTREE_BOHR2MD;

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][ZZ] +=
                static_cast<real>(grd[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2])
                * HARTREE_BOHR2MD;
    }

    // forces on MM atoms then
    for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
    {
        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][XX] +=
                static_cast<real>(grd[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]])
                * HARTREE_BOHR2MD;

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][YY] +=
                static_cast<real>(grd[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1])
                * HARTREE_BOHR2MD;

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][ZZ] +=
                static_cast<real>(grd[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2])
                * HARTREE_BOHR2MD;
    }

    // increase internal step counter
    step_++;
};

} // namespace gmx
