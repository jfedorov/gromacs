/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2010,2014,2015,2016,2017 by the GROMACS development team.
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
#ifndef GMX_MDLIB_MDATOMS_H
#define GMX_MDLIB_MDATOMS_H

#include <cstdio>

#include <memory>
#include <vector>

#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct gmx_mtop_t;
struct t_inputrec;
struct t_mdatoms;

enum class ParticleType : int;

namespace gmx
{

/*! \libinternal
 * \brief Contains Atom information used for efficient distribution.
 */
class MDAtoms
{
private:
    class Impl;
    //! Implementation detail of the datastructure.
    std::unique_ptr<Impl> impl_;

    MDAtoms();

public:
    ~MDAtoms();
    /*! \brief Resizes memory.
     *
     * \throws std::bad_alloc  If out of memory.
     */
    void resizeChargeA(int newSize);
    /*! \brief Resizes memory for state B charges.
     *
     * \throws std::bad_alloc  If out of memory.
     */
    void resizeChargeB(int newSize);
    /*! \brief
     * Reinitializes domain-local data after domain decomposition.
     *
     * For the masses the A-state (lambda=0) mass is used.
     * Sets md->lambda = 0.
     *
     * \param[in] mtop Topology data that holds legacy t_atoms.
     * \param[in] ir   Interaction definitions.
     * \param[in] index  If not empty, store those atoms only.
     * \param[in] homenr Number of atoms in this domain.
     */
    void reinitialize(const gmx_mtop_t& mtop, const t_inputrec& ir, ArrayRef<const int> index, int homenr);

    /*! \brief
     * Sets values for correct lambda state.
     *
     * When necessary, sets all the mass parameters to values corresponding
     * to the free-energy parameter \p lambda.
     *
     * \param[in] lambda Reaction parameter value to use.
     */
    void adjustToLambda(real lambda);

    //! Getter for size of members.
    int size() const;
    //! Getter for homenr.
    int homenr() const;
    //! Getter for number of energy groups.
    int nenergrp() const;
    //! Getter for number of perturbed charges.
    int nChargePerturbed() const;
    //! Getter for perturbed types.
    int nTypePerturbed() const;
    //! Getter for total mass
    double tmass() const;
    //! Getter for lambda.
    real lambda() const;
    //! Are there any perturbed charges?
    bool havePerturbedCharges() const;
    //! Are there any perturbed masses?
    bool havePerturbedMasses() const;
    //! Are there any perturbed types?
    bool havePerturbedTypes() const;
    //! Are there any perturbed interactions.
    bool havePerturbed() const;
    //! Number of perturbed interactions.
    int numPerturbed() const;
    //! Are there any vsites?
    bool haveVsites() const;
    //! Do we have partially frozen atoms?
    bool havePartiallyFrozenAtoms() const;
    //! Getter for atomic mass in A state.
    ArrayRef<const real> massA() const;
    //! Getter for atomic mass in B state.
    ArrayRef<const real> massB() const;
    //! Getter for atomic mass in present state.
    ArrayRef<const real> massT() const;
    //! Getter for inverse atomic mass per atom, 0 for vsites and shells
    ArrayRefWithPadding<const real> invmass() const;
    //! Getter for inverse atomic mass per atom and dimension.
    ArrayRef<const RVec> invMassPerDim() const;
    //! Getter for atomic charge in A state.
    ArrayRef<const real> chargeA() const;
    //! Getter for atomic charge in B state
    ArrayRef<const real> chargeB() const;
    //! Getter for dispersion constant C6 in A state.
    ArrayRef<const real> sqrt_c6A() const;
    //! Getter for dispersion constant C6 in A state.
    ArrayRef<const real> sqrt_c6B() const;
    //! Getter for van der Waals radius sigma in the A state.
    ArrayRef<const real> sigmaA() const;
    //! Getter for van der Waals radius sigma in the B state.
    ArrayRef<const real> sigmaB() const;
    //! Getter for van der Waals radius sigma^3 in the A state.
    ArrayRef<const real> sigma3A() const;
    //! Getter for van der Waals radius sigma^3 in the B state.
    ArrayRef<const real> sigma3B() const;
    //! Getter to check if this is atom perturbed.
    const std::vector<bool>& bPerturbed() const;
    //! Getter for type of atom in the A state.
    ArrayRef<const int> typeA() const;
    //! Getter for type of atom in the B state.
    ArrayRef<const int> typeB() const;
    //! Getter for particle type.
    ArrayRef<const ParticleType> ptype() const;
    //! Getter for group index for temperature coupling.
    ArrayRef<const unsigned short> cTC() const;
    //! Getter for group index for energy matrix.
    ArrayRef<const unsigned short> cENER() const;
    //! Getter for group index for acceleration.
    ArrayRef<const unsigned short> cACC() const;
    //! Getter for group index for freezing.
    ArrayRef<const unsigned short> cFREEZE() const;
    //! Getter for group index for center of mass motion removal
    ArrayRef<const unsigned short> cVCM() const;
    //! Getter for group index for user 1.
    ArrayRef<const unsigned short> cU1() const;
    //! Getter for group index for user 2.
    ArrayRef<const unsigned short> cU2() const;
    //! Getter for group index for orientation restraints.
    ArrayRef<const unsigned short> cORF() const;

    //! Builder function.
    friend std::unique_ptr<MDAtoms>
    makeMDAtoms(FILE* fp, const gmx_mtop_t& mtop, const t_inputrec& ir, bool rankHasPmeGpuTask);
};

//! Builder function for MdAtomsWrapper.
std::unique_ptr<MDAtoms> makeMDAtoms(FILE* fp, const gmx_mtop_t& mtop, const t_inputrec& ir, bool useGpuForPme);

} // namespace gmx

#endif
