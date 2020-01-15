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
 * Declares functionality needed from libcp2k to work with QMMM MdModule
 * Partialy taken from CP2K: A general program to perform molecular dynamics simulations
 * Copyright 2000-2020 CP2K developers group <https://cp2k.org>
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \author Mohammad Hossein Bani-Hashemian
 * \ingroup module_applied_forces
 */

#include <stdbool.h>

#ifndef LIBCP2K_H
#    define LIBCP2K_H

#    ifdef __cplusplus
extern "C"
{
#    endif

    typedef int force_env_t;

    /*******************************************************************************
     * \brief Get the CP2K version string
     * \param version_str The buffer to write the version string into
     * \param str_length The size of the buffer (must be large enough)
     ******************************************************************************/
    void cp2k_get_version(char* version_str, int str_length);

    /*******************************************************************************
     * \brief Initialize CP2K and MPI
     * \warning You are supposed to call cp2k_finalize() before exiting the program.
     ******************************************************************************/
    void cp2k_init();

    /*******************************************************************************
     * \brief Initialize CP2K without initializing MPI
     * \warning You are supposed to call cp2k_finalize() before exiting the program.
     ******************************************************************************/
    void cp2k_init_without_mpi();

    /*******************************************************************************
     * \brief Finalize CP2K and MPI
     ******************************************************************************/
    void cp2k_finalize();

    /*******************************************************************************
     * \brief Finalize CP2K and without finalizing MPI
     ******************************************************************************/
    void cp2k_finalize_without_mpi();

    /*******************************************************************************
     * \brief Create a new force environment
     * \param new_force_env the created force environment
     * \param input_file_path Path to a CP2K input file
     * \param output_file_path Path to a file where CP2K is going to append its
     *                         output (created if non-existent)
     * \warning You are supposed to call cp2k_destroy_force_env() to cleanup,
     *          before cp2k_finalize().
     ******************************************************************************/
    void cp2k_create_force_env(force_env_t* new_force_env,
                               const char*  input_file_path,
                               const char*  output_file_path);

    /*******************************************************************************
     * \brief Create a new force environment (custom managed MPI)
     * \param new_force_env the created force environment
     * \param input_file_path Path to a CP2K input file
     * \param output_file_path Path to a file where CP2K is will write its output.
     *                         Will be created if not existent, otherwise appended.
     * \param mpi_comm Fortran MPI communicator if MPI is not managed by CP2K
     * \warning You are supposed to call cp2k_destroy_force_env() to cleanup,
     *          before cp2k_finalize().
     ******************************************************************************/
    void cp2k_create_force_env_comm(force_env_t* new_force_env,
                                    const char*  input_file_path,
                                    const char*  output_file_path,
                                    int          mpi_comm);

    /*******************************************************************************
     * \brief Destroy/cleanup a force environment
     * \param force_env the force environment
     ******************************************************************************/
    void cp2k_destroy_force_env(force_env_t force_env);

    /*******************************************************************************
     * \brief Set positions of the particles
     * \param force_env the force environment
     * \param new_pos Array containing the new positions of the particles
     * \param n_el Size of the new_pos array
     ******************************************************************************/
    void cp2k_set_positions(force_env_t force_env, const double* new_pos, int n_el);

    /*******************************************************************************
     * \brief Set the size of the cell
     * \param force_env the force environment
     * \param new_cell Array containing the new cell
     ******************************************************************************/
    void cp2k_set_cell(force_env_t force_env, const double* new_cell);

    /*******************************************************************************
     * \brief Get the forces for the particles
     * \param force_env the force environment
     * \param force Pre-allocated array of at least 3*nparticle elements.
     *              Use cp2k_get_nparticle() to get the number of particles.
     * \param n_el Size of the force array
     ******************************************************************************/
    void cp2k_get_forces(force_env_t force_env, double* force, int n_el);

    /*******************************************************************************
     * \brief Get the potential energy of the system
     * \param force_env the force environment
     * \param e_pot The potential energy
     ******************************************************************************/
    void cp2k_get_potential_energy(force_env_t force_env, double* e_pot);

    /*******************************************************************************
     * \brief Calculate energy and forces of the system
     * \param force_env the force environment
     ******************************************************************************/
    void cp2k_calc_energy_force(force_env_t force_env);

#    ifdef __cplusplus
}
#    endif

#endif
