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
 * Declares options for QM/MM
 * QMMMOptions class responsible for all parameters set up during pre-processing
 * also modificatios of topology would be done here
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_QMMMOPTIONS_H
#define GMX_APPLIED_FORCES_QMMMOPTIONS_H

#include "gromacs/mdtypes/imdpoptionprovider.h"

#include "qmmmtypes.h"

struct gmx_mtop_t;

namespace gmx
{

class IndexGroupsAndNames;
class KeyValueTreeObject;
class KeyValueTreeBuilder;
class MDLogger;
struct MdRunInputFilename;
struct CoordinatesAndBoxPreprocessed;

/*! \internal
 * \brief Input data storage for QM/MM
 */
class QMMMOptions final : public IMdpOptionProvider
{
public:
    //! From IMdpOptionProvider
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override;

    /*! \brief
     * Build mdp parameters for QMMM to be output after pre-processing.
     * \param[in, out] builder the builder for the mdp options output KV-tree.
     * \note This should be symmetrical to option initialization without
     *       employing manual prefixing with the section name string once
     *       the legacy code blocking this design is removed.
     */
    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override;

    /*! \brief
     * Connect option name and data.
     */
    void initMdpOptions(IOptionsContainerWithSections* options) override;

    //! Report if this set of options is active
    bool active() const;

    //! Process input options to parameters, including input file reading.
    const QMMMParameters& buildParameters();

    /*! \brief Evaluate and store atom indices.
     * During pre-processing, use the group string from the options to
     * evaluate the indices of the atoms to be subject to forces from this
     * module.
     */
    void setQMMMGroupIndices(const IndexGroupsAndNames& indexGroupsAndNames);

    /*! \brief Sets names for CP2K input/output files and PDB file
     * Choses between default "cp2k", qmmm-qminputfile (provided in mdp) and *.tpr filename
     */
    void setQMFileName(MdRunInputFilename tprName);

    /*! \brief Process coordinates, PbcType and Box in order to produce CP2K sample input
     * Produces parameters_.qmInput_ and qmPdb_ fields.
     */
    void processCoordinates(const CoordinatesAndBoxPreprocessed coord);

    /*! \brief Modifies topology in case of active QMMM module.
     *
     * The following modifications should be made:
     * 1) Charges on all QM atoms should be nulified
     * 2) Make bonds type 5 between QM atoms and nulify any bond constants
     * 3) Angles (with 2 or more QM atoms) and Dihedrals (with 3 or more QM atoms) should be excluded from topology
     * 4) Extract and save in KVT atomic numbers of all atoms
     * 5) Save to KVT two indexes containing pairs of bonded QM - MM atoms (Link frontier)
     */
    void modifyQMMMTopology(gmx_mtop_t* mtop);

    //! Store the paramers that are not mdp options in the tpr file
    void writeInternalParametersToKvt(KeyValueTreeObjectBuilder treeBuilder);

    //! Set the internal parameters that are stored in the tpr file
    void readInternalParametersFromKvt(const KeyValueTreeObject& tree);

    //! Set the MDLogger instance
    void setLogger(const MDLogger& logger);

private:
    //! Write message to the log
    void appendLog(const std::string msg);

    /*! \brief Following Tags denotes names of parameters from .mdp file
     * \note Changing this strings will break .tpr backwards compability
     */
    const std::string c_activeTag_              = "active";
    const std::string c_qmGroupTag_             = "qmgroup";
    const std::string c_mmGroupTag_             = "mmgroup";
    const std::string c_qmChargeTag_            = "qmcharge";
    const std::string c_qmMultTag_              = "qmmultiplicity";
    const std::string c_qmMethodTag_            = "qmmethod";
    const std::string c_qmUserInputFileNameTag_ = "qminputfile";

    /*! \brief This tags for parameters which will be generated during grompp
     * and stored into *.tpr file via KVT
     */
    const std::string c_atomNumbersTag_     = "atomnumbers";
    const std::string c_qmLinkTag_          = "qmlink";
    const std::string c_mmLinkTag_          = "mmlink";
    const std::string c_qmInputTag_         = "qminput";
    const std::string c_qmPdbTag_           = "qmpdb";
    const std::string c_qmInputFileNameTag_ = "cp2kinputfile";
    const std::string c_qmBoxTag_ = "qmbox";
    const std::string c_qmTransTag_ = "qmtrans";

    //! Logger instance
    const MDLogger* logger_ = nullptr;

    //! QM index group name, Default whole System
    std::string groupString_ = "System";

    //! QMMM parameters built from mdp input
    QMMMParameters parameters_;

    //! tpr filename as set by grompp -o option, default "topol.tpr" 
    std::string tprFile_ = "topol.tpr";

    //! Vector with atoms point charges
    std::vector<real> atomCharges_;
};

} // namespace gmx

#endif
