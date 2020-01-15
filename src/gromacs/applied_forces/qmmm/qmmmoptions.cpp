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
 * Implements QM/MM options
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "qmmmoptions.h"

#include <fstream>

#include "gromacs/applied_forces/qmmm/qmmm.h"
#include "gromacs/math/vec.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/mtop_lookup.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreetransform.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mdmodulenotification.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"

#include "qmmminputgenerator.h"
#include "qmmmtopologypreprocessor.h"

namespace gmx
{

namespace
{

/*! \brief Helper to declare mdp transform rules.
 *
 * Enforces uniform mdp options that are always prepended with the correct
 * string for the QMMM mdp options.
 *
 * \tparam ToType type to be transformed to
 * \tparam TransformWithFunctionType type of transformation function to be used
 *
 * \param[in] rules KVT transformation rules
 * \param[in] transformationFunction the function to transform the flat kvt tree
 * \param[in] optionTag string tag that describes the mdp option, appended to the
 *                      default string for the QMMM simulation
 */
template<class ToType, class TransformWithFunctionType>
void QMMMMdpTransformFromString(IKeyValueTreeTransformRules* rules,
                                TransformWithFunctionType    transformationFunction,
                                const std::string&           optionTag)
{
    rules->addRule()
            .from<std::string>("/" + QMMMModuleInfo::name_ + "-" + optionTag)
            .to<ToType>("/" + QMMMModuleInfo::name_ + "/" + optionTag)
            .transformWith(transformationFunction);
}

/*! \brief Helper to declare mdp output.
 *
 * Enforces uniform mdp options output strings that are always prepended with the
 * correct string for the QMMM mdp options and are consistent with the
 * options name and transformation type.
 *
 * \tparam OptionType the type of the mdp option
 * \param[in] builder the KVT builder to generate the output
 * \param[in] option the mdp option
 * \param[in] optionTag string tag that describes the mdp option, appended to the
 *                      default string for the QMMM simulation
 */
template<class OptionType>
void addQMMMMdpOutputValue(KeyValueTreeObjectBuilder* builder, const OptionType& option, const std::string& optionTag)
{
    builder->addValue<OptionType>(QMMMModuleInfo::name_ + "-" + optionTag, option);
}

/*! \brief Helper to declare mdp output comments.
 *
 * Enforces uniform mdp options comment output strings that are always prepended
 * with the correct string for the QMMM mdp options and are consistent
 * with the options name and transformation type.
 *
 * \param[in] builder the KVT builder to generate the output
 * \param[in] comment on the mdp option
 * \param[in] optionTag string tag that describes the mdp option
 */
void addQMMMMdpOutputValueComment(KeyValueTreeObjectBuilder* builder,
                                  const std::string&         comment,
                                  const std::string&         optionTag)
{
    builder->addValue<std::string>("comment-" + QMMMModuleInfo::name_ + "-" + optionTag, comment);
}

} // namespace

void QMMMOptions::initMdpTransform(IKeyValueTreeTransformRules* rules)
{
    const auto& stringIdentityTransform = [](std::string s) { return s; };
    QMMMMdpTransformFromString<bool>(rules, &fromStdString<bool>, c_activeTag_);
    QMMMMdpTransformFromString<std::string>(rules, stringIdentityTransform, c_qmGroupTag_);
    QMMMMdpTransformFromString<std::string>(rules, stringIdentityTransform, c_qmMethodTag_);
    QMMMMdpTransformFromString<std::string>(rules, stringIdentityTransform, c_qmUserInputFileNameTag_);
    QMMMMdpTransformFromString<int>(rules, &fromStdString<int>, c_qmChargeTag_);
    QMMMMdpTransformFromString<int>(rules, &fromStdString<int>, c_qmMultTag_);
}

void QMMMOptions::buildMdpOutput(KeyValueTreeObjectBuilder* builder) const
{

    addQMMMMdpOutputValueComment(builder, "", "empty-line");

    // Active flag
    addQMMMMdpOutputValueComment(builder, "; QM/MM with CP2K", "module");
    addQMMMMdpOutputValue(builder, parameters_.active_, c_activeTag_);

    if (parameters_.active_)
    {
        // Index group for QM atoms, default System
        addQMMMMdpOutputValueComment(builder, "; Index group with QM atoms", c_qmGroupTag_);
        addQMMMMdpOutputValue(builder, groupString_, c_qmGroupTag_);

        // QM method (DFT functional), default PBE
        addQMMMMdpOutputValueComment(builder, "; DFT functional for QM calculations", c_qmMethodTag_);
        addQMMMMdpOutputValue<std::string>(builder, c_qmmmQMMethodNames[parameters_.qmMethod_],
                                           c_qmMethodTag_);

        if (parameters_.qmMethod_ == QMMMQMMethod::INPUT)
        {
            // QM input filename, default "cp2k"
            addQMMMMdpOutputValueComment(builder, "; QM input file name (for qmmm-qmmethod = INPUT)",
                                         c_qmUserInputFileNameTag_);
            addQMMMMdpOutputValue(builder, parameters_.qmInputFileName_, c_qmUserInputFileNameTag_);
        }

        // QM charge, default 0
        addQMMMMdpOutputValueComment(builder, "; QM charge", c_qmChargeTag_);
        addQMMMMdpOutputValue(builder, parameters_.qmCharge_, c_qmChargeTag_);

        // QM mutiplicity, default 1
        addQMMMMdpOutputValueComment(builder, "; QM multiplicity", c_qmMultTag_);
        addQMMMMdpOutputValue(builder, parameters_.qmMult_, c_qmMultTag_);
    }
}

void QMMMOptions::initMdpOptions(IOptionsContainerWithSections* options)
{
    auto section = options->addSection(OptionSection(QMMMModuleInfo::name_.c_str()));

    section.addOption(BooleanOption(c_activeTag_.c_str()).store(&parameters_.active_));
    section.addOption(StringOption(c_qmGroupTag_.c_str()).store(&groupString_));
    section.addOption(EnumOption<QMMMQMMethod>(c_qmMethodTag_.c_str())
                              .enumValue(c_qmmmQMMethodNames)
                              .store(&parameters_.qmMethod_));
    section.addOption(StringOption(c_qmUserInputFileNameTag_.c_str()).store(&parameters_.qmInputFileName_));
    section.addOption(IntegerOption(c_qmChargeTag_.c_str()).store(&parameters_.qmCharge_));
    section.addOption(IntegerOption(c_qmMultTag_.c_str()).store(&parameters_.qmMult_));
}

bool QMMMOptions::active() const
{
    return parameters_.active_;
}

const QMMMParameters& QMMMOptions::buildParameters()
{
    return parameters_;
}

void QMMMOptions::setLogger(const MDLogger& logger)
{
    // Exit if QMMM module is not active
    if (!parameters_.active_)
    {
        return;
    }

    logger_ = &logger;
}

void QMMMOptions::appendLog(const std::string msg)
{
    if (logger_)
    {
        GMX_LOG(logger_->info).asParagraph().appendText(msg);
    }
}

void QMMMOptions::setQMMMGroupIndices(const IndexGroupsAndNames& indexGroupsAndNames)
{
    // Exit if QMMM module is not active
    if (!parameters_.active_)
    {
        return;
    }

    // Create QM index
    parameters_.qmIndices_ = indexGroupsAndNames.indices(groupString_);

    // Create temporary index for the whole System
    auto sindices_ = indexGroupsAndNames.indices(std::string("System"));

    // Sort qmindices_ and sindices_
    std::sort(parameters_.qmIndices_.begin(), parameters_.qmIndices_.end());
    std::sort(sindices_.begin(), sindices_.end());

    // Create MM index
    parameters_.mmIndices_.reserve(sindices_.size());

    // Position in qmindicies_
    size_t j = 0;
    // Now loop over sindices_ and write to mmindices_ only the atoms which does not belong to qmindices_
    for (size_t i = 0; i < sindices_.size(); i++)
    {
        if (sindices_[i] != parameters_.qmIndices_[j])
        {
            parameters_.mmIndices_.push_back(sindices_[i]);
        }
        else
        {
            if (j < parameters_.qmIndices_.size() - 1)
            {
                j++;
            }
        }
    }
}

void QMMMOptions::setQMFileName(MdRunInputFilename tprName)
{
    // Exit if QMMM module is not active
    if (!parameters_.active_)
    {
        return;
    }

    // If filename is not provided by user then setup QM input filename same as *.tpr filename
    if (parameters_.qmMethod_ != QMMMQMMethod::INPUT)
    {

        tprFile_                     = tprName.mdRunFilename_;
        std::string baseName         = tprFile_.substr(tprFile_.find_last_of("/\\") + 1);
        parameters_.qmInputFileName_ = baseName.substr(0, baseName.find_last_of(".")) + ".inp";
    }
}

void QMMMOptions::processCoordinates(const CoordinatesAndBoxPreprocessed coord)
{
    // Exit if QMMM module is not active or qmmm-qmmethod = INPUT
    if (!parameters_.active_ || parameters_.qmMethod_ == QMMMQMMethod::INPUT)
    {
        return;
    }

    /* Check if some of the box vectors dimension lower that 1 nm
     *  For SCF stability Box should be big enough.
     */
    matrix box;
    copy_mat(coord.box_, box);
    if (norm(box[0]) < 1.0 || norm(box[1]) < 1.0 || norm(box[2]) < 1.0)
    {
        GMX_THROW(InconsistentInputError(
                "One of the box vectors is shorter than 1 nm.\n"
                "For stable CP2K SCF convergence all simulation box vectors should be "
                ">= 1 nm. Please consider to increase simulation box or provide custom CP2K input "
                "file using qmmm-qmmethod = INPUT"));
    }

    // Generate input and pdb files for CP2K
    QMMMInputGenerator inpGen(parameters_, coord.pbc_, coord.box_, atomCharges_,
                              coord.coordinates_.unpaddedConstArrayRef());
    parameters_.qmPdb_   = inpGen.generateCP2KPdb();
    parameters_.qmInput_ = inpGen.generateCP2KInput();
    copy_mat(inpGen.qmBox(), parameters_.qmBox_);
    parameters_.qmTrans_ = inpGen.qmTrans();
}

void QMMMOptions::modifyQMMMTopology(gmx_mtop_t* mtop)
{
    // Exit if QMMM module is not active
    if (!parameters_.active_)
    {
        return;
    }

    // Process topology
    QMMMTopologyPreprocessor topPrep(mtop, parameters_.qmIndices_);

    // Get atom numbers
    parameters_.atomNumbers_ = topPrep.atomNumbers();

    // Get atom point charges
    atomCharges_ = topPrep.atomCharges();

    // Get Link Frontier
    parameters_.link_ = topPrep.link();

    // Get info about modifications
    QMMMTopologyInfo topInfo = topPrep.topInfo();

    // Print message to the log about performed modifications
    real qmC = static_cast<real>(parameters_.qmCharge_);

    std::string msg = formatString("\nQMMM Note: topologly in %s will be modified.\n", tprFile_.c_str());

    msg += formatString("Number of MM atoms: %d; Number of QM atoms: %d\n", topInfo.mmNum, topInfo.qmNum);

    msg += formatString("Total charge of the classical system (before modifications): %.5f\n",
                        topInfo.mmQTot + topInfo.qmQTot);

    msg += formatString("Classical charge removed from QM atoms: %.5f\n", topInfo.qmQTot);

    if (topInfo.rVSites > 0)
    {
        msg += formatString(
                "Note: There are %d virtual sites found, which are built from QM atoms only.\n"
                "Classical charges on them have been removed as well.\n",
                topInfo.rVSites);
    }

    if (parameters_.qmMethod_ != QMMMQMMethod::INPUT)
    {
        msg += formatString("Total charge of QMMM system: %.5f\n", qmC + topInfo.mmQTot);

        if (fabs(topInfo.qmQTot - qmC) > 1E-5)
        {
            msg += formatString(
                    "Warning: Total charge of your QMMM system differs from classical system!\n"
                    "Consider to manualy spread %.5lf charge over MM atoms nearby to the QM "
                    "region\n",
                    topInfo.qmQTot - qmC);
        }
    }
    else
    {
        if (fabs(topInfo.qmQTot) > 1E-5)
        {
            msg += formatString(
                    "Warning: Removed classical charge from QM atoms is non-zero.\n"
                    "Ideally total charge of QM + MM subsystems should be the same as the "
                    "classical charge of the system before modifications. Check that and consider "
                    "to spread excess charge over MM atoms nearby to the QM region.");
        }
    }

    if (topInfo.rBonds > 0)
    {
        msg += formatString("Bonds removed: %d\n", topInfo.rBonds);
    }

    if (topInfo.rAngles > 0)
    {
        msg += formatString("Angles removed: %d\n", topInfo.rAngles);
    }

    if (topInfo.rDihedrals > 0)
    {
        msg += formatString("Dihedrals removed: %d\n", topInfo.rDihedrals);
    }

    if (topInfo.rSettle > 0)
    {
        msg += formatString("Settles removed: %d\n", topInfo.rSettle);
    }

    if (topInfo.convBonds > 0)
    {
        msg += formatString("F_CONNBONDS (type 5 bonds) added: %d\n", topInfo.convBonds);
    }

    if (topInfo.linkNum > 0)
    {
        msg += formatString("Links generated: %d\n", topInfo.linkNum);
    }

    if (topInfo.numQMConstr > 2)
    {
        msg += "Warning: seems like your QM subsystem has a lot of constrained bonds.\nThey "
               "probably have been generated automatically.\nThat could produce an artifacts in "
               "the simulation.\nConsider constraints = none in the mdp file.\n";
    }

    appendLog(msg);
}

void QMMMOptions::writeInternalParametersToKvt(KeyValueTreeObjectBuilder treeBuilder)
{
    // Write QM atoms index
    auto GroupIndexAdder =
            treeBuilder.addUniformArray<std::int64_t>(QMMMModuleInfo::name_ + "-" + c_qmGroupTag_);
    for (const auto& indexValue : parameters_.qmIndices_)
    {
        GroupIndexAdder.addValue(indexValue);
    }

    // Write MM atoms index
    GroupIndexAdder =
            treeBuilder.addUniformArray<std::int64_t>(QMMMModuleInfo::name_ + "-" + c_mmGroupTag_);
    for (const auto& indexValue : parameters_.mmIndices_)
    {
        GroupIndexAdder.addValue(indexValue);
    }

    // Write atoms numbers
    GroupIndexAdder =
            treeBuilder.addUniformArray<std::int64_t>(QMMMModuleInfo::name_ + "-" + c_atomNumbersTag_);
    for (const auto& indexValue : parameters_.atomNumbers_)
    {
        GroupIndexAdder.addValue(indexValue);
    }

    // Write link
    GroupIndexAdder =
            treeBuilder.addUniformArray<std::int64_t>(QMMMModuleInfo::name_ + "-" + c_qmLinkTag_);
    for (const auto& indexValue : parameters_.link_)
    {
        GroupIndexAdder.addValue(indexValue.qm);
    }
    GroupIndexAdder =
            treeBuilder.addUniformArray<std::int64_t>(QMMMModuleInfo::name_ + "-" + c_mmLinkTag_);
    for (const auto& indexValue : parameters_.link_)
    {
        GroupIndexAdder.addValue(indexValue.mm);
    }

    /* Save CP2K input filename also as internal parameter
     * That is needed in case if input has been generated
     */
    treeBuilder.addValue<std::string>(QMMMModuleInfo::name_ + "-" + c_qmInputFileNameTag_,
                                      parameters_.qmInputFileName_);

    // Write CP2K Input
    treeBuilder.addValue<std::string>(QMMMModuleInfo::name_ + "-" + c_qmInputTag_, parameters_.qmInput_);
    if (parameters_.qmMethod_ != QMMMQMMethod::INPUT)
    {
        std::string   inputFile = tprFile_.substr(0, tprFile_.find_last_of(".")) + ".inp";
        std::ofstream finp(inputFile);
        finp << parameters_.qmInput_;
    }

    // Write CP2K pdb into KVT and to the disk
    treeBuilder.addValue<std::string>(QMMMModuleInfo::name_ + "-" + c_qmPdbTag_, parameters_.qmPdb_);
    if (parameters_.qmMethod_ != QMMMQMMethod::INPUT)
    {
        std::string   pdbFile = tprFile_.substr(0, tprFile_.find_last_of(".")) + ".pdb";
        std::ofstream fpdb(pdbFile);
        fpdb << parameters_.qmPdb_;
    }

    // Write QM box matrix
    auto DoubleArrayAdder = treeBuilder.addUniformArray<double>(QMMMModuleInfo::name_ + "-" + c_qmBoxTag_);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            DoubleArrayAdder.addValue(static_cast<double>(parameters_.qmBox_[i][j]));
        }
    }

    // Write QM Translation vector
    DoubleArrayAdder = treeBuilder.addUniformArray<double>(QMMMModuleInfo::name_ + "-" + c_qmTransTag_);
    for (int i = 0; i < 3; i++)
    {
        DoubleArrayAdder.addValue(static_cast<double>(parameters_.qmTrans_[i]));
    }
}

void QMMMOptions::readInternalParametersFromKvt(const KeyValueTreeObject& tree)
{
    // Check if active
    if (!parameters_.active_)
    {
        return;
    }

    // Try to read QM atoms index
    if (!tree.keyExists(QMMMModuleInfo::name_ + "-" + c_qmGroupTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find QM atoms index vector required for QM/MM simulation.\nThis could be "
                "caused by incompatible or corrupted tpr input file."));
    }
    auto kvtIndexArray = tree[QMMMModuleInfo::name_ + "-" + c_qmGroupTag_].asArray().values();
    parameters_.qmIndices_.resize(kvtIndexArray.size());
    std::transform(std::begin(kvtIndexArray), std::end(kvtIndexArray), std::begin(parameters_.qmIndices_),
                   [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });

    // Try to read MM atoms index
    if (!tree.keyExists(QMMMModuleInfo::name_ + "-" + c_mmGroupTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find MM atoms index vector required for QM/MM simulation.\nThis could be "
                "caused by incompatible or corrupted tpr input file."));
    }
    kvtIndexArray = tree[QMMMModuleInfo::name_ + "-" + c_mmGroupTag_].asArray().values();
    parameters_.mmIndices_.resize(kvtIndexArray.size());
    std::transform(std::begin(kvtIndexArray), std::end(kvtIndexArray), std::begin(parameters_.mmIndices_),
                   [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });

    // Try to read atoms numbers
    if (!tree.keyExists(QMMMModuleInfo::name_ + "-" + c_atomNumbersTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find Atom Numbers vector required for QM/MM simulation.\nThis could be "
                "caused by incompatible or corrupted tpr input file."));
    }
    kvtIndexArray = tree[QMMMModuleInfo::name_ + "-" + c_atomNumbersTag_].asArray().values();
    parameters_.atomNumbers_.resize(kvtIndexArray.size());
    std::transform(std::begin(kvtIndexArray), std::end(kvtIndexArray),
                   std::begin(parameters_.atomNumbers_),
                   [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });

    // Try to read Link Frontier (two separate vectors and then combine)
    std::vector<index> qmLink;
    std::vector<index> mmLink;

    if (!tree.keyExists(QMMMModuleInfo::name_ + "-" + c_qmLinkTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find QM Link Frontier vector required for QM/MM simulation.\nThis could be "
                "caused by incompatible or corrupted tpr input file."));
    }
    kvtIndexArray = tree[QMMMModuleInfo::name_ + "-" + c_qmLinkTag_].asArray().values();
    qmLink.resize(kvtIndexArray.size());
    std::transform(std::begin(kvtIndexArray), std::end(kvtIndexArray), std::begin(qmLink),
                   [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });

    if (!tree.keyExists(QMMMModuleInfo::name_ + "-" + c_mmLinkTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find MM Link Frontier vector required for QM/MM simulation.\nThis could be "
                "caused by incompatible or corrupted tpr input file."));
    }
    kvtIndexArray = tree[QMMMModuleInfo::name_ + "-" + c_mmLinkTag_].asArray().values();
    mmLink.resize(kvtIndexArray.size());
    std::transform(std::begin(kvtIndexArray), std::end(kvtIndexArray), std::begin(mmLink),
                   [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });

    parameters_.link_.resize(qmLink.size());
    for (size_t i = 0; i < qmLink.size(); i++)
    {
        parameters_.link_[i].qm = qmLink[i];
        parameters_.link_[i].mm = mmLink[i];
    }

    /* Read CP2K input filename
     * That is needed in case if input has been generated
     */
    if (!tree.keyExists(QMMMModuleInfo::name_ + "-" + c_qmInputFileNameTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find filename for CP2K input/output files.\nThis could be "
                "caused by incompatible or corrupted tpr input file."));
    }
    parameters_.qmInputFileName_ =
            tree[QMMMModuleInfo::name_ + "-" + c_qmInputFileNameTag_].cast<std::string>();

    // Try to read CP2K input and pdb strings from *.tpr
    if (!tree.keyExists(QMMMModuleInfo::name_ + "-" + c_qmInputTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find CP2K input string required for QM/MM simulation.\nThis could be "
                "caused by incompatible or corrupted tpr input file."));
    }
    parameters_.qmInput_ = tree[QMMMModuleInfo::name_ + "-" + c_qmInputTag_].cast<std::string>();

    if (!tree.keyExists(QMMMModuleInfo::name_ + "-" + c_qmPdbTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find CP2K pdb string required for QM/MM simulation.\nThis could be "
                "caused by incompatible or corrupted tpr input file."));
    }
    parameters_.qmPdb_ = tree[QMMMModuleInfo::name_ + "-" + c_qmPdbTag_].cast<std::string>();

    // Try to read QM box
    if (!tree.keyExists(QMMMModuleInfo::name_ + "-" + c_qmBoxTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find QM box matrix required for QM/MM simulation.\nThis could be "
                "caused by incompatible or corrupted tpr input file."));
    }
    auto kvtDoubleArray = tree[QMMMModuleInfo::name_ + "-" + c_qmBoxTag_].asArray().values();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            parameters_.qmBox_[i][j] = static_cast<real>(kvtDoubleArray[i * 3 + j].cast<double>());
        }
    }

    // Try to read QM translation vector
    if (!tree.keyExists(QMMMModuleInfo::name_ + "-" + c_qmTransTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find QM subsystem centering information for QM/MM simulation.\nThis could "
                "be caused by incompatible or corrupted tpr input file."));
    }
    kvtDoubleArray = tree[QMMMModuleInfo::name_ + "-" + c_qmTransTag_].asArray().values();
    for (int i = 0; i < 3; i++)
    {
        parameters_.qmTrans_[i] = static_cast<real>(kvtDoubleArray[i].cast<double>());
    }
}

} // namespace gmx
