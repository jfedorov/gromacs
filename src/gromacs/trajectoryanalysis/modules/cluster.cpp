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
 * \brief
 * Implements gmx::analysismodules::Cluster.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "gmxpre.h"

#include "cluster.h"

#include <algorithm>
#include <string>

#include "gromacs/coordinateio/coordinatefile.h"
#include "gromacs/coordinateio/requirements.h"
#include "gromacs/fileio/filetypes.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/gmxana/cluster_methods.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"

namespace gmx
{

namespace analysismodules
{

namespace
{

constexpr gmx::EnumerationArray<ClusterMethods, const char*> c_clusterMethodNames = {
    "linkage",
    "jarvis-patrick",
    "monte-carlo",
    "diagonalization",
    "gromos"
};


/*
 * Cluster
 */

class Cluster : public TrajectoryAnalysisModule
{
public:
    Cluster();

    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void optionsFinished(TrajectoryAnalysisSettings* settings) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;

    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    TrajectoryFrameWriterPointer    output_;
    Selection                       sel_;
    std::string                     clusterBaseName_;
    std::string                     rmsdInputMatrixFileName_;
    std::string                     rawRmsdOutputFileName_;
    std::string                     rmsdOutputFileName_;
    std::string                     logFileName_;
    std::string                     rmsdDistributionOutputFileName_;
    std::string                     rmsdEigenVectorFileName_;
    std::string                     mcConversionFileName_;
    std::string                     clusterSizeFileName_;
    std::string                     clusterTransitionsFileName_;
    std::string                     numClusterTransitionsFileName_;
    std::string                     clusterIDFileName_;
    std::string                     clusterMappingFileName_;
    OutputRequirementOptionDirector requirementsBuilder_;
    bool                            useRmsdDistances_                 = false;
    bool                            useLeastSquaresFitting_           = false;
    bool                            setMaximumRmsdLevel_              = false;
    bool                            writeAverageStructure_            = false;
    bool                            writeNumCluster_                  = false;
    bool                            useBinaryValues_                  = false;
    bool                            useNearestNeghbors_               = false;
    bool                            useRandomMCSteps_                 = false;
    bool                            useRmsdInputMatrix_               = false;
    bool                            writeRmsdDistribution_            = false;
    bool                            writeRmsdEigenVectors_            = false;
    bool                            writeMCConversion_                = false;
    bool                            writeClusterSize_                 = false;
    bool                            writeClusterTransitions_          = false;
    bool                            writeNumClusterTransitions_       = false;
    bool                            writeClusterIDs_                  = false;
    bool                            writeClusterMappingFile_          = false;
    int                             rmsdMatrixDiscretizationLevels_   = 40;
    int                             numClusterWrite_                  = 0;
    int                             writeOverThisCount_               = 1;
    int                             minStructuresForColouring_        = 1;
    int                             numJarvisPatrickNearestNeighbors_ = 10;
    int                             numNearestNeighborsForCluster_    = 3;
    int                             randomNumberSeed_                 = -1;
    int                             numRandomMCSteps_                 = 0;
    real                            rmsdCutoff_                       = 0.1;
    real                            maximumRmsdLevel_                 = -1.0;
    real                            minimumRmsDistance_               = 0.0;
    real                            kT_                               = 1e-3;
    ClusterMethods                  method_                           = ClusterMethods::Linkage;
};

Cluster::Cluster() {}


void Cluster::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] can cluster structures using several different methods.",
        "Distances between structures can be determined from a trajectory",
        "or read from an [REF].xpm[ref] matrix file with the [TT]-dm[tt] option.",
        "RMS deviation after fitting or RMS deviation of atom-pair distances",
        "can be used to define the distance between structures.[PAR]",

        "single linkage: add a structure to a cluster when its distance to any",
        "element of the cluster is less than [TT]cutoff[tt].[PAR]",

        "Jarvis Patrick: add a structure to a cluster when this structure",
        "and a structure in the cluster have each other as neighbors and",
        "they have a least [TT]P[tt] neighbors in common. The neighbors",
        "of a structure are the M closest structures or all structures within",
        "[TT]cutoff[tt].[PAR]",

        "Monte Carlo: reorder the RMSD matrix using Monte Carlo such that",
        "the order of the frames is using the smallest possible increments.",
        "With this it is possible to make a smooth animation going from one",
        "structure to another with the largest possible (e.g.) RMSD between",
        "them, however the intermediate steps should be as small as possible.",
        "Applications could be to visualize a potential of mean force",
        "ensemble of simulations or a pulling simulation. Obviously the user",
        "has to prepare the trajectory well (e.g. by not superimposing frames).",
        "The final result can be inspect visually by looking at the matrix",
        "[REF].xpm[ref] file, which should vary smoothly from bottom to top.[PAR]",

        "diagonalization: diagonalize the RMSD matrix.[PAR]",

        "gromos: use algorithm as described in Daura [IT]et al.[it]",
        "([IT]Angew. Chem. Int. Ed.[it] [BB]1999[bb], [IT]38[it], pp 236-240).",
        "Count number of neighbors using cut-off, take structure with",
        "largest number of neighbors with all its neighbors as cluster",
        "and eliminate it from the pool of clusters. Repeat for remaining",
        "structures in pool.[PAR]",

        "When the clustering algorithm assigns each structure to exactly one",
        "cluster (single linkage, Jarvis Patrick and gromos) and a trajectory",
        "file is supplied, the structure with",
        "the smallest average distance to the others or the average structure",
        "or all structures for each cluster will be written to a trajectory",
        "file. When writing all structures, separate numbered files are made",
        "for each cluster.[PAR]",

        "Two output files are always written:",
        "",
        " * [TT]-o[tt] writes the RMSD values in the upper left half of the matrix",
        "   and a graphical depiction of the clusters in the lower right half",
        "   When [TT]-minstruct[tt] = 1 the graphical depiction is black",
        "   when two structures are in the same cluster.",
        "   When [TT]-minstruct[tt] > 1 different colors will be used for each",
        "   cluster.",
        " * [TT]-g[tt] writes information on the options used and a detailed list",
        "   of all clusters and their members.",
        "",

        "Additionally, a number of optional output files can be written:",
        "",
        " * [TT]-dist[tt] writes the RMSD distribution.",
        " * [TT]-ev[tt] writes the eigenvectors of the RMSD matrix",
        "   diagonalization.",
        " * [TT]-sz[tt] writes the cluster sizes.",
        " * [TT]-tr[tt] writes a matrix of the number transitions between",
        "   cluster pairs.",
        " * [TT]-ntr[tt] writes the total number of transitions to or from",
        "   each cluster.",
        " * [TT]-clid[tt] writes the cluster number as a function of time.",
        " * [TT]-clndx[tt] writes the frame numbers corresponding to the clusters to the",
        "   specified index file to be read into trjconv.",
        " * [TT]-cl[tt] writes average (with option [TT]-av[tt]) or central",
        "   structure of each cluster or writes numbered files with cluster members",
        "   for a selected set of clusters (with option [TT]-wcl[tt], depends on",
        "   [TT]-nst[tt] and [TT]-rmsmin[tt]). The center of a cluster is the",
        "   structure with the smallest average RMSD from all other structures",
        "   of the cluster.",
    };

    options->addOption(BooleanOption("dista")
                               .store(&useRmsdDistances_)
                               .description("Use RMSD of distances instead of RMS deviation")
                               .defaultValue(false));
    options->addOption(IntegerOption("nlevels")
                               .store(&rmsdMatrixDiscretizationLevels_)
                               .description("Discretize RMSD matrix in this number of levels")
                               .defaultValue(40));
    options->addOption(RealOption("cutoff")
                               .store(&rmsdCutoff_)
                               .description("RMSD cut-off in nm for two structures to be neighbor")
                               .defaultValue(0.1));
    options->addOption(BooleanOption("fit")
                               .store(&useLeastSquaresFitting_)
                               .description("Use least squares fitting before RMSD calculation")
                               .defaultValue(true));
    options->addOption(RealOption("max")
                               .store(&maximumRmsdLevel_)
                               .description("Maximum level in RMSD matrix")
                               .defaultValueIfSet(1)
                               .storeIsSet(&setMaximumRmsdLevel_));
    options->addOption(
            BooleanOption("av")
                    .store(&writeAverageStructure_)
                    .description("Write average instead of middle structure for each cluster")
                    .defaultValue(false));
    options->addOption(
            IntegerOption("wcl")
                    .store(&numClusterWrite_)
                    .storeIsSet(&writeNumCluster_)
                    .description(
                            "Write the structures for this number of clusters to numbered files")
                    .defaultValueIfSet(1));
    options->addOption(IntegerOption("nst")
                               .store(&writeOverThisCount_)
                               .description("Only write all structures if more than this number of "
                                            "structures per cluster")
                               .defaultValue(1));
    options->addOption(
            RealOption("rmsmin")
                    .store(&minimumRmsDistance_)
                    .defaultValue(0.0)
                    .description(
                            "Minimum rms difference with rest of cluster for writing structures"));
    options->addOption(EnumOption<ClusterMethods>("method")
                               .store(&method_)
                               .enumValue(c_clusterMethodNames)
                               .description("Method for cluster determination"));
    options->addOption(
            IntegerOption("minstruct").store(&minStructuresForColouring_).defaultValue(1).description("Minimum number of structures in cluster for coloring in the [REF].xpm[ref] file"));
    options->addOption(
            BooleanOption("binary")
                    .store(&useBinaryValues_)
                    .defaultValue(false)
                    .description(
                            "Treat the RMSD matrix as consisting of 0 and 1, where the cut-off "
                            "is given by [TT]-cutoff[tt]"));
    options->addOption(
            IntegerOption("M")
                    .store(&numJarvisPatrickNearestNeighbors_)
                    .storeIsSet(&useNearestNeghbors_)
                    .defaultValueIfSet(10)
                    .description(
                            "Number of nearest neighbors considered for Jarvis-Patrick algorithm"));
    options->addOption(
            IntegerOption("P")
                    .store(&numNearestNeighborsForCluster_)
                    .defaultValue(3)
                    .description(
                            "Number of identical nearest neighbors required to form a cluster"));
    options->addOption(
            IntegerOption("seed").store(&randomNumberSeed_).defaultValue(-1).description("Random number seed for Monte Carlo clustering algorithm (-1 means generate)"));
    options->addOption(IntegerOption("nrandom")
                               .store(&numRandomMCSteps_)
                               .storeIsSet(&useRandomMCSteps_)
                               .defaultValueIfSet(1)
                               .description("The first iterations for MC may be done complete "
                                            "random, to shuffle the frames"));
    options->addOption(RealOption("kT").store(&kT_).defaultValue(1e-3).description(
            "Boltzmann weighting factor for Monte Carlo optimization "
            "(zero turns off uphill steps)"));


    options->addOption(SelectionOption("select").store(&sel_).onlyAtoms().description(
            "Selection of particles to write to the file"));

    options->addOption(FileNameOption("cl")
                               .filetype(eftTrajectory)
                               .outputFile()
                               .store(&clusterBaseName_)
                               .defaultBasename("clusters")
                               .description("Clustered structures"));

    options->addOption(FileNameOption("dm")
                               .legacyType(efXPM)
                               .inputFile()
                               .defaultBasename("rmsd")
                               .store(&rmsdInputMatrixFileName_)
                               .storeIsSet(&useRmsdInputMatrix_)
                               .description("RMSD input matrix?"));

    options->addOption(FileNameOption("om")
                               .legacyType(efXPM)
                               .outputFile()
                               .defaultBasename("rmsd-raw")
                               .required()
                               .store(&rawRmsdOutputFileName_)
                               .description("File to write raw RMSD values to?"));
    options->addOption(FileNameOption("o")
                               .legacyType(efXPM)
                               .outputFile()
                               .defaultBasename("rmsd-clust")
                               .required()
                               .store(&rmsdOutputFileName_)
                               .description("File to write cluster RMSD values to"));
    options->addOption(FileNameOption("g")
                               .legacyType(efLOG)
                               .outputFile()
                               .defaultBasename("cluster")
                               .required()
                               .store(&logFileName_)
                               .description("File to write detailed logging information to"));
    options->addOption(FileNameOption("dist")
                               .filetype(eftPlot)
                               .outputFile()
                               .defaultBasename("rmsd-dist")
                               .storeIsSet(&writeRmsdDistribution_)
                               .store(&rmsdDistributionOutputFileName_)
                               .description("File to write optional Cluster RMSD distribution to"));
    options->addOption(FileNameOption("ev")
                               .filetype(eftPlot)
                               .outputFile()
                               .defaultBasename("rmsd-eig")
                               .storeIsSet(&writeRmsdEigenVectors_)
                               .store(&rmsdEigenVectorFileName_)
                               .description("File to write optional RMSD eigenvectors to"));
    options->addOption(FileNameOption("conv")
                               .filetype(eftPlot)
                               .outputFile()
                               .defaultBasename("mc-conv")
                               .storeIsSet(&writeMCConversion_)
                               .store(&mcConversionFileName_)
                               .description("File to write MC conversion info to"));
    options->addOption(FileNameOption("sz")
                               .filetype(eftPlot)
                               .outputFile()
                               .defaultBasename("clust-size")
                               .storeIsSet(&writeClusterSize_)
                               .store(&clusterSizeFileName_)
                               .description("File to write cluster size to"));
    options->addOption(FileNameOption("tr")
                               .legacyType(efXPM)
                               .outputFile()
                               .defaultBasename("clust-trans")
                               .storeIsSet(&writeClusterTransitions_)
                               .store(&clusterTransitionsFileName_)
                               .description("File to write cluster transitions to"));
    options->addOption(FileNameOption("ntr")
                               .filetype(eftPlot)
                               .outputFile()
                               .defaultBasename("cust-trans")
                               .storeIsSet(&writeNumClusterTransitions_)
                               .store(&numClusterTransitionsFileName_)
                               .description("File to write number of transitions to"));
    options->addOption(FileNameOption("clid")
                               .filetype(eftPlot)
                               .outputFile()
                               .defaultBasename("clust-id")
                               .storeIsSet(&writeClusterIDs_)
                               .store(&clusterIDFileName_)
                               .description("File to write cluster ids to"));
    options->addOption(FileNameOption("clndx")
                               .filetype(eftIndex)
                               .outputFile()
                               .defaultBasename("clusters")
                               .storeIsSet(&writeClusterMappingFile_)
                               .store(&clusterMappingFileName_)
                               .description("Index file containing mapping of clusters to frames"));

    requirementsBuilder_.initOptions(options);

    settings->setHelpText(desc);
}

void Cluster::optionsFinished(TrajectoryAnalysisSettings* settings)
{
    int frameFlags = TRX_NEED_X;

    frameFlags |= TRX_READ_V;
    frameFlags |= TRX_READ_F;

    settings->setFrameFlags(frameFlags);
}


void Cluster::initAnalysis(const TrajectoryAnalysisSettings& /*settings*/, const TopologyInformation& top)
{
    output_ = createTrajectoryFrameWriter(top.mtop(),
                                          sel_,
                                          clusterBaseName_,
                                          top.hasTopology() ? top.copyAtoms() : nullptr,
                                          requirementsBuilder_.process());
}

void Cluster::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* /* pbc */, TrajectoryAnalysisModuleData* /*pdata*/)
{
    output_->prepareAndWriteFrame(frnr, fr);
}

void Cluster::finishAnalysis(int /*nframes*/) {}


void Cluster::writeOutput() {}

} // namespace

const char ClusterInfo::name[]             = "cluster-rewrite";
const char ClusterInfo::shortDescription[] = "Clusters structures from trajectory";

TrajectoryAnalysisModulePointer ClusterInfo::create()
{
    return TrajectoryAnalysisModulePointer(new Cluster);
}

} // namespace analysismodules

} // namespace gmx
