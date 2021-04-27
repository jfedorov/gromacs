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
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/histogram.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/analysisdata/paralleloptions.h"
#include "gromacs/coordinateio/coordinatefile.h"
#include "gromacs/coordinateio/requirements.h"
#include "gromacs/fileio/filetypes.h"
#include "gromacs/fileio/gmxfio.h"
#include "gromacs/fileio/matio.h"
#include "gromacs/fileio/rgb.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/gmxana/cmat.h"
#include "gromacs/math/do_fit.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/matrix.h"
#include "gromacs/math/multidimarray.h"
#include "gromacs/mdspan/extensions.h"
#include "gromacs/mdspan/extents.h"
#include "gromacs/mdspan/mdspan.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/trajectory//trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/modules/cluster_diagonalize.h"
#include "gromacs/trajectoryanalysis/modules/cluster_gromos.h"
#include "gromacs/trajectoryanalysis/modules/cluster_jarvis_patrick.h"
#include "gromacs/trajectoryanalysis/modules/cluster_linkage.h"
#include "gromacs/trajectoryanalysis/modules/cluster_monte_carlo.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/loggerbuilder.h"
#include "gromacs/utility/path.h"

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

struct ClusterTrajectoryData
{
    //! Get size of all frames.
    int size() const { return frames.size(); }
    //! All frames in the trajectory for clustering
    std::vector<std::vector<RVec>> frames;
    //! All times for the trajectory frames.
    std::vector<real> times;
    //! Trajectory frame boxes.
    std::vector<Matrix3x3> boxes;
    //! Which actual trajectory frame the cluster frame corresponds to.
    std::vector<int> frameIndex;
};

void calculateDistance(ArrayRef<const RVec>                                        x,
                       basic_mdspan<real, extents<dynamic_extent, dynamic_extent>> distances)
{
    const int numCoordinates = x.size();
    for (int i = 0; i < numCoordinates - 1; ++i)
    {
        RVec xi = x[i];
        for (int j = i + 1; j < numCoordinates; ++j)
        {
            RVec dx;
            rvec_sub(xi, x[j], dx);
            distances(i, j) = norm(dx);
        }
    }
}

real rmsDistance(int                                                               numCoordinates,
                 basic_mdspan<const real, extents<dynamic_extent, dynamic_extent>> distances1,
                 basic_mdspan<const real, extents<dynamic_extent, dynamic_extent>> distances2)
{
    real r2 = 0.0;
    for (int i = 0; i < numCoordinates - 1; ++i)
    {
        for (int j = i + 1; j < numCoordinates; ++j)
        {
            const real r = distances1(i, j) - distances2(i, j);
            r2 += r * r;
        }
    }
    r2 /= gmx::exactDiv(numCoordinates * (numCoordinates - 1), 2);
    return std::sqrt(r2);
}

void addClusterFrame(ClusterTrajectoryData* data,
                     int                    currentFrame,
                     const t_trxframe&      fr,
                     ArrayRef<const int>    index,
                     ArrayRef<const real>   masses,
                     bool                   useLeastSquaresFitting)
{
    auto& currentStructure = data->frames.emplace_back();
    currentStructure.resize(index.size());
    for (int i = 0; i < gmx::ssize(index); ++i)
    {
        copy_rvec(fr.x[index[i]], currentStructure[i]);
    }
    data->times.emplace_back(fr.time);
    data->boxes.emplace_back(createMatrix3x3FromLegacyMatrix(fr.box));
    data->frames.emplace_back(currentFrame);
    if (useLeastSquaresFitting)
    {
        reset_x(index.size(),
                index.data(),
                index.size(),
                nullptr,
                as_rvec_array(currentStructure.data()),
                masses.data());
    }
}

void addCoordsToFile(int                    frame,
                     TrajectoryFrameWriter* writer,
                     ArrayRef<const RVec>   coords,
                     ArrayRef<const int>    index,
                     real                   time,
                     Matrix3x3ConstSpan     box)
{
    t_trxframe writeFrame;
    clear_trxframe(&writeFrame, true);
    fillLegacyMatrix(box, writeFrame.box);
    writeFrame.bBox   = true;
    writeFrame.natoms = coords.size();
    writeFrame.index  = const_cast<int*>(index.data());
    writeFrame.bIndex = true;
    writeFrame.x      = const_cast<rvec*>(as_rvec_array(coords.data()));
    writeFrame.time   = time;
    writer->prepareAndWriteFrame(frame, writeFrame);
}

t_mat* generateClusterMatrix(int                                   numFrames,
                             int                                   numCoordinates,
                             real                                  rmsdCutoff,
                             real                                  minimumRmsDistance,
                             bool                                  runFullAnalysis,
                             bool                                  oneDimensional,
                             bool                                  useRmsdCutoff,
                             bool                                  useRmsdDistances,
                             bool                                  useLeastSquaresFitting,
                             const std::vector<std::vector<RVec>>& coordinates,
                             ArrayRef<const real>                  masses,
                             const MDLogger&                       logger)
{
    t_mat*  matrix     = init_mat(numFrames, oneDimensional);
    int64_t numEntries = (static_cast<int64_t>(numFrames) * static_cast<int64_t>(numFrames - 1)) / 2;
    if (!useRmsdDistances)
    {
        GMX_LOG(logger.info)
                .asParagraph()
                .appendTextFormatted("Computing %dx%d RMS deviation matrix\n", numFrames, numFrames);
        std::vector<RVec> currentCoordinates(numCoordinates);
        for (int i1 = 0; i1 < numFrames; ++i1)
        {
            for (int i2 = i1 + 1; i2 < numFrames; ++i2)
            {
                for (int index = 0; index < numCoordinates; ++index)
                {
                    copy_rvec(coordinates[i1][index], currentCoordinates[index]);
                }
                if (useLeastSquaresFitting)
                {
                    do_fit(numCoordinates,
                           masses.data(),
                           as_rvec_array(coordinates[i2].data()),
                           as_rvec_array(currentCoordinates.data()));
                }
                const real rmsd = rmsdev(numCoordinates,
                                         masses.data(),
                                         as_rvec_array(coordinates[i2].data()),
                                         as_rvec_array(currentCoordinates.data()));
                set_mat_entry(matrix, i1, i2, rmsd);
            }
            numEntries -= numFrames - i1 - 1;
            GMX_LOG(logger.info)
                    .appendTextFormatted(
                            "\r# RMSD calculations left: "
                            "%" PRId64 "   ",
                            numEntries);
        }
    }
    else
    {
        GMX_LOG(logger.info)
                .asParagraph()
                .appendTextFormatted("Computing %dx%d RMS distance deviation matrix\n", numFrames, numFrames);
        MultiDimArray<std::vector<real>, extents<dynamic_extent, dynamic_extent>> dist1;
        MultiDimArray<std::vector<real>, extents<dynamic_extent, dynamic_extent>> dist2;
        dist1.resize(numCoordinates, numCoordinates);
        dist2.resize(numCoordinates, numCoordinates);
        for (int i1 = 0; i1 < numFrames; ++i1)
        {
            calculateDistance(coordinates[i1], dist1);
            for (int i2 = i1 + 1; i2 < numFrames; ++i2)
            {
                calculateDistance(coordinates[i2], dist2);
                set_mat_entry(matrix, i1, i2, rmsDistance(numCoordinates, dist1, dist2));
            }
            numEntries -= numFrames - i1 - 1;
            GMX_LOG(logger.info)
                    .appendTextFormatted(
                            "\r# RMSD calculations left: "
                            "%" PRId64 "   ",
                            numEntries);
        }
    }
    GMX_LOG(logger.info).appendTextFormatted("The RMSD ranges from %g to %g nm", matrix->minrms, matrix->maxrms);
    GMX_LOG(logger.info).appendTextFormatted("Average RMSD is %g", 2 * matrix->sumrms / (numFrames * (numFrames - 1)));
    GMX_LOG(logger.info).appendTextFormatted("Number of structures for matrix %d", numFrames);
    GMX_LOG(logger.info).appendTextFormatted("Energy of the matrix is %g.", mat_energy(matrix));
    if (useRmsdCutoff && (rmsdCutoff < matrix->minrms || rmsdCutoff > matrix->maxrms))
    {
        GMX_LOG(logger.warning)
                .asParagraph()
                .appendTextFormatted(
                        "WARNING: rmsd cutoff %g is outside range of rmsd values "
                        "%g to %g",
                        rmsdCutoff,
                        matrix->minrms,
                        matrix->maxrms);
    }
    if (runFullAnalysis)
    {
        if (minimumRmsDistance < matrix->minrms)
        {
            GMX_LOG(logger.warning)
                    .asParagraph()
                    .appendTextFormatted("WARNING: rmsd minimum %g is below lowest rmsd value %g",
                                         minimumRmsDistance,
                                         matrix->minrms);
        }
    }

    return matrix;
}

void convertToBinary(t_mat* matrix, int numFrames, real rmsdCutoff)
{
    for (int i1 = 0; i1 < numFrames; ++i1)
    {
        for (int i2 = 0; i2 < numFrames; ++i2)
        {
            if (matrix->mat[i1][i2] < rmsdCutoff)
            {
                matrix->mat[i1][i2] = 0;
            }
            else
            {
                matrix->mat[i1][i2] = 1;
            }
        }
    }
}

void convertBinaryToMaxRms(t_mat* matrix, int numFrames)
{
    for (int i2 = 0; i2 < numFrames; ++i2)
    {
        for (int i1 = i2 + 1; i1 < numFrames; ++i1)
        {
            if (matrix->mat[i1][i2] != 0.0F)
            {
                matrix->mat[i1][i2] = matrix->maxrms;
            }
        }
    }
}

void calculateClusterTransitions(ArrayRef<const int> clusterIndex,
                                 bool                writeClusterTransitions,
                                 bool                writeNumClusterTransitions,
                                 const std::string&  clusterTransitionsFileName,
                                 AnalysisDataHandle* transitionDataHandle,
                                 const MDLogger&     logger)
{
    MultiDimArray<std::vector<real>, extents<dynamic_extent, dynamic_extent>> transitions;
    std::vector<int>  numTransitions(clusterIndex.size(), 0);
    std::vector<real> transitionAxis(clusterIndex.size());
    transitions.resize(clusterIndex.ssize(), clusterIndex.ssize());
    std::iota(transitionAxis.begin(), transitionAxis.end(), 1);
    std::fill(transitions.toArrayRef().begin(), transitions.toArrayRef().end(), 0);
    int maximumNumTransitions = 0;
    int totalTransitions      = 0;
    for (auto it = clusterIndex.begin() + 1; it != clusterIndex.end(); ++it)
    {
        const auto prevIt = it - 1;
        if (*it != *prevIt)
        {
            totalTransitions++;
            const int currPos = (*it) - 1;
            const int prevPos = (*prevIt) - 1;
            numTransitions[prevPos]++;
            numTransitions[currPos]++;
            transitions(prevPos, currPos)++;
            maximumNumTransitions =
                    std::max(static_cast<real>(maximumNumTransitions), transitions(prevPos, currPos));
        }
    }
    GMX_LOG(logger.info)
            .asParagraph()
            .appendTextFormatted(
                    "Counted %d transitions in total, "
                    "max %d between two specific clusters\n",
                    totalTransitions,
                    maximumNumTransitions);
    if (writeClusterTransitions)
    {
        constexpr t_rgb rlo   = { 1.0, 1.0, 1.0 };
        constexpr t_rgb rhi   = { 0.0, 0.0, 1.0 };
        FILE*           out   = gmx_ffopen(clusterTransitionsFileName, "w");
        int             scale = std::min(maximumNumTransitions + 1, 80);
        write_xpm(out,
                  0,
                  "Cluster Transitions",
                  "# transitions",
                  "from cluster",
                  "to cluster",
                  clusterIndex.size(),
                  clusterIndex.size(),
                  transitionAxis.data(),
                  transitionAxis.data(),
                  transitions,
                  0,
                  maximumNumTransitions,
                  rlo,
                  rhi,
                  &scale);
    }
    if (writeNumClusterTransitions)
    {
        for (int i = 0; i < clusterIndex.ssize(); ++i)
        {
            transitionDataHandle->startFrame(i, i + 1);
            transitionDataHandle->setPoint(0, numTransitions[i]);
            transitionDataHandle->finishFrame();
        }
        transitionDataHandle->finishData();
    }
}

int colorClustersWithMinimumPopulation(t_mat*              matrix,
                                       ArrayRef<const int> clusters,
                                       int                 minNumForColoring,
                                       const MDLogger&     logger)
{
    const int        numFrames = clusters.size();
    std::vector<int> clusterNumbers(numFrames, 0);
    std::vector<int> clusterColorIndex(numFrames, 0);
    for (int i = 0; i < numFrames; ++i)
    {
        clusterNumbers[clusters[i]]++;
    }
    int totalNumClusters = 0;
    for (int i = 0; i < numFrames; ++i)
    {
        if (clusterNumbers[i] > minNumForColoring)
        {
            totalNumClusters++;
            for (int j = 0; j < numFrames; ++j)
            {
                if (clusters[j] == i)
                {
                    clusterColorIndex[j] = totalNumClusters;
                }
            }
        }
    }
    totalNumClusters++;
    GMX_LOG(logger.info)
            .asParagraph()
            .appendTextFormatted("There are %d clusters with at least %d conformations\n",
                                 totalNumClusters,
                                 minNumForColoring);
    for (int i = 0; i < numFrames; ++i)
    {
        const int clusterID = clusters[i];
        for (int j = 0; j < i; ++j)
        {
            if ((clusterID == clusters[j]) && (clusterNumbers[clusterID] >= minNumForColoring))
            {
                matrix->mat[i][j] = clusterColorIndex[i];
            }
            else
            {
                matrix->mat[i][j] = 0;
            }
        }
    }
    return totalNumClusters;
}

void markClusterWithValue(t_mat* matrix, ArrayRef<const int> clusters)
{
    const int numFrames = clusters.size();
    for (int i = 0; i < numFrames; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if (clusters[i] == clusters[j])
            {
                matrix->mat[i][j] = matrix->maxrms;
            }
            else
            {
                matrix->mat[i][j] = 0;
            }
        }
    }
}

TrajectoryFrameWriterPointer openClusterOutputFile(const std::string&                     fileName,
                                                   const OutputRequirementOptionDirector& requirementsBuilder,
                                                   const TopologyInformation&             top,
                                                   const Selection&                       sel)
{
    return createTrajectoryFrameWriter(
            top.mtop(), sel, fileName, top.copyAtoms(), requirementsBuilder.process());
}

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
    bool                            useRmsdCutoff_                    = false;
    bool                            useRmsdDistances_                 = false;
    bool                            useLeastSquaresFitting_           = false;
    bool                            setMaximumRmsdLevel_              = false;
    bool                            writeAverageStructure_            = false;
    bool                            useWriteNumCluster_               = false;
    bool                            useBinaryValues_                  = false;
    bool                            useNearestNeighbors_              = false;
    bool                            useRandomMCSteps_                 = false;
    bool                            useWriteOverThisCount_            = false;
    bool                            useWriteOutputFile_               = false;
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
    int                             maxMCIterations_                  = 10000;
    int                             numRandomMCSteps_                 = 0;
    real                            rmsdCutoff_                       = 0.1;
    real                            maximumRmsdLevel_                 = -1.0;
    real                            minimumRmsDistance_               = 0.0;
    real                            kT_                               = 1e-3;
    ClusterMethods                  method_                           = ClusterMethods::Linkage;
    ClusterTrajectoryData           frameData_;
    std::vector<int>                localIndices_;
    std::vector<real>               masses_;
    std::vector<RVec>               referenceCoordinates_;
    std::unique_ptr<LoggerOwner>    loggerOwner_;
    std::unique_ptr<ICluster>       clusterMethod_;

    t_mat* clusterMatrixHandle_ = nullptr;

    const TopologyInformation* top_ = nullptr;

    AnalysisDataPlotSettings  plotSettings_;
    AnalysisData              clusterTransitionData_;
    AnalysisData              clusterRmsd_;
    AnalysisData              clusterSizeData_;
    AnalysisData              clusterIdData_;
    AnalysisHistogramSettings histogramSettings_;
};

Cluster::Cluster()
{
    LoggerBuilder builder;
    builder.addTargetStream(gmx::MDLogger::LogLevel::Info, &gmx::TextOutputFile::standardOutput());
    builder.addTargetStream(gmx::MDLogger::LogLevel::Warning, &gmx::TextOutputFile::standardError());
    loggerOwner_ = std::make_unique<LoggerOwner>(builder.build());
}


void Cluster::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] can cluster structures using several different methods.",
        "Distances between structures can be determined from a trajectory.",
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
                               .storeIsSet(&useRmsdCutoff_)
                               .description("RMSD cut-off in nm for two structures to be neighbor")
                               .defaultValueIfSet(0.1));
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
                    .storeIsSet(&useWriteNumCluster_)
                    .description(
                            "Write the structures for this number of clusters to numbered files")
                    .defaultValueIfSet(1));
    options->addOption(IntegerOption("nst")
                               .store(&writeOverThisCount_)
                               .storeIsSet(&useWriteOverThisCount_)
                               .description("Only write all structures if more than this number of "
                                            "structures per cluster")
                               .defaultValueIfSet(1));
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
                    .storeIsSet(&useNearestNeighbors_)
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

    options->addOption(
            IntegerOption("niter").store(&maxMCIterations_).defaultValue(10000).description("Number of iterations for MC"));

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
                               .filetype(OptionFileType::Trajectory)
                               .outputFile()
                               .store(&clusterBaseName_)
                               .storeIsSet(&useWriteOutputFile_)
                               .defaultBasename("clusters")
                               .description("Clustered structures"));

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
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .defaultBasename("rmsd-dist")
                               .storeIsSet(&writeRmsdDistribution_)
                               .store(&rmsdDistributionOutputFileName_)
                               .description("File to write optional Cluster RMSD distribution to"));
    options->addOption(FileNameOption("ev")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .defaultBasename("rmsd-eig")
                               .storeIsSet(&writeRmsdEigenVectors_)
                               .store(&rmsdEigenVectorFileName_)
                               .description("File to write optional RMSD eigenvectors to"));
    options->addOption(FileNameOption("conv")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .defaultBasename("mc-conv")
                               .storeIsSet(&writeMCConversion_)
                               .store(&mcConversionFileName_)
                               .description("File to write MC conversion info to"));
    options->addOption(FileNameOption("sz")
                               .filetype(OptionFileType::Plot)
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
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .defaultBasename("cust-trans")
                               .storeIsSet(&writeNumClusterTransitions_)
                               .store(&numClusterTransitionsFileName_)
                               .description("File to write number of transitions to"));
    options->addOption(FileNameOption("clid")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .defaultBasename("clust-id")
                               .storeIsSet(&writeClusterIDs_)
                               .store(&clusterIDFileName_)
                               .description("File to write cluster ids to"));
    options->addOption(FileNameOption("clndx")
                               .filetype(OptionFileType::Index)
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
    settings->setFlag(TrajectoryAnalysisSettings::efRequireTop);
    settings->setFlag(TrajectoryAnalysisSettings::efUseTopX);

    const MDLogger& logger = loggerOwner_->logger();
    if (method_ == ClusterMethods::JarvisPatrick)
    {
        std::string message;
        const bool  useRmsdForJP = (!useNearestNeighbors_ || useRmsdCutoff_);
        if (useNearestNeighbors_
            && (numJarvisPatrickNearestNeighbors_ < 0 || numJarvisPatrickNearestNeighbors_ == 1))
        {
            auto errorMessage = formatString("M (%d) must be 0 or larger than 1",
                                             numJarvisPatrickNearestNeighbors_);
            GMX_THROW(InvalidInputError(errorMessage));
        }
        if (numJarvisPatrickNearestNeighbors_ < 2)
        {
            message = formatString(
                    "Will use P=%d and RMSD cutoff (%g)", numNearestNeighborsForCluster_, rmsdCutoff_);
            useRmsdCutoff_ = true;
        }
        else
        {
            if (numNearestNeighborsForCluster_ >= numJarvisPatrickNearestNeighbors_)
            {
                auto errorMessage = formatString(
                        "Number of neighbors required (P = %d) must be less than M = %d",
                        numNearestNeighborsForCluster_,
                        numJarvisPatrickNearestNeighbors_);
                GMX_THROW(InvalidInputError(errorMessage));
            }
            if (useRmsdForJP)
            {
                message        = formatString("Will use P=%d, M=%d and RMSD cutoff (%g)",
                                       numNearestNeighborsForCluster_,
                                       numJarvisPatrickNearestNeighbors_,
                                       rmsdCutoff_);
                useRmsdCutoff_ = true;
            }
            else
            {
                message = formatString("Will use P=%d, M=%d",
                                       numNearestNeighborsForCluster_,
                                       numJarvisPatrickNearestNeighbors_);
            }
        }
        GMX_LOG(logger.info).asParagraph().appendTextFormatted("%s for determining the neighbors", message.c_str());
    }
    else
    {
        useRmsdCutoff_ = (useBinaryValues_ || method_ == ClusterMethods::Linkage
                          || method_ == ClusterMethods::Gromos);
    }
    if (useRmsdCutoff_ && method_ != ClusterMethods::JarvisPatrick)
    {
        GMX_LOG(logger.info).asParagraph().appendTextFormatted("Using RMSD cutoff %g nm\n", rmsdCutoff_);
    }
    if (useRmsdCutoff_ && (minimumRmsDistance_ > rmsdCutoff_))
    {
        GMX_LOG(logger.warning)
                .asParagraph()
                .appendTextFormatted("WARNING: rmsd minimum %g is above rmsd cutoff %g",
                                     minimumRmsDistance_,
                                     rmsdCutoff_);
    }
    if (method_ == ClusterMethods::MonteCarlo)
    {
        GMX_LOG(logger.info).asParagraph().appendTextFormatted("Using %d iterations\n", maxMCIterations_);
    }
}


void Cluster::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
    plotSettings_ = settings.plotSettings();
    if (!sel_.isValid())
    {
        localIndices_.resize(top.mtop()->natoms);
        std::iota(localIndices_.begin(), localIndices_.end(), 0);
    }
    ArrayRef<const int> atomIndexRef = sel_.isValid() ? sel_.atomIndices() : localIndices_;
    const int           natoms       = sel_.isValid() ? sel_.atomCount() : localIndices_.size();
    if (!top.hasFullTopology())
    {
        GMX_THROW(InvalidInputError("Need full topology with masses for clustering"));
    }
    masses_.resize(natoms);
    referenceCoordinates_.resize(natoms);
    const auto* localAtoms = top.atoms();
    for (int i = 0; i < gmx::ssize(atomIndexRef); ++i)
    {
        masses_[i] = localAtoms->atom[atomIndexRef[i]].m;
        copy_rvec(top.x()[atomIndexRef[i]], referenceCoordinates_[i]);
    }
    top_ = &top;
}

void Cluster::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* /* pbc */, TrajectoryAnalysisModuleData* /*pdata*/)
{
    ArrayRef<const int> atomIndexRef = sel_.isValid() ? sel_.atomIndices() : localIndices_;
    addClusterFrame(&frameData_, frnr, fr, atomIndexRef, masses_, useLeastSquaresFitting_);
}

void Cluster::finishAnalysis(int nframes)
{
    if (nframes < 2)
    {
        GMX_THROW(InvalidInputError("Need at least two frames to start clustering"));
    }

    const MDLogger& logger = loggerOwner_->logger();
    const int       natoms = sel_.isValid() ? sel_.atomCount() : localIndices_.size();
    GMX_LOG(logger.info)
            .asParagraph()
            .appendTextFormatted("Allocated %zu bytes for frames\n", (nframes * natoms * sizeof(RVec)));
    GMX_LOG(logger.info).asParagraph().appendTextFormatted("Read %d frames from trajectory\n", nframes);
    const bool runAnalysis = (method_ == ClusterMethods::Linkage || method_ == ClusterMethods::JarvisPatrick
                              || method_ == ClusterMethods::Gromos);

    clusterMatrixHandle_ = generateClusterMatrix(nframes,
                                                 natoms,
                                                 rmsdCutoff_,
                                                 minimumRmsDistance_,
                                                 runAnalysis,
                                                 method_ == ClusterMethods::Diagonalization,
                                                 useRmsdCutoff_,
                                                 useRmsdDistances_,
                                                 useLeastSquaresFitting_,
                                                 frameData_.frames,
                                                 masses_,
                                                 logger);

    if (writeRmsdDistribution_)
    {
        const real scalingFactor = 100 / clusterMatrixHandle_->maxrms;
        histogramSettings_ = AnalysisHistogramSettings(gmx::histogramFromBins(0, 101, 1 / scalingFactor));
        AnalysisDataSimpleHistogramModulePointer rmsdHistogramModule(
                new AnalysisDataSimpleHistogramModule(histogramSettings_));

        clusterRmsd_.addModule(rmsdHistogramModule);
        registerBasicDataset(&rmsdHistogramModule->averager(), "histogram");

        AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule(plotSettings_));
        plotm->setFileName(rmsdDistributionOutputFileName_);
        plotm->setTitle("RMS Distribution Histogram");
        plotm->setXLabel("RMS (nm)");
        plotm->setYLabel("counts");
        rmsdHistogramModule->averager().addModule(plotm);
        clusterRmsd_.setDataSetCount(1);
        clusterRmsd_.setColumnCount(0, 1);

        AnalysisDataHandle dh  = clusterRmsd_.startData({});
        int                pos = 0;
        for (int i = 0; i < nframes; ++i)
        {
            for (int j = i + 1; j < nframes; ++j)
            {
                int xPos = gmx::roundToInt(scalingFactor * clusterMatrixHandle_->mat[i][j]);
                dh.startFrame(pos, xPos);
                dh.setPoint(0, clusterMatrixHandle_->mat[i][j]);
                dh.finishFrame();
                pos++;
            }
        }
        dh.finishData();
        rmsdHistogramModule->averager().normalizeProbability();
        rmsdHistogramModule->averager().done();
    }

    if (useBinaryValues_)
    {
        convertToBinary(clusterMatrixHandle_, nframes, rmsdCutoff_);
    }
    switch (method_)
    {
        case ClusterMethods::Linkage:
            clusterMethod_ = std::make_unique<ClusterLinkage>(
                    clusterMatrixHandle_, rmsdCutoff_, nframes, logger);
            break;
        case ClusterMethods::Diagonalization:
            clusterMethod_ = std::make_unique<ClusterDiagonalize>(
                    clusterMatrixHandle_, rmsdCutoff_, nframes, logger);
            break;
        case ClusterMethods::MonteCarlo:
            clusterMethod_ = std::make_unique<ClusterMonteCarlo>(clusterMatrixHandle_,
                                                                 rmsdCutoff_,
                                                                 kT_,
                                                                 nframes,
                                                                 randomNumberSeed_,
                                                                 maxMCIterations_,
                                                                 numRandomMCSteps_,
                                                                 logger,
                                                                 frameData_.times);
            break;
        case ClusterMethods::JarvisPatrick:
            clusterMethod_ = std::make_unique<ClusterJarvisPatrick>(clusterMatrixHandle_,
                                                                    rmsdCutoff_,
                                                                    numJarvisPatrickNearestNeighbors_,
                                                                    numNearestNeighborsForCluster_,
                                                                    logger);
            break;
        case ClusterMethods::Gromos:
            clusterMethod_ =
                    std::make_unique<ClusterGromos>(clusterMatrixHandle_, rmsdCutoff_, nframes, logger);
            break;
        default: GMX_THROW(InvalidInputError("No such clustering method"));
    }
}


void Cluster::writeOutput()
{
    const MDLogger&     logger       = loggerOwner_->logger();
    ArrayRef<const int> atomIndexRef = sel_.isValid() ? sel_.atomIndices() : localIndices_;
    const int           natoms       = atomIndexRef.size();
    const int           numFrames    = frameData_.size();
    ArrayRef<const int> clusters;
    if (method_ != ClusterMethods::Diagonalization)
    {
        clusters = clusterMethod_->clusterList();
    }
    const bool runAnalysis = (method_ == ClusterMethods::Linkage || method_ == ClusterMethods::JarvisPatrick
                              || method_ == ClusterMethods::Gromos);

    if (runAnalysis)
    {
        std::vector<RVec> averageCoordinates(natoms);

        if (minStructuresForColouring_ > 1)
        {
            colorClustersWithMinimumPopulation(
                    clusterMatrixHandle_, clusters, minStructuresForColouring_, logger);
        }
        else
        {
            markClusterWithValue(clusterMatrixHandle_, clusters);
        }
        const bool writeOutputStructures = (writeAverageStructure_ || useWriteNumCluster_ || useWriteOverThisCount_
                                            || useRmsdCutoff_ || useWriteOutputFile_);
        std::string outputBaseName;
        if (writeOutputStructures)
        {
            outputBaseName = clusterBaseName_;
            if (useWriteNumCluster_)
            {
                numClusterWrite_    = std::max(numClusterWrite_, static_cast<int>(clusters.size()));
                const int numDigits = static_cast<int>(
                        (std::log(static_cast<real>(numClusterWrite_)) / std::log(10.0)) + 1);
                outputBaseName = Path::concatenateBeforeExtension(
                        outputBaseName, formatString("%%0%dd", numDigits));
            }
            GMX_LOG(logger.info)
                    .asParagraph()
                    .appendTextFormatted("Writing %s structure for each cluster to %s",
                                         writeAverageStructure_ ? "average" : "middle",
                                         outputBaseName.c_str());
            if (useWriteNumCluster_)
            {
                GMX_LOG(logger.info)
                        .asParagraph()
                        .appendTextFormatted(
                                "Writing %s for %sclusters%s to %s",
                                useRmsdCutoff_
                                        ? formatString("structures with rmsd > %g", rmsdCutoff_).c_str()
                                        : "all structures",
                                numClusterWrite_ == clusters.ssize()
                                        ? "all "
                                        : formatString("the first %d ", numClusterWrite_).c_str(),
                                useWriteOverThisCount_
                                        ? formatString(" with more than %d structures", writeOverThisCount_)
                                                  .c_str()
                                        : "",
                                outputBaseName.c_str());
            }
            if (useLeastSquaresFitting_)
            {
                reset_x(atomIndexRef.size(),
                        atomIndexRef.data(),
                        atomIndexRef.size(),
                        nullptr,
                        as_rvec_array(referenceCoordinates_.data()),
                        masses_.data());
            }
        }
        if (writeClusterTransitions_ || writeNumClusterTransitions_)
        {
            registerAnalysisDataset(&clusterTransitionData_, "Cluster transitions");
            if (writeNumClusterTransitions_)
            {
                AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule(plotSettings_));
                plotm->setFileName(numClusterTransitionsFileName_);
                plotm->setTitle("Cluster transitions");
                plotm->setXLabel("Cluster #");
                plotm->setYLabel("# transitions");
                clusterTransitionData_.addModule(plotm);
                clusterTransitionData_.setDataSetCount(1);
                clusterTransitionData_.setColumnCount(0, 1);
            }

            AnalysisDataHandle dh = clusterTransitionData_.startData({});
            calculateClusterTransitions(clusters,
                                        writeClusterTransitions_,
                                        writeNumClusterTransitions_,
                                        clusterTransitionsFileName_,
                                        &dh,
                                        logger);
        }
        if (writeClusterIDs_)
        {
            registerAnalysisDataset(&clusterIdData_, "Cluster IDs");
            AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule(plotSettings_));
            plotm->setFileName(clusterIDFileName_);
            plotm->setTitle("Clusters");
            plotm->setXAxisIsTime();
            plotm->setYLabel("Cluster #");
            clusterIdData_.addModule(plotm);
            clusterIdData_.setDataSetCount(1);
            clusterIdData_.setColumnCount(0, 1);
            AnalysisDataHandle dh = clusterIdData_.startData({});
            for (int i = 0; i < clusters.ssize(); ++i)
            {
                dh.startFrame(i, frameData_.times[i]);
                dh.setPoint(0, clusters[i]);
                dh.finishFrame();
            }
            dh.finishData();
        }
        AnalysisDataHandle clusterSizeDataHandle;
        if (writeClusterSize_)
        {
            registerAnalysisDataset(&clusterSizeData_, "Cluster Size");
            AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule(plotSettings_));
            plotm->setFileName(clusterSizeFileName_);
            plotm->setTitle("Cluster sizes");
            plotm->setXLabel("Cluster #");
            plotm->setYLabel("# Structures");
            clusterSizeData_.addModule(plotm);
            clusterSizeData_.setDataSetCount(1);
            clusterSizeData_.setColumnCount(0, 1);
            clusterSizeDataHandle = clusterSizeData_.startData({});
        }
        FILE* clusterMappingFile = nullptr;
        if (writeClusterMappingFile_)
        {
            clusterMappingFile = gmx_ffopen(clusterMappingFileName_, "w");
        }
        std::vector<int> structure(clusters.size());
        for (int i = 1; i <= clusters.ssize(); ++i)
        {
            if (writeOutputStructures)
            {
                for (auto x : averageCoordinates)
                {
                    clear_rvec(x);
                }
                int nstr = 0;
                for (int i1 = 0; i1 < clusters.ssize(); ++i1)
                {
                    if (clusters[i1] == i)
                    {
                        structure[nstr] = i1;
                        nstr++;
                        ArrayRef<const RVec> referenceCoords;
                        if (writeOutputStructures && (writeAverageStructure_ || useWriteNumCluster_))
                        {
                            if (useLeastSquaresFitting_)
                            {
                                reset_x(atomIndexRef.size(),
                                        atomIndexRef.data(),
                                        natoms,
                                        nullptr,
                                        as_rvec_array(frameData_.frames[i1].data()),
                                        masses_.data());
                            }
                            if (nstr == 1)
                            {
                                referenceCoords = frameData_.frames[i1];
                            }
                            else
                            {
                                do_fit(natoms,
                                       masses_.data(),
                                       as_rvec_array(referenceCoords.data()),
                                       as_rvec_array(frameData_.frames[i1].data()));
                            }
                            if (writeAverageStructure_)
                            {
                                for (int coord = 0; coord < natoms; ++coord)
                                {
                                    rvec_inc(averageCoordinates[coord], frameData_.frames[i1][coord]);
                                }
                            }
                        }
                    }
                }
                if (writeClusterSize_)
                {
                    clusterSizeDataHandle.startFrame(i, i);
                    clusterSizeDataHandle.setPoint(0, nstr);
                    clusterSizeDataHandle.finishFrame();
                }
                if (writeClusterMappingFile_)
                {
                    fprintf(clusterMappingFile, "[Cluster_%04d]\n", i);
                }
                real   clusterRmsd      = 0;
                int    midStructure     = 0;
                real   midStructureRmsd = 10000;
                t_mat* matrix           = clusterMatrixHandle_;
                for (int i1 = 0; i1 < nstr; ++i1)
                {
                    real r = 0;
                    if (nstr > 1)
                    {
                        for (int str = 0; str < nstr; ++str)
                        {
                            if (str < nstr)
                            {
                                r += matrix->mat[structure[str]][structure[i1]];
                            }
                            else
                            {
                                r += matrix->mat[structure[i1]][structure[str]];
                            }
                        }
                        r /= (nstr - 1);
                    }
                    if (r < midStructureRmsd)
                    {
                        midStructure     = structure[i1];
                        midStructureRmsd = r;
                    }
                    clusterRmsd += r;
                }
                clusterRmsd /= nstr;
                if (writeClusterMappingFile_)
                {
                    for (int str = 0; str < nstr; ++str)
                    {
                        if ((str % 7 == 0) && str)
                        {
                            fprintf(clusterMappingFile, "\n");
                        }
                        int i1 = structure[str];
                        fprintf(clusterMappingFile, " %6d", frameData_.frameIndex[i1] + 1);
                    }
                    fprintf(clusterMappingFile, "\n");
                }
                std::vector<bool> doWriteStructures(clusters.size(), true);
                if (writeOutputStructures)
                {
                    if (useWriteNumCluster_)
                    {
                        for (int str = 0; str < nstr; ++str)
                        {
                            doWriteStructures[str] = false;
                        }
                    }
                    if ((i < (numClusterWrite_ + 1)) && (nstr < writeOverThisCount_))
                    {
                        auto fileName = formatString(outputBaseName.c_str(), i);
                        auto outputFile =
                                openClusterOutputFile(fileName, requirementsBuilder_, *top_, sel_);
                        for (int str = 0; str < nstr; ++str)
                        {
                            doWriteStructures[str] = true;
                            if (useRmsdCutoff_)
                            {
                                for (int i1 = 0; i1 < str && doWriteStructures[str]; ++i1)
                                {
                                    if (doWriteStructures[i1])
                                    {
                                        doWriteStructures[i] =
                                                (matrix->mat[structure[i1]][structure[str]] > rmsdCutoff_);
                                    }
                                }
                            }
                            if (doWriteStructures[str])
                            {
                                const int index = structure[str];
                                addCoordsToFile(str,
                                                outputFile.get(),
                                                frameData_.frames[index],
                                                atomIndexRef,
                                                frameData_.times[index],
                                                frameData_.boxes[index]);
                            }
                        }
                    }
                    if (writeAverageStructure_)
                    {
                        for (auto coord : averageCoordinates)
                        {
                            svmul(1.0 / nstr, coord, coord);
                        }
                    }
                    else
                    {
                        for (int atom = 0; atom < natoms; ++atom)
                        {
                            copy_rvec(frameData_.frames[midStructure][atom], averageCoordinates[atom]);
                        }
                        if (useLeastSquaresFitting_)
                        {
                            reset_x(atomIndexRef.size(),
                                    atomIndexRef.data(),
                                    natoms,
                                    nullptr,
                                    as_rvec_array(averageCoordinates.data()),
                                    masses_.data());
                        }
                    }
                    if (useLeastSquaresFitting_)
                    {
                        do_fit(natoms,
                               masses_.data(),
                               as_rvec_array(referenceCoordinates_.data()),
                               as_rvec_array(averageCoordinates.data()));
                    }
                    auto averageStructure =
                            openClusterOutputFile(clusterBaseName_, requirementsBuilder_, *top_, sel_);
                    addCoordsToFile(0,
                                    averageStructure.get(),
                                    averageCoordinates,
                                    atomIndexRef,
                                    frameData_.times[midStructure],
                                    frameData_.boxes[midStructure]);
                }
            }
        }
        clusterSizeDataHandle.finishData();
    }
    if (useBinaryValues_ && !runAnalysis)
    {
        convertBinaryToMaxRms(clusterMatrixHandle_, numFrames);
    }
}

} // namespace

const char ClusterInfo::name[]             = "cluster-rewrite";
const char ClusterInfo::shortDescription[] = "Clusters structures from trajectory";

TrajectoryAnalysisModulePointer ClusterInfo::create()
{
    return TrajectoryAnalysisModulePointer(new Cluster);
}

} // namespace analysismodules

} // namespace gmx
