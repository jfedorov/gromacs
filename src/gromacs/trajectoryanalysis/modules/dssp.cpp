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
 * Implements gmx::analysismodules::Trajectory.
 *
 * \author Sergey Gorelov <gorelov_sv@pnpi.nrcki.ru>
 * \author Anatoly Titov <titov_ai@pnpi.nrcki.ru>
 * \author Alexey Shvetsov <alexxyum@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "dssp.h"

#include <algorithm>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/selection/nbsearch.h"
#include "gromacs/analysisdata/modules/average.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/pdbio.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/selection.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/topology/atomprop.h"
#include "gromacs/topology/symtab.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysismodule.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/pleasecite.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/unique_cptr.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/selection/selection.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include <set>
#include <iterator>

namespace gmx
{

namespace analysismodules
{

namespace
{

/********************************************************************
 * Trajectory
 */

//! BackBone atom types
enum class backboneAtomTypes : size_t
{
    AtomCA = 0,
    AtomC  = 1,
    AtomO  = 2,
    AtomN  = 3,
    AtomH  = 4,
    Count
};
//! String values corresponding to backbone atom types
const gmx::EnumerationArray<backboneAtomTypes, const char*> backboneAtomTypeNames = {
    { "CA", "C", "O", "N", "H" }
};


class backboneAtomIndexes
{
public:
    void   setIndex(backboneAtomTypes atomTypeName, size_t atomIndex);
    size_t getIndex(backboneAtomTypes atomTypeName) const;

private:
    std::array<size_t, 5> _backBoneAtomIndexes{ 0, 0, 0, 0, 0 };
};


void backboneAtomIndexes::setIndex(backboneAtomTypes atomTypeName, size_t atomIndex)
{
    _backBoneAtomIndexes.at(static_cast<size_t>(atomTypeName)) = atomIndex;
}
size_t backboneAtomIndexes::getIndex(backboneAtomTypes atomTypeName) const
{
    return _backBoneAtomIndexes[static_cast<size_t>(atomTypeName)];
}


class Dssp : public TrajectoryAnalysisModule
{
public:
    Dssp();

    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;

    void optionsFinished(TrajectoryAnalysisSettings* settings) override;


    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;

    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;

    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    real                 cutoff_ = 1.0;
    Selection            sel_;
    AtomsDataPtr         atoms_;
    std::string          filename_;
    void                     calculateBends(const t_trxframe& fr, const t_pbc* pbc);
    void                     PatternSearch();
    void                     PrintOutput(int frnr, std::string name);
    std::vector<std::string> ResiNames;
    std::vector<std::size_t> AtomResi, SecondaryStructuresMap, v1, bendmap, breakmap;
    std::vector<backboneAtomIndexes> IndexMap;
    std::vector<std::vector<bool>> HBondsMap;
    std::vector<std::vector<std::size_t>> nturnsmap, Bridges, AntiBridges, v2;
    std::vector<std::vector<std::vector<std::size_t>>> v3;
    bool  isHbondExist(const backboneAtomIndexes& resA, const backboneAtomIndexes& resB, const t_trxframe& fr, const t_pbc*      pbc);
    float CalculateAtomicDistances(const int& A, const int& B, const t_trxframe& fr, const t_pbc* pbc);
};

Dssp::Dssp() {}

void Dssp::calculateBends(const t_trxframe& fr, const t_pbc* pbc)
{
    const float benddegree{ 70.0 }, maxdist{ 2.5 };
    float       degree{ 0 }, vdist{ 0 }, vprod{ 0 };
    gmx::RVec   a{ 0, 0, 0 }, b{ 0, 0, 0 };
    bendmap.resize(0);
    breakmap.resize(0);
    bendmap.resize(IndexMap.size(), 0);
    breakmap.resize(IndexMap.size(), 0);
    for (size_t i{ 0 }; i < IndexMap.size() - 1; ++i)
    {
        if (CalculateAtomicDistances(IndexMap[i].getIndex(backboneAtomTypes::AtomC), IndexMap[i + 1].getIndex(backboneAtomTypes::AtomN), fr, pbc) > maxdist)
        {
            breakmap[i]     = 1;
            breakmap[i + 1] = 1;
        }
    }
    for (size_t i{ 2 }; i < IndexMap.size() - 2; ++i)
    {
        if (breakmap[i - 1] || breakmap[i] || breakmap[i + 1])
        {
            continue;
        }
        for (int j{ 0 }; j < 3; ++j)
        {
            a[j] = fr.x[IndexMap[i].getIndex(backboneAtomTypes::AtomCA)][j] - fr.x[IndexMap[i - 2].getIndex(backboneAtomTypes::AtomCA)][j];
            b[j] = fr.x[IndexMap[i + 2].getIndex(backboneAtomTypes::AtomCA)][j] - fr.x[IndexMap[i].getIndex(backboneAtomTypes::AtomCA)][j];
        }
        vdist = (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
        vprod = CalculateAtomicDistances(IndexMap[i - 2].getIndex(backboneAtomTypes::AtomCA), IndexMap[i].getIndex(backboneAtomTypes::AtomCA), fr, pbc) * gmx::c_angstrom
                / gmx::c_nano * CalculateAtomicDistances(IndexMap[i].getIndex(backboneAtomTypes::AtomCA), IndexMap[i + 2].getIndex(backboneAtomTypes::AtomCA), fr, pbc)
                * gmx::c_angstrom / gmx::c_nano;
        degree = std::acos(vdist / vprod) * gmx::c_rad2Deg;
        if (degree > benddegree)
        {
            bendmap[i] = 1;
        }
    }
}

void Dssp::PatternSearch()
{
    v1.resize(0);
    v1.resize(HBondsMap.size(), false);
    SecondaryStructuresMap.resize(0);
    SecondaryStructuresMap.resize(HBondsMap.size(), 0);
    nturnsmap.resize(0);
    nturnsmap.resize(6, v1);
    v1.resize(0);
    Bridges.resize(0);
    Bridges.resize(HBondsMap.size(), v1);
    AntiBridges.resize(0);
    AntiBridges.resize(HBondsMap.size(), v1);
    for (size_t i{ 0 }; i < HBondsMap.front().size(); ++i)
    {
        if (bendmap[i])
        {
            SecondaryStructuresMap[i] = 7;
        }
    }
    for (int n{ 3 }; n <= 5; ++n)
    {
        for (size_t i{ 0 }; i + n < HBondsMap.front().size(); ++i)
        {
            if (HBondsMap[i][i + n])
            {
                nturnsmap[n - 3][i] = n;
                for (int j{ 1 }; j < n; ++j)
                {
                    if ((i + j) < nturnsmap.front().size())
                    {
                        SecondaryStructuresMap[i + j] = 6;
                    }
                }
            }
            // Fuck go back
            if (HBondsMap[i + n][i])
            {
                nturnsmap[n][i + n] = n;
                for (int j{ 1 }; j < n; ++j)
                {
                    if ((i + n - j) >= 0)
                    {
                        SecondaryStructuresMap[i + n - j] = 6;
                    }
                }
            }
        }
    }
    for (size_t i{ 1 }; i < HBondsMap.front().size() - 1; ++i)
    {
        for (size_t j{ 1 }; j < HBondsMap.front().size() - 1; ++j)
        {
            if (abs(i - j) > 2)
            {
                if ((HBondsMap[i - 1][j] && HBondsMap[j][i + 1])
                    || (HBondsMap[j - 1][i] && HBondsMap[i][j + 1]))
                {
                    Bridges[i].push_back(j);
                }
                if ((HBondsMap[i][j] && HBondsMap[j][i])
                    || (HBondsMap[i - 1][j + 1] && HBondsMap[j - 1][i + 1]))
                {
                    AntiBridges[i].push_back(j);
                }
            }
        }
    }
    for (int n{ 3 }; n <= 5; ++n)
    {
        for (size_t i{ 1 }; i < HBondsMap.front().size() - 1; ++i)
        {
            if (nturnsmap[n - 3][i - 1] && nturnsmap[n - 3][i])
            {
                for (int j{ 0 }; j < n; ++j)
                {
                    if ((j + i) < SecondaryStructuresMap.size())
                    {
                        SecondaryStructuresMap[j + i] = n;
                    }
                }
            }
        }
        for (int i = (HBondsMap.front().size() - 2); i >= 0; --i)
        {
            if (nturnsmap[n][i] && nturnsmap[n][i + 1])
            {
                for (int j{ 0 }; j < n; ++j)
                {
                    if ((i - j) >= 0)
                    {
                        SecondaryStructuresMap[i - j] = n;
                    }
                }
            }
        }

        if (n == 3)
        {
            for (size_t i{ 0 }; i < HBondsMap.front().size(); ++i)
            {
                if ((Bridges[i].size() || AntiBridges[i].size()))
                {
                    SecondaryStructuresMap[i] = 1;
                }
            }
            for (size_t i{ 2 }; i < HBondsMap.front().size() - 3; ++i)
            {
                for (int j = { i - 2 }; j <= (i + 2); ++j)
                {
                    if (j == i)
                    {
                        continue;
                    }
                    else
                    {
                        if (Bridges[i].size() || Bridges[j].size())
                        {
                            for (size_t i_resi{ 0 }; i_resi < Bridges[i].size(); ++i_resi)
                            {
                                for (size_t j_resi{ 0 }; j_resi < Bridges[j].size(); ++j_resi)
                                {
                                    if (abs(static_cast<int>(Bridges[i][i_resi])
                                            - static_cast<int>(Bridges[j][j_resi]))
                                        && (abs(static_cast<int>(Bridges[i][i_resi])
                                                - static_cast<int>(Bridges[j][j_resi]))
                                            < 5))
                                    {
                                        if (j < i)
                                        {
                                            for (int k{ 0 }; k <= abs(i - j); ++k)
                                            {
                                                SecondaryStructuresMap[j + k] = 2;
                                            }
                                        }
                                        else
                                        {
                                            for (int k{ 0 }; k <= abs(j - i); ++k)
                                            {
                                                SecondaryStructuresMap[i + k] = 2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (AntiBridges[i].size() || AntiBridges[j].size())
                        {
                            for (size_t i_resi{ 0 }; i_resi < AntiBridges[i].size(); ++i_resi)
                            {
                                for (size_t j_resi{ 0 }; j_resi < AntiBridges[j].size(); ++j_resi)
                                {
                                    if (abs(static_cast<int>(AntiBridges[i][i_resi])
                                            - static_cast<int>(AntiBridges[j][j_resi]))
                                        && (abs(static_cast<int>(AntiBridges[i][i_resi])
                                                - static_cast<int>(AntiBridges[j][j_resi]))
                                            < 5))
                                    {
                                        if (j < i)
                                        {
                                            for (int k{ 0 }; k <= abs(i - j); ++k)
                                            {
                                                SecondaryStructuresMap[j + k] = 2;
                                            }
                                        }
                                        else
                                        {
                                            for (int k{ 0 }; k <= abs(j - i); ++k)
                                            {
                                                SecondaryStructuresMap[i + k] = 2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Dssp::PrintOutput(int frnr, std::string name)
{
    FILE* file{ std::fopen((name).c_str(), "a") };
    for (size_t i{ 0 }; i < SecondaryStructuresMap.size(); ++i)
    {
        switch (SecondaryStructuresMap[i])
        {
            case 0: std::fprintf(file, "~"); break;
            case 1: std::fprintf(file, "B"); break;
            case 2: std::fprintf(file, "E"); break;
            case 3: std::fprintf(file, "G"); break;
            case 4: std::fprintf(file, "H"); break;
            case 5: std::fprintf(file, "I"); break;
            case 6: std::fprintf(file, "T"); break;
            case 7: std::fprintf(file, "S"); break;
            default: std::fprintf(file, "?"); break;
        }

        if (i != SecondaryStructuresMap.size() - 1)
        {
            if (breakmap[i] && breakmap[i + 1])
            {
                std::fprintf(file, "=");
            }
        }
    }
    for (size_t i{ 0 }; i < SecondaryStructuresMap.size() - 1; ++i) {}
    std::fprintf(file, "\n");
    std::fclose(file);
}

bool Dssp::isHbondExist(const backboneAtomIndexes& resA,
                                 const backboneAtomIndexes& resB,
                                 const t_trxframe& fr,
                                 const t_pbc*      pbc)
{
    /*
     * DSSP uses eq from dssp 2.x
     * kCouplingConstant = 27.888,  //  = 332 * 0.42 * 0.2
     * E = k * (1/rON + 1/rCH - 1/rOH - 1/rCN) where CO comes from one AA and NH from another
     * if R is in A
     * Hbond if E < -0.5
     */
    const float kCouplingConstant = 27.888;
    const float HBondEnergyCutOff{ -0.5 }; // from dssp
    const float minimalAtomDistance{ 0.5 }, minimalCAdistance{ 9.0 }, minEnergy{ -9.9 }; // from original dssp algo in A
    float HbondEnergy{ 0 };
    float distanceON{ 0 }, distanceCH{ 0 }, distanceOH{ 0 }, distanceCN{ 0 };
    distanceON = CalculateAtomicDistances(resA.getIndex(backboneAtomTypes::AtomO), resB.getIndex(backboneAtomTypes::AtomN), fr, pbc);
    distanceCH = CalculateAtomicDistances(resA.getIndex(backboneAtomTypes::AtomC), resB.getIndex(backboneAtomTypes::AtomH), fr, pbc);
    distanceOH = CalculateAtomicDistances(resA.getIndex(backboneAtomTypes::AtomO), resB.getIndex(backboneAtomTypes::AtomH), fr, pbc);
    distanceCN = CalculateAtomicDistances(resA.getIndex(backboneAtomTypes::AtomC), resB.getIndex(backboneAtomTypes::AtomN), fr, pbc);

    if (resA.getIndex(backboneAtomTypes::AtomC) && resA.getIndex(backboneAtomTypes::AtomO) && resB.getIndex(backboneAtomTypes::AtomN) && resB.getIndex(backboneAtomTypes::AtomH))
    {
        if (CalculateAtomicDistances(resA.getIndex(backboneAtomTypes::AtomCA), resB.getIndex(backboneAtomTypes::AtomCA), fr, pbc) < minimalCAdistance)
        {
            if ((distanceON < minimalAtomDistance) || (distanceCH < minimalAtomDistance) || (distanceOH < minimalAtomDistance) || (distanceCN < minimalAtomDistance))
            {
                HbondEnergy = minEnergy;
            }
            else
            {
                HbondEnergy = kCouplingConstant * ((1 / distanceON) + (1 / distanceCH) - (1 / distanceOH) - (1 / distanceCN));
            }

            if (HbondEnergy < HBondEnergyCutOff)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }
}

float Dssp::CalculateAtomicDistances(const int& A, const int& B, const t_trxframe& fr, const t_pbc* pbc)
{
    gmx::RVec r{ 0, 0, 0 };
    pbc_dx(pbc, fr.x[B], fr.x[A], r.as_vec());
    return r.norm() * gmx::c_nm2A;
}

void Dssp::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] todo",
    };
    options->addOption(
            StringOption("o").store(&filename_).required().defaultValue("SSP.dat").description("Filename for DSSP output"));
    options->addOption(RealOption("cutoff").store(&cutoff_).required().defaultValue(1.0).description(
            "cutoff for neighbour search"));
    options->addOption(
            SelectionOption("sel").store(&sel_).required().defaultSelectionText("Protein").description(
                    "Group for DSSP"));
    settings->setHelpText(desc);
}


void Dssp::optionsFinished(TrajectoryAnalysisSettings* settings) {}


void Dssp::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
    backboneAtomIndexes      _backboneAtoms;
    int                      i{ 0 };
    int resicompare{ top.atoms()->atom[static_cast<std::size_t>(*(sel_.atomIndices().begin()))].resind };
    AtomResi.resize(0);
    IndexMap.resize(0);
    IndexMap.push_back(_backboneAtoms);
    for (gmx::ArrayRef<const int>::iterator ai{ sel_.atomIndices().begin() };
         (ai < sel_.atomIndices().end());
         ++ai)
    {

        if (resicompare != top.atoms()->atom[static_cast<std::size_t>(*ai)].resind)
        {
            ++i;
            resicompare = top.atoms()->atom[static_cast<std::size_t>(*ai)].resind;
            IndexMap.push_back(_backboneAtoms);
        }
        AtomResi.push_back(i);
        std::string atomname(*(top.atoms()->atomname[static_cast<std::size_t>(*ai)]));
        if (atomname == backboneAtomTypeNames[backboneAtomTypes::AtomCA])
        {
            IndexMap[i].setIndex(backboneAtomTypes::AtomCA, static_cast<std::size_t>(*ai));
        }
        else if (atomname == backboneAtomTypeNames[backboneAtomTypes::AtomC])
        {
            IndexMap[i].setIndex(backboneAtomTypes::AtomC, static_cast<std::size_t>(*ai));
        }
        else if (atomname == backboneAtomTypeNames[backboneAtomTypes::AtomO])
        {
            IndexMap[i].setIndex(backboneAtomTypes::AtomO, static_cast<std::size_t>(*ai));
        }
        else if (atomname == backboneAtomTypeNames[backboneAtomTypes::AtomN])
        {
            IndexMap[i].setIndex(backboneAtomTypes::AtomN, static_cast<std::size_t>(*ai));
        }
        else if (atomname == backboneAtomTypeNames[backboneAtomTypes::AtomH])
        {
            IndexMap[i].setIndex(backboneAtomTypes::AtomH, static_cast<std::size_t>(*ai));
        }
    }
}

void Dssp::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata)
{
    // store positions of CA atoms to use them for nbSearch
    std::vector<gmx::RVec> positionsCA_;
    for (auto i{ 0 }; i < IndexMap.size(); ++i)
    {
        positionsCA_.push_back(fr.x[IndexMap[i].getIndex(backboneAtomTypes::AtomCA)]);
    }
    // resize HBondsMap
    HBondsMap.resize(0);
    HBondsMap.resize(IndexMap.size(), std::vector<bool>(IndexMap.size(), 0));

    // Init nbSearch
    AnalysisNeighborhood nb_;
    nb_.setCutoff(cutoff_);
    AnalysisNeighborhoodPositions       nbPos_(positionsCA_);
    gmx::AnalysisNeighborhoodSearch     start      = nb_.initSearch(pbc, nbPos_);
    gmx::AnalysisNeighborhoodPairSearch pairSearch = start.startPairSearch(nbPos_);
    gmx::AnalysisNeighborhoodPair       pair;
    while (pairSearch.findNextPair(&pair))
    {
        HBondsMap[pair.refIndex()][pair.testIndex()] = isHbondExist(IndexMap[pair.refIndex()],
                         IndexMap[pair.testIndex()],
                         fr,
                         pbc);
    }

    calculateBends(fr, pbc);
    PatternSearch();
    PrintOutput(frnr, filename_);
}

void Dssp::finishAnalysis(int /*nframes*/)
{
}


void Dssp::writeOutput() {
}

} // namespace

const char DsspInfo::name[] = "dssp";
const char DsspInfo::shortDescription[] =
        "Calculate protein secondary structure via DSSP algo";

TrajectoryAnalysisModulePointer DsspInfo::create()
{
    return TrajectoryAnalysisModulePointer(new Dssp);
}

} // namespace analysismodules

} // namespace gmx
