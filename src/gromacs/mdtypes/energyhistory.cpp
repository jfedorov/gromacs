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

#include "gmxpre.h"

#include "energyhistory.h"

#include "gromacs/utility/stringutil.h"

#include "checkpointdata.h"

//! \cond INTERNAL
// mirroring the \cond from energyhistory.h to avoid Doxygen errors

enum class DeltaHHistoryCheckpointVersion
{
    Base
};

template<gmx::CheckpointDataOperation operation>
void delta_h_history_t::doCheckpoint(gmx::CheckpointData checkpointData)
{
    auto version = DeltaHHistoryCheckpointVersion::Base;
    checkpointData.scalar<operation>("version", &version);

    auto numDeltaH = static_cast<int64_t>(dh.size());
    checkpointData.scalar<operation>("numDeltaH", &numDeltaH);
    if (operation == gmx::CheckpointDataOperation::Read)
    {
        dh.resize(numDeltaH);
    }
    checkpointData.scalar<operation>("start_time", &start_time);
    checkpointData.scalar<operation>("start_lambda", &start_lambda);
    checkpointData.scalar<operation>("start_lambda_set", &start_lambda_set);
    for (std::size_t idx = 0; idx < dh.size(); ++idx)
    {
        auto vecSize = static_cast<int64_t>(dh[idx].size());
        checkpointData.scalar<operation>(gmx::formatString("vecSize %lu", idx), &vecSize);
        if (operation == gmx::CheckpointDataOperation::Read)
        {
            dh[idx].resize(vecSize);
        }
        checkpointData.arrayRef<operation>(gmx::formatString("vec %lu", idx),
                                           gmx::makeCheckpointArrayRef<operation>(dh[idx]));
    }
}

enum class EnergyHistoryCheckpointVersion
{
    Base
};

template<gmx::CheckpointDataOperation operation>
void energyhistory_t::doCheckpoint(gmx::CheckpointData checkpointData)
{
    auto version = EnergyHistoryCheckpointVersion::Base;
    checkpointData.scalar<operation>("version", &version);

    bool useCheckpoint = (nsum <= 0 && nsum_sim <= 0);
    checkpointData.scalar<operation>("useCheckpoint", &useCheckpoint);

    if (!useCheckpoint)
    {
        return;
    }

    // lambda expression allowing to checkpoint vector size and resize if reading
    auto checkpointVectorSize = [&checkpointData](const std::string& name, std::vector<double>& vector) {
        auto size = static_cast<int64_t>(vector.size());
        checkpointData.scalar<operation>(name, &size);
        if (operation == gmx::CheckpointDataOperation::Read)
        {
            vector.resize(size);
        }
    };

    checkpointVectorSize("enerAveSize", ener_ave);
    checkpointVectorSize("enerSumSize", ener_sum);
    checkpointVectorSize("enerSumSimSize", ener_sum_sim);

    checkpointData.scalar<operation>("nsteps", &nsteps);
    checkpointData.scalar<operation>("nsteps_sim", &nsteps_sim);

    checkpointData.scalar<operation>("nsum", &nsum);
    checkpointData.scalar<operation>("nsum_sim", &nsum_sim);

    auto hasForeignLambdas = (deltaHForeignLambdas != nullptr);
    checkpointData.scalar<operation>("has foreign lambdas", &hasForeignLambdas);
    if (hasForeignLambdas && deltaHForeignLambdas == nullptr)
    {
        deltaHForeignLambdas = std::make_unique<delta_h_history_t>();
    }

    if (nsum > 0)
    {
        checkpointData.arrayRef<operation>("ener_ave", gmx::makeCheckpointArrayRef<operation>(ener_ave));
        checkpointData.arrayRef<operation>("ener_sum", gmx::makeCheckpointArrayRef<operation>(ener_sum));
    }
    if (nsum_sim > 0)
    {
        checkpointData.arrayRef<operation>("ener_sum_sim",
                                           gmx::makeCheckpointArrayRef<operation>(ener_sum_sim));
    }
    if (hasForeignLambdas)
    {
        deltaHForeignLambdas->doCheckpoint<operation>(
                checkpointData.subCheckpointData<operation>("deltaHForeignLambdas"));
    }
}

// explicit template instatiation
template void energyhistory_t::doCheckpoint<gmx::CheckpointDataOperation::Read>(gmx::CheckpointData checkpointData);
template void energyhistory_t::doCheckpoint<gmx::CheckpointDataOperation::Write>(gmx::CheckpointData checkpointData);


//! \endcond
