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
 *
 * \brief Defines the MD Graph class
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 *
 * \ingroup module_mdlib
 */

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.cuh"
#include "gromacs/utility/gmxmpi.h"

#include "mdgraph_gpu.h"


// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                                                     \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess)                                                                \
        {                                                                                    \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(0);                                                                         \
        }                                                                                    \
    }

namespace gmx
{

MdGraph::MdGraph()
{
    eventGraph = new GpuEventSynchronizer();
}

void MdGraph::startGraphCapture(const DeviceStream& deviceStream)
{
    cudaStreamBeginCapture(deviceStream.stream(), cudaStreamCaptureModeGlobal);
    cudaCheckError();
};


void MdGraph::endGraphCapture(const DeviceStream& deviceStream)
{
    cudaStreamEndCapture(deviceStream.stream(), &graph_);
    cudaCheckError();
    if(!updateGraph_)
    {
        cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0);
    }
    else
    {
        cudaGraphNode_t hErrorNode_out;
        cudaGraphExecUpdateResult updateResult_out;
        cudaGraphExecUpdate (instance_, graph_, &hErrorNode_out, &updateResult_out );
    }
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    int size=1;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size==1)
    {
        updateGraph_=true;
    }
};


void MdGraph::launchGraph(const DeviceStream& deviceStream)
{
    cudaGraphLaunch(instance_, deviceStream.stream());
    cudaCheckError();
};

void MdGraph::syncGpu()
{
    cudaDeviceSynchronize();
}

}
