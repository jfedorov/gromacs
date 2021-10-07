/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
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
/*! \file
 * \brief Exports TPR I/O tools during Python module initialization.
 *
 * Provides _gmxapi.SimulationParameters and _gmxapi.TprFile classes, as well
 * as module functions read_tprfile, write_tprfile, copy_tprfile, and rewrite_tprfile.
 *
 * TprFile is a Python object that holds a gmxapicompat::TprReadHandle.
 *
 * SimulationParameters is the Python type for data sources providing the
 * simulation parameters aspect of input to simulation operations.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 * \ingroup module_python
 */

#include "gmxapi/exceptions.h"

#include "gmxapi/gmxapicompat.h"
#include "gmxapi/compat/mdparams.h"
#include "gmxapi/compat/tpr.h"

#include "module.h"

using gmxapi::GmxapiType;

namespace gmxpy
{


void detail::export_tprfile(pybind11::module& module)
{
    namespace py = pybind11;
    using gmxapicompat::GmxMdParams;
    using gmxapicompat::readTprFile;
    using gmxapicompat::StructureSource;
    using gmxapicompat::TprReadHandle;

    py::class_<GmxMdParams> mdparams(module, "SimulationParameters");
    // We don't want Python users to create invalid params objects, so don't
    // export a constructor until we can default initialize a valid one.
    //    mdparams.def(py::init());
    mdparams.def(
            "extract",
            [](const GmxMdParams& self) {
                py::dict dictionary;
                for (const auto& key : gmxapicompat::keys(self))
                {
                    try
                    {
                        // TODO: More complete typing and dispatching.
                        // This only handles the two types described in the initial implementation.
                        // Less trivial types (strings, maps, arrays) warrant additional
                        // design discussion before being exposed through an interface
                        // like this one.
                        // Also reference https://gitlab.com/gromacs/gromacs/-/issues/2993

                        // We can use templates and/or tag dispatch in a more complete
                        // future implementation.
                        const auto& paramType = gmxapicompat::mdParamToType(key);
                        if (paramType == GmxapiType::FLOAT64)
                        {
                            dictionary[key.c_str()] = extractParam(self, key, double());
                        }
                        else if (paramType == GmxapiType::INT64)
                        {
                            dictionary[key.c_str()] = extractParam(self, key, int64_t());
                        }
                    }
                    catch (const gmxapicompat::ValueError& e)
                    {
                        throw gmxapi::ProtocolError(std::string("Unknown parameter: ") + key);
                    }
                }
                return dictionary;
            },
            "Get a dictionary of the parameters.");

    // Overload a setter for each known type and None
    mdparams.def(
            "set",
            [](GmxMdParams* self, const std::string& key, int64_t value) {
                gmxapicompat::setParam(self, key, value);
            },
            py::arg("key").none(false),
            py::arg("value").none(false),
            "Use a dictionary to update simulation parameters.");
    mdparams.def(
            "set",
            [](GmxMdParams* self, const std::string& key, double value) {
                gmxapicompat::setParam(self, key, value);
            },
            py::arg("key").none(false),
            py::arg("value").none(false),
            "Use a dictionary to update simulation parameters.");
    mdparams.def(
            "set",
            [](GmxMdParams* self, const std::string& key, py::none) {
                // unsetParam(self, key);
                throw gmxapi::NotImplementedError(
                        "Un-setting parameters is not currently supported.");
            },
            py::arg("key").none(false),
            py::arg("value"),
            "Use a dictionary to update simulation parameters.");

    py::class_<StructureSource> structureSource(module, "SimulationStructure", py::buffer_protocol());
    // Note that Python extends the life of the buffer provider while the buffer_info is held,
    // with corresponding side effects for any resources held by the StructureSource object.
    structureSource.def_buffer([](StructureSource& source) -> py::buffer_info {
        gmxapicompat::CoordinatesBuffer buf;
        try
        {
            buf = gmxapicompat::coordinates(source, float());
        }
        catch (gmxapicompat::PrecisionError&)
        {
            buf = gmxapicompat::coordinates(source, double());
        }

        std::string format_descriptor;
        if (buf.itemType != gmxapi::GmxapiType::FLOAT64 || (buf.itemSize != 4 && buf.itemSize != 8)
            || buf.shape.size() != 2)
        {
            throw gmxapi::ProtocolError("Bug: Expected Nx3 floating point numbers.");
        }
        if (buf.itemSize == 4)
        {
            format_descriptor = py::format_descriptor<float>::format();
        }
        else
        {
            assert(buf.itemSize == 8);
            format_descriptor = py::format_descriptor<double>::format();
        }
        // Reference: https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#buffer-protocol
        return py::buffer_info(buf.ptr,
                               buf.itemSize,
                               format_descriptor,
                               2,
                               { buf.shape[0], buf.shape[1] },
                               { buf.strides[0], buf.itemSize },
                               true // readonly
        );
    });

    py::class_<gmxapicompat::TprBuilder> tprBuilder(module, "TprBuilder");
    tprBuilder.def(py::init(&gmxapicompat::editTprFile));
    tprBuilder.def("set_coordinates", [](gmxapicompat::TprBuilder* self, py::buffer buf) {
        // "Acquire" the buffer. (Destructor "release"s the buffer.)
        py::buffer_info info     = buf.request();
        const size_t    itemSize = info.itemsize;

        // Confirm that the buffer is an Nx3 array of floating point numbers of
        // appropriate precision.
        if (info.format != py::format_descriptor<float>::format() || info.itemsize != self->get_precision()
            || info.shape.size() != 2 || info.shape[0] == 0 || info.shape[1] != 3)
        {
            py::value_error("Input is not compatible with existing simulation input data.");
        }

        auto shape   = { static_cast<size_t>(info.shape[0]), static_cast<size_t>(info.shape[1]) };
        auto strides = { static_cast<size_t>(info.strides[0]), static_cast<size_t>(info.strides[1]) };

        // Re-wrap the buffer as a CoordinatesBuffer and pass to
        // ::gmxapicompat to update the structure data.
        self->set(gmxapicompat::CoordinatesBuffer{
                info.ptr, gmxapi::GmxapiType::FLOAT64, itemSize, 2, shape, strides });
    });
    tprBuilder.def("write", &gmxapicompat::TprBuilder::write);


    py::class_<TprReadHandle> tprfile(module, "TprFile");
    tprfile.def("params", [](const TprReadHandle& self) {
        auto params = gmxapicompat::getMdParams(self);
        return params;
    });
    tprfile.def("coordinates", [](const TprReadHandle& self) {
        auto structure = gmxapicompat::getStructureSource(self);
        return structure;
    });

    module.def("read_tprfile",
               &readTprFile,
               py::arg("filename"),
               "Get a handle to a TPR file resource for a given file name.");

    module.def(
            "write_tprfile",
            [](std::string filename, const GmxMdParams& parameterObject) {
                auto tprReadHandle = gmxapicompat::getSourceFileHandle(parameterObject);
                auto params        = gmxapicompat::getMdParams(tprReadHandle);
                auto structure     = gmxapicompat::getStructureSource(tprReadHandle);
                auto state         = gmxapicompat::getSimulationState(tprReadHandle);
                auto topology      = gmxapicompat::getTopologySource(tprReadHandle);
                gmxapicompat::writeTprFile(filename, *params, *structure, *state, *topology);
            },
            py::arg("filename").none(false),
            py::arg("parameters"),
            "Write a new TPR file with the provided data.");

    module.def(
            "copy_tprfile",
            [](const gmxapicompat::TprReadHandle& input, std::string outFile) {
                return gmxapicompat::copy_tprfile(input, outFile);
            },
            py::arg("source"),
            py::arg("destination"),
            "Copy a TPR file from ``source`` to ``destination``.");

    module.def(
            "rewrite_tprfile",
            [](std::string input, std::string output, double end_time) {
                return gmxapicompat::rewrite_tprfile(input, output, end_time);
            },
            py::arg("source"),
            py::arg("destination"),
            py::arg("end_time"),
            "Copy a TPR file from ``source`` to ``destination``, replacing `nsteps` with "
            "``end_time``.");
}

} // end namespace gmxpy
