import re

with open("bindings/python/src/binding_main.cpp", "r") as f:
    content = f.read()

# Need to include headers
headers = """
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/sites/spins_site.hpp"
#include "qkrylov/sites/fermion_site.hpp"
#include "qkrylov/sites/hubbard_site.hpp"
#include "qkrylov/sites/tj_site.hpp"
#include "qkrylov/models/models.hpp"
"""
content = content.replace('#include "qkrylov/symmetry/sector.hpp"', headers + '\n#include "qkrylov/symmetry/sector.hpp"')

bind_str = """
    nb::module_ m_models = m.def_submodule("models");
    nb::class_<qkrylov::models::Heisenberg1D, OpSum>(m_models, "Heisenberg1D")
        .def(nb::init<int, double, double, bool>(), "L"_a, "J"_a = 1.0, "Jz"_a = 1.0, "pbc"_a = false);

    nb::class_<qkrylov::models::TransverseFieldIsing1D, OpSum>(m_models, "TransverseFieldIsing1D")
        .def(nb::init<int, double, double, bool>(), "L"_a, "J"_a = 1.0, "h"_a = 1.0, "pbc"_a = false);
"""

content = content.replace('nb::class_<Basis>(m, "Basis");', bind_str + '\n    nb::class_<Basis>(m, "Basis");')

with open("bindings/python/src/binding_main.cpp", "w") as f:
    f.write(content)
