import re

with open("bindings/python/src/binding_main.cpp", "r") as f:
    code = f.read()

old_sector = """    nb::class_<Sector>(m, "Sector")
        .def(nb::init<>())
        .def_rw("use_sz", &Sector::use_sz)
        .def_rw("sz2", &Sector::sz2)
        .def_rw("use_nup", &Sector::use_nup)
        .def_rw("use_ndn", &Sector::use_ndn)
        .def_rw("nup", &Sector::nup)
        .def_rw("ndn", &Sector::ndn)
        .def_rw("use_n", &Sector::use_n)
        .def_rw("n", &Sector::n)
        .def_rw("use_nb", &Sector::use_nb)
        .def_rw("nb", &Sector::nb);"""

new_sector = """    nb::class_<Sector>(m, "Sector")
        .def(nb::init<>())
        .def_rw("sz", &Sector::sz)
        .def_rw("nup", &Sector::nup)
        .def_rw("ndn", &Sector::ndn)
        .def_rw("n", &Sector::n)
        .def_rw("nb", &Sector::nb)
        .def("unconstrained", &Sector::unconstrained);"""

code = code.replace(old_sector, new_sector)

with open("bindings/python/src/binding_main.cpp", "w") as f:
    f.write(code)
