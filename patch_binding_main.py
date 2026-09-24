import re

with open("bindings/python/src/binding_main.cpp", "r") as f:
    content = f.read()

bind_str = """
    nb::class_<OperatorFactor>(m, "OperatorFactor")
        .def(nb::init<std::string, int>(), "op"_a, "site"_a)
        .def_rw("op", &OperatorFactor::op)
        .def_rw("site", &OperatorFactor::site);

    nb::class_<OperatorTerm>(m, "OperatorTerm")
        .def(nb::init<>())
        .def_rw("coeff", &OperatorTerm::coeff)
        .def_rw("factors", &OperatorTerm::factors);
        
    nb::class_<LocalOp>(m, "LocalOp")
        .def(nb::init<std::string, int>(), "name"_a, "site"_a)
        .def_rw("name", &LocalOp::name)
        .def_rw("site", &LocalOp::site);
        
    nb::class_<OpSumExpr>(m, "OpSumExpr")
        .def(nb::init<const OperatorTerm&>())
        .def_rw("terms", &OpSumExpr::terms);

    nb::class_<OpSum>(m, "OpSum")
        .def(nb::init<>())
        .def("add_term", &OpSum::add_term)
        .def("__iadd__", [](OpSum& os, nb::tuple tuple) {
            if (tuple.size() < 3 || tuple.size() % 2 == 0) {
                throw std::invalid_argument("OpSum += requires (coeff, op1, site1, [op2, site2, ...]) with odd tuple length >= 3");
            }
            OperatorTerm term;
            term.coeff = nb::cast<ComplexDouble>(tuple[0]);
            for (size_t i = 1; i < tuple.size(); i += 2) {
                term.factors.push_back({nb::cast<std::string>(tuple[i]), nb::cast<int>(tuple[i+1])});
            }
            os.add_term(term);
            return &os;
        })
        .def("__iadd__", [](OpSum& os, const LocalOp& op) {
            OperatorTerm t; t.coeff = 1.0; t.factors.push_back({op.name, op.site});
            os.add_term(t); return &os;
        })
        .def("__iadd__", [](OpSum& os, const OperatorTerm& term) {
            os.add_term(term); return &os;
        })
        .def("__iadd__", [](OpSum& os, const OpSumExpr& expr) {
            for(auto& t : expr.terms) os.add_term(t); return &os;
        })
        .def("clear", &OpSum::clear)
        .def("size", &OpSum::size)
        .def("terms", &OpSum::terms);

    nb::class_<Site>(m, "Site");

    nb::class_<SpinHalfSite, Site>(m, "SpinHalfSite")
        .def(nb::init<>());

    nb::class_<SpinSSite, Site>(m, "SpinSSite")
        .def(nb::init<double>(), "S"_a = 0.5)
        .def_prop_ro("spin", &SpinSSite::spin)
        .def_prop_ro("dimension_per_site", &SpinSSite::dimension_per_site);

    nb::class_<FermionSite, Site>(m, "FermionSite")
        .def(nb::init<>());

    nb::class_<HubbardSite, Site>(m, "HubbardSite")
        .def(nb::init<>());

    nb::class_<TJSite, Site>(m, "TJSite")
        .def(nb::init<>());
"""

content = content.replace('nb::class_<Sector>(m, "Sector")', bind_str + '\n    nb::class_<Sector>(m, "Sector")')

with open("bindings/python/src/binding_main.cpp", "w") as f:
    f.write(content)
