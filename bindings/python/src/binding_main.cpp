#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>


#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/sites/spin_s_site.hpp"
#include "qkrylov/sites/fermion_site.hpp"
#include "qkrylov/sites/hubbard_site.hpp"
#include "qkrylov/sites/tj_site.hpp"
#include "qkrylov/models/models.hpp"

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/core/device.hpp"
#include "qkrylov/basis/basis.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/basis/fermion_basis.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"
#include "qkrylov/basis/tj_basis.hpp"
#include "qkrylov/basis/spin_s_basis.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace qkrylov;

static void bind_common(nb::module_& m) {
    
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

    nb::class_<Sector>(m, "Sector")
        .def(nb::init<>())
        .def_rw("sz", &Sector::sz)
        .def_rw("nup", &Sector::nup)
        .def_rw("ndn", &Sector::ndn)
        .def_rw("n", &Sector::n)
        .def_rw("nb", &Sector::nb)
        .def("unconstrained", &Sector::unconstrained);

    nb::class_<Device>(m, "Device")
        .def(nb::init<>())
        .def(nb::init<const std::string&>(), "device_string"_a)
        .def(nb::init<int>(), "device_id"_a)
        .def_ro("id", &Device::id)
        .def_static("is_gpu_build", &Device::is_gpu_build)
        .def_static("backend_name", &Device::backend_name)
        .def_static("gpu_count", &Device::gpu_count);

    
    nb::module_ m_models = m.def_submodule("models");
    nb::class_<qkrylov::models::Heisenberg1D, OpSum>(m_models, "Heisenberg1D")
        .def(nb::init<int, double, double, bool>(), "L"_a, "J"_a = 1.0, "Jz"_a = 1.0, "pbc"_a = false);

    nb::class_<qkrylov::models::TransverseFieldIsing1D, OpSum>(m_models, "TransverseFieldIsing1D")
        .def(nb::init<int, double, double, bool>(), "L"_a, "J"_a = 1.0, "h"_a = 1.0, "pbc"_a = false);

    nb::class_<Basis>(m, "Basis");

    nb::class_<SpinHalfBasis, Basis>(m, "SpinHalfBasis")
        .def(nb::init<int, const Sector&>(), "N"_a, "sector"_a = Sector())
        .def("size", &SpinHalfBasis::size)
        .def("state", &SpinHalfBasis::state)
        .def("index", &SpinHalfBasis::index)
        .def("contains", &SpinHalfBasis::contains)
        .def("nsites", &SpinHalfBasis::nsites);

    nb::class_<FermionBasis, Basis>(m, "FermionBasis")
        .def(nb::init<int, const Sector&>(), "N"_a, "sector"_a = Sector())
        .def("size", &FermionBasis::size)
        .def("state", &FermionBasis::state)
        .def("index", &FermionBasis::index)
        .def("contains", &FermionBasis::contains)
        .def("nsites", &FermionBasis::nsites);

    nb::class_<HubbardBasis, Basis>(m, "HubbardBasis")
        .def(nb::init<int, const Sector&>(), "N"_a, "sector"_a = Sector())
        .def("size", &HubbardBasis::size)
        .def("state", &HubbardBasis::state)
        .def("index", &HubbardBasis::index)
        .def("contains", &HubbardBasis::contains)
        .def("nsites", &HubbardBasis::nsites);

    nb::class_<TJBasis, Basis>(m, "TJBasis")
        .def(nb::init<int, const Sector&>(), "N"_a, "sector"_a = Sector())
        .def("size", &TJBasis::size)
        .def("state", &TJBasis::state)
        .def("index", &TJBasis::index)
        .def("contains", &TJBasis::contains)
        .def("nsites", &TJBasis::nsites);

    nb::class_<SpinSBasis, Basis>(m, "SpinSBasis")
        .def(nb::init<int, double, const Sector&>(), "N"_a, "S"_a = 0.5, "sector"_a = Sector())
        .def("size", &SpinSBasis::size)
        .def("state", &SpinSBasis::state)
        .def("index", &SpinSBasis::index)
        .def("contains", &SpinSBasis::contains)
        .def("nsites", &SpinSBasis::nsites)
        .def_prop_ro("spin", &SpinSBasis::spin)
        .def_prop_ro("dimension_per_site", &SpinSBasis::dimension_per_site);

    const char* names[] = {
        "Sector", "Device", "Basis", "SpinHalfBasis", "FermionBasis", "HubbardBasis", "TJBasis", "SpinSBasis"
    };
    for (const char* name : names) {
        m.attr((std::string(name) + "_FP32").c_str()) = m.attr(name);
        m.attr((std::string(name) + "_FP64").c_str()) = m.attr(name);
    }
}

void bind_fp32(nb::module_& m);
void bind_fp64(nb::module_& m);

NB_MODULE(_qkrylov_cpp, m) {
    bind_common(m);
    bind_fp32(m);
    bind_fp64(m);
}
