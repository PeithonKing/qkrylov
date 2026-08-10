#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_fp32(nb::module_& m);
void bind_fp64(nb::module_& m);

NB_MODULE(_qkrylov_cpp, m) {
    bind_fp32(m);
    bind_fp64(m);
}
