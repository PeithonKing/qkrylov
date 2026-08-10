#define QKRYLOV_DOUBLE_PRECISION
#include "binding_impl.hpp"

void bind_fp64(nb::module_& m) {
    bind_impl(m, "_FP64");
}
