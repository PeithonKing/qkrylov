#define QKRYLOV_SINGLE_PRECISION
#include "binding_impl.hpp"

void bind_fp32(nb::module_& m) {
    bind_impl(m, "_FP32");
}
