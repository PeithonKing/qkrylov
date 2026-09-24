#pragma once

#include "qkrylov/core/types.hpp"

namespace qkrylov {



struct LocalAction
{
    bool valid = false;

    StateID new_state = 0;

    ComplexDouble matrix_element = 0.0;
};

}
