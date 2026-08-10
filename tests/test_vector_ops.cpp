#include <iostream>
#include <cassert>
#include "qkrylov/linalg/vector_ops.hpp"
#include <Kokkos_Core.hpp>

using namespace qkrylov;

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        HostVector hx = {1.0, 2.0, 3.0};
        HostVector hy = {4.0, 5.0, 6.0};

        VectorView<Kokkos::DefaultExecutionSpace> x("x", 3);
        VectorView<Kokkos::DefaultExecutionSpace> y("y", 3);

        copy_host_to_device(hx, x);
        copy_host_to_device(hy, y);

        std::cout << "dot = " << dot(x,y) << "\n";
        std::cout << "norm(x) = " << norm(x) << "\n";
    }
    Kokkos::finalize();
    return 0;
}
