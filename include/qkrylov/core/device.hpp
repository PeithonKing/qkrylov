#pragma once

#include "qkrylov/core/types.hpp"

#include <Kokkos_Core.hpp>

#include <string>
#include <cstdlib>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


/// Selects which device to target.
///
/// - On GPU builds: the `id` field selects which GPU (0, 1, 2, ...).
/// - On CPU builds: the `id` field is ignored.
///
/// Accepted string formats: "cpu", "cuda:0", "cuda:1", "hip:0",
/// "gpu:0", "gpu:1", "sycl:0", or just "gpu" (defaults to device 0).
struct Device {
    int id = 0;

    Device() = default;

    explicit Device(int device_id) : id(device_id) {}

    explicit Device(const std::string& s) {
        if (s == "cpu") {
            id = 0;
            return;
        }
        auto colon = s.find(':');
        if (colon == std::string::npos) {
            id = 0; // "cuda", "hip", "gpu" → device 0
        } else {
            id = std::stoi(s.substr(colon + 1));
        }
    }

    /// True if the compiled Kokkos backend targets a GPU.
    static bool is_gpu_build() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
        return true;
#else
        return false;
#endif
    }

    /// Human-readable name of the compiled Kokkos backend.
    static std::string backend_name() {
#if defined(KOKKOS_ENABLE_CUDA)
        return "cuda";
#elif defined(KOKKOS_ENABLE_HIP)
        return "hip";
#elif defined(KOKKOS_ENABLE_SYCL)
        return "sycl";
#elif defined(KOKKOS_ENABLE_OPENMP)
        return "openmp";
#else
        return "serial";
#endif
    }

    static int gpu_count() {
#if defined(KOKKOS_ENABLE_CUDA)
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
        return count;
#elif defined(KOKKOS_ENABLE_HIP)
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess) return 0;
        return count;
#else
        return 0;
#endif
    }
};

namespace detail {

/// Ensures Kokkos is initialized exactly once.
/// Safe to call multiple times — subsequent calls are no-ops.
/// Registers Kokkos::finalize() via std::atexit on first call.
inline void ensure_kokkos_initialized() {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        std::atexit([]() {
            if (Kokkos::is_initialized()) {
                Kokkos::finalize();
            }
        });
    }
}

/// Initialize Kokkos with a specific device ID.
/// Must be called before any Kokkos View allocation or parallel dispatch.
/// If Kokkos is already initialized, this is a no-op.
inline void initialize_kokkos(const Device& dev = Device()) {
    if (!Kokkos::is_initialized()) {
        Kokkos::InitializationSettings settings;
        settings.set_device_id(dev.id);
        Kokkos::initialize(settings);
        std::atexit([]() {
            if (Kokkos::is_initialized()) {
                Kokkos::finalize();
            }
        });
    }
}

} // namespace detail


} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
