#pragma once

#include "c_api_internal.hpp"
#include "qkrylov/core/device.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <type_traits>

/// \file c_api_space.hpp
/// \brief Per-Hamiltonian execution space for the C API.
///
/// A Hamiltonian handle remembers whether it runs on the CPU ("host") or on the build's default
/// device. In a CPU-only build both are the same Kokkos space, so nothing is duplicated.

enum { QK_SPACE_DEFAULT = 0, QK_SPACE_HOST = 1 };

inline constexpr bool qk_host_is_default =
    std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;

/// Run `f(std::type_identity<Space>{})` with Space = the execution space of the handle.
/// `f` must return int (a C API status code).
template <typename F>
inline int qk_dispatch(int space, F&& f) {
    if constexpr (qk_host_is_default) {
        (void)space;
        return f(std::type_identity<Kokkos::DefaultExecutionSpace>{});
    } else {
        if (space == QK_SPACE_HOST) return f(std::type_identity<Kokkos::DefaultHostExecutionSpace>{});
        return f(std::type_identity<Kokkos::DefaultExecutionSpace>{});
    }
}

/// True if the handle lives on the build's default device (its memory is what DevVector uses).
inline bool qk_on_default_space(const qkrylov_hamiltonian_t* h) {
    return qk_host_is_default || h->space != QK_SPACE_HOST;
}

struct QkDeviceChoice {
    bool ok = true;
    int space = QK_SPACE_DEFAULT;
    std::string device;   // string handed to qkrylov::Device (may carry a ":id" suffix)
    std::string error;
};

/// Interpret a device string: "cpu" (also host/openmp/serial/threads), "gpu", "cuda[:id]", "hip[:id]",
/// "sycl[:id]", or ""/"auto"/"default" (= whatever the build's default device is).
inline QkDeviceChoice qk_parse_device(const char* device_str) {
    QkDeviceChoice r;
    std::string s = device_str ? device_str : "auto";
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    const std::string base = s.substr(0, s.find(':'));
    r.device = s;
    if (base.empty() || base == "auto" || base == "default") {
        r.space = QK_SPACE_DEFAULT;
        r.device = "cpu";
        if (!qk_host_is_default) r.device = "gpu";
    } else if (base == "cpu" || base == "host" || base == "openmp" || base == "serial" || base == "threads") {
        r.space = QK_SPACE_HOST;
        r.device = "cpu";
    } else if (base == "gpu" || base == "cuda" || base == "hip" || base == "sycl") {
        if (!qkrylov::Device::is_gpu_build()) {
            r.ok = false;
            r.error = "this libqkrylov was not built with GPU support; use device=\"cpu\" or install a GPU build";
            return r;
        }
        std::string backend = qkrylov::Device::backend_name();
        std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char c) { return std::tolower(c); });
        if (base != "gpu" && backend.find(base) == std::string::npos) {
            r.ok = false;
            r.error = "this libqkrylov was built for the '" + backend + "' backend; the '" + base +
                      "' device is not available (use \"gpu\" or \"" + backend + "\")";
            return r;
        }
        r.space = QK_SPACE_DEFAULT;
    } else {
        r.ok = false;
        r.error = "unknown device '" + s + "' (expected cpu, gpu, cuda, hip, sycl or auto)";
    }
    return r;
}
