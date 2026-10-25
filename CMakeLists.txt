cmake_minimum_required(VERSION 3.20)

# Version: the Python build (scikit-build-core) passes it in; a plain CMake build derives MAJOR.MINOR.PATCH
# from the newest "vX.Y.Z" git tag; -DQKRYLOV_VERSION=X.Y.Z overrides both. "0.0.0" only if nothing is known.
if(QKRYLOV_VERSION)
    set(QKRYLOV_PROJECT_VERSION "${QKRYLOV_VERSION}")
elseif(DEFINED SKBUILD_PROJECT_VERSION)
    set(QKRYLOV_PROJECT_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
    set(QKRYLOV_PROJECT_VERSION "0.0.0")
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0 --match "v[0-9]*"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE QKRYLOV_GIT_TAG OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE QKRYLOV_GIT_RC)
        if(QKRYLOV_GIT_RC EQUAL 0 AND QKRYLOV_GIT_TAG MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+)")
            set(QKRYLOV_PROJECT_VERSION "${CMAKE_MATCH_1}")
        endif()
    endif()
endif()

project(
    qkrylov
    VERSION ${QKRYLOV_PROJECT_VERSION}
    DESCRIPTION "Matrix-free Krylov methods for quantum many-body physics"
    LANGUAGES CXX
)

option(QKRYLOV_BUILD_TESTS "Build qkrylov test executables" ON)
option(QKRYLOV_BUILD_EXAMPLES "Build qkrylov examples" OFF)
option(QKRYLOV_BUILD_PYTHON "Build Python bindings" OFF)

set(QKRYLOV_COMMON_SOURCES
    src/basis/spinhalf_basis.cpp
    src/basis/spin_s_basis.cpp
    src/basis/fermion_basis.cpp
    src/basis/hubbard_basis.cpp
    src/basis/tj_basis.cpp
    # Precision-independent, so compiled once: building them into both the fp32 and fp64
    # object sets defines every symbol twice, which breaks a shared libqkrylov (Julia needs it).
    src/operators/opsum.cpp
    src/sites/spinhalf_site.cpp
    src/sites/spin_s_site.cpp
    src/sites/fermion_site.cpp
    src/sites/hubbard_site.cpp
    src/sites/tj_site.cpp
)

set(QKRYLOV_SOURCES
    src/hamiltonian/matrix_free_hamiltonian.cpp
    src/solvers/lanczos.cpp
    src/solvers/davidson.cpp
    src/solvers/dynamics.cpp
    src/solvers/ftlm.cpp
    src/solvers/correction_vector.cpp
)

add_library(qkrylov_common OBJECT ${QKRYLOV_COMMON_SOURCES})
target_compile_features(qkrylov_common PUBLIC cxx_std_20)
set_target_properties(qkrylov_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(qkrylov_fp32 OBJECT ${QKRYLOV_SOURCES})
target_compile_definitions(qkrylov_fp32 PUBLIC QKRYLOV_SINGLE_PRECISION)
target_compile_features(qkrylov_fp32 PUBLIC cxx_std_20)
set_target_properties(qkrylov_fp32 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(qkrylov_fp64 OBJECT ${QKRYLOV_SOURCES})
target_compile_definitions(qkrylov_fp64 PUBLIC QKRYLOV_DOUBLE_PRECISION)
target_compile_features(qkrylov_fp64 PUBLIC cxx_std_20)
set_target_properties(qkrylov_fp64 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(qkrylov_c_api_fp32 OBJECT src/c_api_fp32.cpp)
target_compile_definitions(qkrylov_c_api_fp32 PUBLIC QKRYLOV_SINGLE_PRECISION)
target_compile_features(qkrylov_c_api_fp32 PUBLIC cxx_std_20)
set_target_properties(qkrylov_c_api_fp32 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(qkrylov_c_api_fp64 OBJECT src/c_api_fp64.cpp)
target_compile_definitions(qkrylov_c_api_fp64 PUBLIC QKRYLOV_DOUBLE_PRECISION)
target_compile_features(qkrylov_c_api_fp64 PUBLIC cxx_std_20)
set_target_properties(qkrylov_c_api_fp64 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(qkrylov
    $<TARGET_OBJECTS:qkrylov_common>
    $<TARGET_OBJECTS:qkrylov_fp32>
    $<TARGET_OBJECTS:qkrylov_fp64>
    $<TARGET_OBJECTS:qkrylov_c_api_fp32>
    $<TARGET_OBJECTS:qkrylov_c_api_fp64>
    src/c_api.cpp
)
add_library(qkrylov::qkrylov ALIAS qkrylov)
target_compile_definitions(qkrylov PRIVATE QKRYLOV_DOUBLE_PRECISION)
target_compile_features(qkrylov PUBLIC cxx_std_20)
set_target_properties(qkrylov PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_definitions(qkrylov_common PUBLIC _USE_MATH_DEFINES)
    target_compile_definitions(qkrylov_fp32 PUBLIC _USE_MATH_DEFINES)
    target_compile_definitions(qkrylov_fp64 PUBLIC _USE_MATH_DEFINES)
    target_compile_definitions(qkrylov_c_api_fp32 PUBLIC _USE_MATH_DEFINES)
    target_compile_definitions(qkrylov_c_api_fp64 PUBLIC _USE_MATH_DEFINES)
    target_compile_definitions(qkrylov PUBLIC _USE_MATH_DEFINES)
endif()

# Kokkos: prefer one the user already has (HPC sites must not get a second, mismatched copy). The Python
# wheel build (scikit-build) fetches its own pinned version so wheels are reproducible. Set
# -DQKRYLOV_USE_SYSTEM_KOKKOS=OFF to force fetching, or point CMAKE_PREFIX_PATH / Kokkos_ROOT at your install.
if(DEFINED SKBUILD_PROJECT_VERSION)
    set(_qk_system_kokkos_default OFF)
else()
    set(_qk_system_kokkos_default ON)
endif()
option(QKRYLOV_USE_SYSTEM_KOKKOS "Use an installed Kokkos (find_package) before fetching one" ${_qk_system_kokkos_default})
if(QKRYLOV_USE_SYSTEM_KOKKOS)
    # Only the 4.x series is supported (4.3 up to, not including, 5.0): a different major version changes the C++
    # standard and the device-launch internals the code relies on, so a 5.x install is skipped, not used by accident.
    # The version range is checked before any imported target is created, so the fetch below stays possible.
    find_package(Kokkos 4.3...<5.0 QUIET CONFIG)
    if(Kokkos_FOUND)
        message(STATUS "Using installed Kokkos ${Kokkos_VERSION} (${Kokkos_DIR})")
    endif()
endif()
if(NOT Kokkos_FOUND)
    message(STATUS "Fetching Kokkos 4.7.04 from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        4.7.04
    )
    # Force Position Independent Code globally so Kokkos thread_local variables use global-dynamic TLS
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(Kokkos_ENABLE_PIC ON CACHE BOOL "Enable Position Independent Code for Kokkos" FORCE)
    
    # If no backend was explicitly passed by the user, default to OpenMP for CPU fallback
    if(NOT DEFINED Kokkos_ENABLE_CUDA AND NOT DEFINED Kokkos_ENABLE_HIP AND NOT DEFINED Kokkos_ENABLE_SYCL)
        if(MSVC)
            set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Disable Kokkos OpenMP on MSVC" FORCE)
        else()
            set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP backend" FORCE)
        endif()
        set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Kokkos Serial backend" FORCE)
    endif()
    # Suppress Kokkos warnings during FetchContent
    set(Kokkos_ENABLE_DEPRECATION_WARNINGS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(kokkos)
endif()

target_link_libraries(qkrylov_common PUBLIC Kokkos::kokkos)
target_link_libraries(qkrylov_fp32 PUBLIC Kokkos::kokkos)
target_link_libraries(qkrylov_fp64 PUBLIC Kokkos::kokkos)
target_link_libraries(qkrylov_c_api_fp32 PUBLIC Kokkos::kokkos)
target_link_libraries(qkrylov_c_api_fp64 PUBLIC Kokkos::kokkos)
target_link_libraries(qkrylov PUBLIC Kokkos::kokkos)

target_include_directories(qkrylov_common PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(qkrylov_fp32 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(qkrylov_fp64 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(qkrylov_c_api_fp32 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(qkrylov_c_api_fp64 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(qkrylov PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(QKRYLOV_HAS_OPENMP ON)
else()
    set(QKRYLOV_HAS_OPENMP OFF)
endif()

# Generated version header (QKRYLOV_VERSION_*), installed next to the other headers.
configure_file(include/qkrylov/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/qkrylov/version.h @ONLY)
foreach(t qkrylov_common qkrylov_fp32 qkrylov_fp64 qkrylov_c_api_fp32 qkrylov_c_api_fp64 qkrylov)
    target_include_directories(${t} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)
endforeach()

# Shared library: real SONAME + version symlinks (libqkrylov.so.0.1 -> libqkrylov.so.0.1.5). While the major
# version is 0 the interface may change between minors, so the SONAME carries MAJOR.MINOR.
set_target_properties(qkrylov PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
)

# Installed library finds its own dependencies (Kokkos etc.) next to itself or where they were found at build time.
# Needed because a consumer's RUNPATH is not searched for the library's own dependencies.
if(APPLE)
    set_target_properties(qkrylov PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_target_properties(qkrylov PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
set_target_properties(qkrylov PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

# Windows DLLs: export the C API (QKRYLOV_API in c_api.h) when building the shared library, import it when using it.
if(WIN32 AND BUILD_SHARED_LIBS)
    foreach(t qkrylov_c_api_fp32 qkrylov_c_api_fp64 qkrylov)
        target_compile_definitions(${t} PRIVATE QKRYLOV_SHARED_BUILD)
    endforeach()
    target_compile_definitions(qkrylov INTERFACE QKRYLOV_SHARED_USE)
endif()

configure_package_config_file(
    cmake/qkrylovConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/qkrylovConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qkrylov
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/qkrylovConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

if(BUILD_SHARED_LIBS)
    # Relocatable pkg-config file: prefix is found relative to the .pc file, so --prefix at install time works.
    file(RELATIVE_PATH QKRYLOV_PC_REL_PREFIX "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
    configure_file(cmake/qkrylov.pc.in ${CMAKE_CURRENT_BINARY_DIR}/qkrylov.pc @ONLY)
endif()

if(NOT DEFINED SKBUILD_PROJECT_NAME)
    install(
        TARGETS qkrylov
        EXPORT qkrylovTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    if(BUILD_SHARED_LIBS)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/qkrylov.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
    endif()

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/qkrylov/version.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qkrylov)

    install(
        EXPORT qkrylovTargets
        NAMESPACE qkrylov::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qkrylov
    )

    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/qkrylovConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/qkrylovConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qkrylov
    )
endif()

if(QKRYLOV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(QKRYLOV_BUILD_PYTHON)
    find_package(Python 3.11 REQUIRED
        COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule
    )
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(
        _qkrylov_cpp
        STABLE_ABI
        NB_DOMAIN qkrylov
        bindings/python/src/binding_main.cpp bindings/python/src/binding_fp32.cpp bindings/python/src/binding_fp64.cpp
    )

    target_link_libraries(_qkrylov_cpp PRIVATE qkrylov::qkrylov)
    if(Kokkos_ENABLE_SYCL)
        target_link_options(_qkrylov_cpp PRIVATE -fsycl)
        target_compile_options(_qkrylov_cpp PRIVATE -fsycl)
    endif()

    install(
        TARGETS _qkrylov_cpp
        LIBRARY DESTINATION qkrylov
        RUNTIME DESTINATION qkrylov
    )

    install(
        DIRECTORY bindings/python/qkrylov/
        DESTINATION qkrylov
        FILES_MATCHING
            PATTERN "*.py"
            PATTERN "py.typed"
    )
endif()


# Link OpenMP if it was found above (the package config then asks consumers for it too)
if(OpenMP_CXX_FOUND)
    target_link_libraries(qkrylov_common PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(qkrylov_fp32 PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(qkrylov_fp64 PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(qkrylov PUBLIC OpenMP::OpenMP_CXX)
endif()

# Tune for the build machine (e.g. NEON/AVX2 variants). OFF by default: binaries built
# with it only run on CPUs like the build host, so release wheels must leave it off.
option(QKRYLOV_NATIVE_ARCH "Compile with -march=native (local benchmarking only)" OFF)
if(QKRYLOV_NATIVE_ARCH AND NOT MSVC)
    foreach(t qkrylov_common qkrylov_fp32 qkrylov_fp64 qkrylov_c_api_fp32 qkrylov_c_api_fp64 qkrylov)
        target_compile_options(${t} PUBLIC -march=native)
    endforeach()
endif()

option(QKRYLOV_BUILD_BENCH "Build the custom_bench C++ benchmark executables" OFF)
if(QKRYLOV_BUILD_BENCH)
    foreach(prec fp64 fp32)
        add_executable(bench_${prec} custom_bench/bench.cpp)
        target_link_libraries(bench_${prec} PUBLIC qkrylov_${prec} qkrylov_common)
    endforeach()
    target_compile_definitions(bench_fp64 PUBLIC QKRYLOV_DOUBLE_PRECISION)
    target_compile_definitions(bench_fp32 PUBLIC QKRYLOV_SINGLE_PRECISION)
    # C++ side of benchmarks/xlang (same problem as bench.py / bench.jl)
    foreach(prec fp64 fp32)
        add_executable(xlang_${prec} benchmarks/xlang/bench.cpp)
        target_link_libraries(xlang_${prec} PUBLIC qkrylov_${prec} qkrylov_common)
    endforeach()
    target_compile_definitions(xlang_fp64 PUBLIC QKRYLOV_DOUBLE_PRECISION)
    target_compile_definitions(xlang_fp32 PUBLIC QKRYLOV_SINGLE_PRECISION)
endif()

if(QKRYLOV_BUILD_EXAMPLES)
    add_executable(quickstart examples/quickstart.cpp)
    target_link_libraries(quickstart PUBLIC qkrylov_fp64 qkrylov_common)

    # The same example in plain C, through the C API (needs a shared or static libqkrylov).
    enable_language(C)
    add_executable(quickstart_c examples/quickstart.c)
    target_link_libraries(quickstart_c PRIVATE qkrylov::qkrylov)
    if(UNIX)
        target_link_libraries(quickstart_c PRIVATE m)
    endif()
endif()
