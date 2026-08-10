# =====================================================================
# QKrylov Builder Container (The Compiler Factory)
# =====================================================================
# This container is designed purely to compile a native Python wheel for
# your specific hardware without polluting your host OS with C++ toolchains.
#
# Usage:
#   docker run --rm -v $(pwd):/output -e BACKEND=cuda peithonking/qkrylov-builder
#
# The container will compile the wheel and drop it in your current directory.
# You can then install it on your host via: pip install ./qkrylov-*.whl

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV BACKEND=openmp

# Install essential build tools. 
# (Users can swap this base image for nvidia/cuda or rocm/dev-ubuntu for GPUs)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies
RUN pip3 install --no-cache-dir build scikit-build-core nanobind

WORKDIR /src
COPY . /src

# The entrypoint compiles the wheel based on the BACKEND env var
# and copies it to the /output volume mount.
ENTRYPOINT ["/bin/bash", "-c", "\
    echo \"Building qkrylov for backend: $BACKEND\" && \
    if [ \"$BACKEND\" = \"cuda\" ]; then \
        export CMAKE_ARGS=\"-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=ON\"; \
    elif [ \"$BACKEND\" = \"hip\" ]; then \
        export CMAKE_ARGS=\"-DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_OPENMP=ON\"; \
    elif [ \"$BACKEND\" = \"sycl\" ]; then \
        export CMAKE_ARGS=\"-DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_OPENMP=ON\"; \
    else \
        export CMAKE_ARGS=\"-DKokkos_ENABLE_OPENMP=ON\"; \
    fi && \
    python3 -m build --wheel && \
    cp dist/*.whl /output/ && \
    echo \"Success! Wheel copied to your host directory.\" \
"]
