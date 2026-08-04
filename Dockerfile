# =====================================================================
# Artifact 12: Docker Image (Jupyter Lab + GPU Support)
# =====================================================================
# For now, this builds the CPU version. 
# Once Kokkos is enabled, you can change the base image to:
# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 (for NVIDIA)
# FROM rocm/dev-ubuntu-22.04:5.6 (for AMD)
# FROM intel/oneapi-basekit:latest (for Intel SYCL)

FROM ubuntu:22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy the repository into the container
COPY . /workspace/qkrylov

# Install Python dependencies and Jupyter Lab
RUN pip3 install --no-cache-dir jupyterlab numpy scipy pytest

# Build and install qkrylov
WORKDIR /workspace/qkrylov
RUN pip3 install --no-cache-dir .

# Create a notebook directory for the user
WORKDIR /workspace/notebooks

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
