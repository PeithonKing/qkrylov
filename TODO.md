# qkrylov Roadmap & TODO List

## 1. Python Binding Safety Hardening
- [x] Add array dimension validation in `bindings/python/src/binding.cpp` for `MatrixFreeHamiltonian::apply` (throw clean Python exception if array size != `H.dimension()` instead of segfaulting).
- [x] Add tuple length validation in `OpSum.__iadd__` in `binding.cpp` to guard against malformed tuple indexing.
- [x] Wrap raw pointer allocation in `vec_to_numpy` with `std::unique_ptr` prior to `nb::capsule` creation to prevent memory leaks on exception.

## 2. Documentation System (Zensical)
- [x] Set up `zensical.toml` configuration and install `zensical` in `.venv`.
- [ ] Populate `docs/` with quickstart guide, API references, and zero-copy performance tips.
- [x] Configure `.github/workflows/docs.yml` for automated deployment to GitHub Pages using Zensical.

## 3. Complete GitHub Actions & `.github` Setup
- [x] Finalize `.github/workflows/pypi_publish.yaml` for PyPI wheel release builds (Linux x86_64, Linux ARM64, Windows, Mac Intel, Mac Apple Silicon).
- [x] Configure PyPI index publishing & GitHub Pages redirect index generation (`--extra-index-url` PyTorch-style setup).
- [x] Add `.github/workflows/tests.yaml` for fast CI test suite execution on push/PR.
- [x] Add Issue Templates and Pull Request Templates under `.github/`.

## 4. Universal `extern "C"` API Layer for Julia and even more languages
- [x] Create `include/qkrylov/c_api.h` exposing flat `extern "C"` functions for Basis, Site, OpSum, and MatrixFreeHamiltonian.
- [x] Implement `src/c_api.cpp` to bridge C calls to internal C++ classes without copying array buffers.
- [x] Support seamless zero-copy interoperability for Julia (`ccall`), Rust (`bindgen`), C, and Go.
