# Contributing

Contributions are very welcome! qkrylov is an open-source project and we appreciate help in all forms.

## Ways to Contribute

- **Bug reports:** Open an issue on GitHub with a minimal reproducible example.
- **Feature requests:** Open a GitHub issue describing the feature and its use case.
- **Pull requests:** For code contributions, please discuss the change in an issue first.
- **Documentation:** Help write tutorials, fix typos, or add examples.
- **Benchmarks:** Add performance comparisons against other ED codes.

## Development Setup

```bash
git clone https://github.com/sjp95/qkrylov
cd qkrylov
python -m venv .venv
source .venv/bin/activate
make build
pip install bindings/python/
```

## Code Style

- C++: formatted with `clang-format` (see `.clang-format` at repo root).
- Python: PEP 8 compliant.
- Tests: add C++ tests in `tests/` and Python tests in `bindings/python/tests/`.

## Running Tests

```bash
# C++ tests
make build && cd build && ctest --output-on-failure

# Python tests
pytest bindings/python/tests/
```
