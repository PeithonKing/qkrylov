.PHONY: help build test clean

help:
	@echo "qkrylov development commands:"
	@echo "  build        Build the C++ library and tests"
	@echo "  test         Run C++ tests"
	@echo "  clean        Remove build artifacts"

build:
	cmake -B build -S .
	cmake --build build

test: build
	cd build && ctest --output-on-failure

clean:
	rm -rf build
