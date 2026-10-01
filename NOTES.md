# Ethan's ExaDiS Notes

## Build

### 1. Configure (once, or after changing options)
Turns examples and tests on so they are *available* — nothing is compiled yet.
Note: `configure.sh` wipes `build/` and starts fresh.

CPU
```
./configure.sh -DEXADIS_BUILD_EXAMPLES=On -DEXADIS_BUILD_TESTS=On
```

GPU
```
./configure.sh -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_ARCH_ADA89=On -DEXADIS_BUILD_EXAMPLES=On -DEXADIS_BUILD_TESTS=On
```

### 2. Build
While developing — only what you're working on:
```
cmake --build build -j$(nproc) --target test_frank_read_src
cmake --build build -j$(nproc) --target test_force test_neighbor test_exadis
```

Before committing — everything:
```
cmake --build build -j$(nproc)
```

List all targets: `cmake --build build --target help`
List all options: `cmake -LH build` (also README "Build options")
