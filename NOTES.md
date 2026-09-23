# Ethan's Notes

A little markdown file for Ethan to keep track of notes

## Build Commands

Quick build (Good enough for when editing source code)
```
cd build
make -j4
```

Full reconfigure (if you change CMakeLists.txt and add new directories)
```
cd build
cmake .
make -j4
```

From scratch (build/ is deleted)
```
bash ./configure
cd build
make -j4
```

To build just one target instead of everything
```
cd build
make test_frank_read_src -j4
make exadis_stress_field -j4
```

## Examples

### C1_bcc_Fe_infinite_edge_dislocation

A simple test creating one inifinite edge dislocation in BCC Iron. This will form the basis tests for dislocation climb.

## Design Notes

- Creating a new phase field module to compile with exadis, so that people don't have to call the very expensive functionalities all the time.