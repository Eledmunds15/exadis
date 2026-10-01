set(TARGET_NAME test_C1_edge_dipole)
add_executable(${TARGET_NAME} test_C1_edge_dipole.cpp)
target_include_directories(${TARGET_NAME} PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE exadis ${EXADIS_EXTERN_LIBS})
