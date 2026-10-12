set(TARGET_NAME test_C2_single_dislocation_loop)
add_executable(${TARGET_NAME} test_C2_single_dislocation_loop.cpp)
target_include_directories(${TARGET_NAME} PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE exadis ${EXADIS_EXTERN_LIBS})
