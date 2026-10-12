cmake_minimum_required(VERSION 3.24)

project(tile_infer VERSION 0.1.0 LANGUAGES CXX)

option(TILE_INFER_ENABLE_CUDA "Build the CUDA runtime and checks" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_executable(cpp_smoke tests/cpp_smoke.cpp)
if(MSVC)
    target_compile_options(cpp_smoke PRIVATE /W4)
else()
    target_compile_options(cpp_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(BUILD_TESTING)
    add_test(NAME cpp_smoke COMMAND cpp_smoke)
    set_tests_properties(cpp_smoke PROPERTIES TIMEOUT 30)
endif()

if(TILE_INFER_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # The executable is deliberately a CLI scaffold until the CCCL gate passes.
    add_executable(tile_runtime src/runtime/main.cpp)
    target_link_libraries(tile_runtime PRIVATE CUDA::cudart)
    if(MSVC)
        target_compile_options(tile_runtime PRIVATE /W4)
    else()
        target_compile_options(tile_runtime PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(cuda_smoke tests/cuda_smoke.cu)
    target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)

    if(BUILD_TESTING)
        add_test(NAME tile_runtime_help COMMAND tile_runtime --help)
        add_test(NAME cuda_smoke COMMAND cuda_smoke)
        set_tests_properties(tile_runtime_help PROPERTIES TIMEOUT 30)
        set_tests_properties(cuda_smoke PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
    endif()
endif()
