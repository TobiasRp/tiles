cmake_minimum_required(VERSION 3.24)

project(tile_infer VERSION 0.1.0 LANGUAGES CXX)

option(TILE_INFER_ENABLE_CUDA "Build the CUDA runtime and checks" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_executable(cpp_smoke tests/cpp_smoke.cpp)
add_executable(runtime_unit tests/runtime_unit.cpp src/runtime/plan.cpp)
target_include_directories(runtime_unit PRIVATE src/runtime third_party)
if(MSVC)
    target_compile_options(cpp_smoke PRIVATE /W4)
    target_compile_options(runtime_unit PRIVATE /W4)
else()
    target_compile_options(cpp_smoke PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(runtime_unit PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(BUILD_TESTING)
    add_test(NAME cpp_smoke COMMAND cpp_smoke)
    add_test(NAME runtime_unit COMMAND runtime_unit)
    set_tests_properties(cpp_smoke PROPERTIES TIMEOUT 30)
    set_tests_properties(runtime_unit PROPERTIES TIMEOUT 30)
endif()

if(TILE_INFER_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    add_executable(tile_runtime src/runtime/main.cpp src/runtime/plan.cpp src/runtime/worker.cu)
    target_include_directories(tile_runtime PRIVATE third_party ${CUDAToolkit_INCLUDE_DIRS}/cccl)
    target_link_libraries(tile_runtime PRIVATE CUDA::cudart)
    if(MSVC)
        target_compile_options(tile_runtime PRIVATE /W4)
    else()
        target_compile_options(tile_runtime PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
    endif()

    add_executable(cuda_smoke tests/cuda_smoke.cu)
    target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)

    add_executable(cccl_cooperative_smoke tests/cccl_cooperative_smoke.cu)
    target_include_directories(cccl_cooperative_smoke PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
    target_link_libraries(cccl_cooperative_smoke PRIVATE CUDA::cudart)

    if(BUILD_TESTING)
        add_test(NAME tile_runtime_help COMMAND tile_runtime --help)
        add_test(NAME cuda_smoke COMMAND cuda_smoke)
        add_test(NAME cccl_cooperative_smoke COMMAND cccl_cooperative_smoke)
        set_tests_properties(tile_runtime_help PROPERTIES TIMEOUT 30)
        set_tests_properties(cuda_smoke PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
        set_tests_properties(cccl_cooperative_smoke PROPERTIES TIMEOUT 30)

        find_program(UV_EXECUTABLE uv HINTS "$ENV{HOME}/.local/bin")
        if(UV_EXECUTABLE)
            add_test(NAME runtime_regression
                COMMAND ${UV_EXECUTABLE} run --project ${PROJECT_SOURCE_DIR}/../tileplan
                    python ${PROJECT_SOURCE_DIR}/tests/regression.py
                    --executor $<TARGET_FILE:tile_runtime>)
            set_tests_properties(runtime_regression PROPERTIES TIMEOUT 180)
        endif()
    endif()
endif()
