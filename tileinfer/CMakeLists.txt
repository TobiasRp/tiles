cmake_minimum_required(VERSION 3.24)

project(tile_infer VERSION 0.2.0 LANGUAGES CXX)

option(TILE_INFER_ENABLE_CUDA "Build the CUDA runtime and checks" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

find_program(CLANG_FORMAT_EXECUTABLE clang-format)
if(BUILD_TESTING AND CLANG_FORMAT_EXECUTABLE)
    add_test(NAME clang_format
        COMMAND ${CLANG_FORMAT_EXECUTABLE} --dry-run --Werror
            ${PROJECT_SOURCE_DIR}/src/runtime/main.cpp
            ${PROJECT_SOURCE_DIR}/src/runtime/plan.cpp
            ${PROJECT_SOURCE_DIR}/src/runtime/plan.hpp
            ${PROJECT_SOURCE_DIR}/src/runtime/runtime.hpp
            ${PROJECT_SOURCE_DIR}/src/runtime/diagnostic.hpp
            ${PROJECT_SOURCE_DIR}/src/runtime/ops.cuh
            ${PROJECT_SOURCE_DIR}/src/runtime/detail/cuda_check.hpp
            ${PROJECT_SOURCE_DIR}/src/runtime/detail/input_ring_cursor.hpp
            ${PROJECT_SOURCE_DIR}/src/runtime/detail/launch.cuh
            ${PROJECT_SOURCE_DIR}/src/runtime/detail/uploaded_plan.cuh
            ${PROJECT_SOURCE_DIR}/src/runtime/detail/ticket_state.cuh
            ${PROJECT_SOURCE_DIR}/src/runtime/worker_kernel.cu
            ${PROJECT_SOURCE_DIR}/src/runtime/sampling.cu
            ${PROJECT_SOURCE_DIR}/src/runtime/uploaded_plan.cu
            ${PROJECT_SOURCE_DIR}/src/runtime/ticket.cu
            ${PROJECT_SOURCE_DIR}/src/runtime/sequence_context.cu
            ${PROJECT_SOURCE_DIR}/src/runtime/diagnostic_context.cu
            ${PROJECT_SOURCE_DIR}/tests/cccl_cooperative_smoke.cu
            ${PROJECT_SOURCE_DIR}/tests/input_ring_copy_smoke.cu
            ${PROJECT_SOURCE_DIR}/tests/runtime_unit.cpp
            ${PROJECT_SOURCE_DIR}/tests/sequence_context_smoke.cpp)
    set_tests_properties(clang_format PROPERTIES TIMEOUT 30)
endif()

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

    add_library(tileinfer_runtime STATIC src/runtime/plan.cpp
        src/runtime/worker_kernel.cu src/runtime/sampling.cu
        src/runtime/uploaded_plan.cu src/runtime/ticket.cu
        src/runtime/sequence_context.cu src/runtime/ops.cuh)
    target_include_directories(tileinfer_runtime PUBLIC src/runtime)
    target_include_directories(tileinfer_runtime PRIVATE third_party ${CUDAToolkit_INCLUDE_DIRS}/cccl)
    target_link_libraries(tileinfer_runtime PUBLIC CUDA::cudart)

    add_library(tileinfer_diagnostics STATIC src/runtime/diagnostic_context.cu)
    target_include_directories(tileinfer_diagnostics PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
    target_link_libraries(tileinfer_diagnostics PUBLIC tileinfer_runtime)

    add_executable(tile_runtime src/runtime/main.cpp)
    target_include_directories(tile_runtime PRIVATE third_party)
    target_link_libraries(tile_runtime PRIVATE tileinfer_runtime)
    add_executable(tile_runtime_test src/runtime/main.cpp)
    target_include_directories(tile_runtime_test PRIVATE third_party)
    target_link_libraries(tile_runtime_test PRIVATE tileinfer_diagnostics)
    target_compile_definitions(tile_runtime_test PRIVATE TILE_TEST_HARNESS=1)
    add_executable(sequence_context_smoke tests/sequence_context_smoke.cpp)
    target_link_libraries(sequence_context_smoke PRIVATE tileinfer_runtime)
    if(MSVC)
        target_compile_options(tileinfer_runtime PRIVATE /W4)
        target_compile_options(tileinfer_diagnostics PRIVATE /W4)
        target_compile_options(tile_runtime PRIVATE /W4)
        target_compile_options(tile_runtime_test PRIVATE /W4)
        target_compile_options(sequence_context_smoke PRIVATE /W4)
    else()
        target_compile_options(tileinfer_runtime PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
        target_compile_options(tileinfer_diagnostics PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
        target_compile_options(tile_runtime PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
        target_compile_options(tile_runtime_test PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
        target_compile_options(sequence_context_smoke PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
    endif()

    add_executable(cuda_smoke tests/cuda_smoke.cu)
    target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)

    add_executable(cccl_cooperative_smoke tests/cccl_cooperative_smoke.cu)
    target_include_directories(cccl_cooperative_smoke PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
    target_link_libraries(cccl_cooperative_smoke PRIVATE CUDA::cudart)

    add_executable(input_ring_copy_smoke tests/input_ring_copy_smoke.cu)
    target_include_directories(input_ring_copy_smoke PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
    target_link_libraries(input_ring_copy_smoke PRIVATE CUDA::cudart)

    if(BUILD_TESTING)
        add_test(NAME tile_runtime_help COMMAND tile_runtime --help)
        add_test(NAME cuda_smoke COMMAND cuda_smoke)
        add_test(NAME cccl_cooperative_smoke COMMAND cccl_cooperative_smoke)
        add_test(NAME input_ring_copy_smoke COMMAND input_ring_copy_smoke)
        set_tests_properties(tile_runtime_help PROPERTIES TIMEOUT 30)
        set_tests_properties(cuda_smoke PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
        set_tests_properties(cccl_cooperative_smoke PROPERTIES TIMEOUT 30)

        find_program(UV_EXECUTABLE uv HINTS "$ENV{HOME}/.local/bin")
        if(UV_EXECUTABLE)
            add_test(NAME decoder_regression
                COMMAND ${UV_EXECUTABLE} run --locked --project ${PROJECT_SOURCE_DIR}/../tileplan
                    python ${PROJECT_SOURCE_DIR}/tests/decoder_regression.py
                    --executor $<TARGET_FILE:tile_runtime_test>)
            set_tests_properties(decoder_regression PROPERTIES TIMEOUT 180)
            add_test(NAME stack_regression
                COMMAND ${UV_EXECUTABLE} run --locked --project ${PROJECT_SOURCE_DIR}/../tileplan
                    python ${PROJECT_SOURCE_DIR}/tests/stack_regression.py
                    --executor $<TARGET_FILE:tile_runtime>
                    --sequence-smoke $<TARGET_FILE:sequence_context_smoke>)
            set_tests_properties(stack_regression PROPERTIES TIMEOUT 180)
        endif()
    endif()
endif()
