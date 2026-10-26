cmake_minimum_required(VERSION 3.24)
project(tile_infer VERSION 0.2.0 LANGUAGES CXX)

option(TILE_INFER_ENABLE_CUDA "Build the CUDA runtime and checks" ON)
option(TILE_INFER_ENABLE_UBSAN "Instrument the CPU package loader test" ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CTest)

function(tileinfer_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    else()
        # NVCC-generated GCC line directives trigger -Wpedantic noise; retain Wall/Wextra for CUDA host code.
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
    endif()
endfunction()

set(TILEINFER_PLAN_SOURCES src/runtime/plan.cpp src/runtime/plan_validation.cpp)
add_library(tileinfer_plan STATIC ${TILEINFER_PLAN_SOURCES})
target_include_directories(tileinfer_plan PUBLIC src/runtime PRIVATE third_party)
tileinfer_warnings(tileinfer_plan)

find_program(CLANG_FORMAT_EXECUTABLE clang-format)
if(BUILD_TESTING AND CLANG_FORMAT_EXECUTABLE)
    file(GLOB_RECURSE TILEINFER_FORMAT_SOURCES CONFIGURE_DEPENDS
        src/*.cpp src/*.cu src/*.hpp src/*.cuh tests/*.cpp tests/*.cu tests/*.hpp tests/*.cuh)
    add_test(NAME clang_format COMMAND ${CLANG_FORMAT_EXECUTABLE} --dry-run --Werror ${TILEINFER_FORMAT_SOURCES})
    set_tests_properties(clang_format PROPERTIES TIMEOUT 30)
endif()

find_program(UV_EXECUTABLE uv HINTS "$ENV{HOME}/.local/bin")
if(BUILD_TESTING)
    add_executable(cpp_smoke tests/cpp_smoke.cpp)
    add_executable(runtime_unit tests/runtime_unit.cpp)
    target_link_libraries(runtime_unit PRIVATE tileinfer_plan)
    add_executable(plan_loader_test tests/plan_loader_test.cpp ${TILEINFER_PLAN_SOURCES})
    target_include_directories(plan_loader_test PRIVATE src/runtime third_party)
    foreach(target cpp_smoke runtime_unit plan_loader_test)
        tileinfer_warnings(${target})
    endforeach()
    if(TILE_INFER_ENABLE_UBSAN AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(plan_loader_test PRIVATE -fsanitize=undefined -fno-sanitize-recover=undefined)
        target_link_options(plan_loader_test PRIVATE -fsanitize=undefined)
    endif()
    add_test(NAME cpp_smoke COMMAND cpp_smoke)
    add_test(NAME runtime_unit COMMAND runtime_unit)
    set_tests_properties(cpp_smoke runtime_unit PROPERTIES TIMEOUT 30)
    if(UV_EXECUTABLE)
        add_test(NAME loader_regression COMMAND ${UV_EXECUTABLE} run --locked
            --project ${PROJECT_SOURCE_DIR}/../tileplan python ${PROJECT_SOURCE_DIR}/tests/loader_regression.py
            --loader $<TARGET_FILE:plan_loader_test>)
        set_tests_properties(loader_regression PROPERTIES TIMEOUT 180)
    else()
        message(WARNING "uv was not found: CPU package, decoder, and stack regressions are unavailable. Install uv and reconfigure to run the full suite.")
    endif()
endif()

if(TILE_INFER_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    add_library(tileinfer_runtime STATIC src/runtime/worker_kernel.cu
        src/runtime/uploaded_plan.cu src/runtime/ticket.cu src/runtime/sequence_context.cu
        src/runtime/snapshot.cu src/runtime/ops.cuh)
    target_include_directories(tileinfer_runtime PUBLIC src/runtime PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
    target_link_libraries(tileinfer_runtime PUBLIC CUDA::cudart tileinfer_plan)
    add_executable(tile_runtime src/runtime/main.cpp)
    target_include_directories(tile_runtime PRIVATE third_party)
    target_link_libraries(tile_runtime PRIVATE tileinfer_runtime)
    tileinfer_warnings(tileinfer_runtime)
    tileinfer_warnings(tile_runtime)

    if(BUILD_TESTING)
        add_library(tileinfer_diagnostics STATIC src/runtime/diagnostic_context.cu)
        target_include_directories(tileinfer_diagnostics PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
        target_link_libraries(tileinfer_diagnostics PUBLIC tileinfer_runtime)
        add_executable(tile_runtime_test src/runtime/main.cpp)
        target_include_directories(tile_runtime_test PRIVATE third_party)
        target_link_libraries(tile_runtime_test PRIVATE tileinfer_diagnostics)
        target_compile_definitions(tile_runtime_test PRIVATE TILE_TEST_HARNESS=1)
        add_executable(ops_unit tests/ops_unit.cu)
        target_include_directories(ops_unit PRIVATE src/runtime third_party ${CUDAToolkit_INCLUDE_DIRS}/cccl)
        target_link_libraries(ops_unit PRIVATE CUDA::cudart)
        tileinfer_warnings(ops_unit)
        # Diagnostic bandwidth roofline and row-dot microbenchmarks; not a CTest.
        add_executable(bandwidth_probe tests/bandwidth_probe.cu)
        target_include_directories(bandwidth_probe PRIVATE src/runtime third_party ${CUDAToolkit_INCLUDE_DIRS}/cccl)
        target_link_libraries(bandwidth_probe PRIVATE CUDA::cudart)
        tileinfer_warnings(bandwidth_probe)
        # Attention wave microbenchmark and ablations; not a CTest.
        add_executable(attention_probe tests/attention_probe.cu)
        target_include_directories(attention_probe PRIVATE src/runtime third_party ${CUDAToolkit_INCLUDE_DIRS}/cccl)
        target_link_libraries(attention_probe PRIVATE CUDA::cudart)
        tileinfer_warnings(attention_probe)
        # Grid barriers against counter waits around attention on a real package; not a CTest.
        add_executable(counter_probe tests/counter_probe.cu)
        target_include_directories(counter_probe PRIVATE third_party /cccl)
        target_link_libraries(counter_probe PRIVATE tileinfer_runtime)
        tileinfer_warnings(counter_probe)
        add_executable(sequence_context_smoke tests/sequence_context_smoke.cpp)
        target_link_libraries(sequence_context_smoke PRIVATE tileinfer_runtime)
        add_executable(sequence_ring_smoke tests/sequence_ring_smoke.cu)
        target_include_directories(sequence_ring_smoke PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
        target_link_libraries(sequence_ring_smoke PRIVATE tileinfer_runtime)
        foreach(target cuda_smoke cccl_cooperative_smoke input_ring_copy_smoke)
            add_executable(${target} tests/${target}.cu)
            target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
            target_link_libraries(${target} PRIVATE CUDA::cudart)
        endforeach()
        target_include_directories(cccl_cooperative_smoke PRIVATE src/runtime)
        foreach(target tileinfer_diagnostics tile_runtime_test sequence_context_smoke sequence_ring_smoke
                cuda_smoke cccl_cooperative_smoke input_ring_copy_smoke)
            tileinfer_warnings(${target})
        endforeach()
        add_test(NAME tile_runtime_help COMMAND tile_runtime --help)
        foreach(target cuda_smoke cccl_cooperative_smoke input_ring_copy_smoke)
            add_test(NAME ${target} COMMAND ${target})
        endforeach()
        set_tests_properties(tile_runtime_help cuda_smoke cccl_cooperative_smoke input_ring_copy_smoke PROPERTIES TIMEOUT 30)
        set_tests_properties(cuda_smoke PROPERTIES SKIP_RETURN_CODE 77)
        if(UV_EXECUTABLE)
            add_test(NAME ops_unit COMMAND ${UV_EXECUTABLE} run --locked --project ${PROJECT_SOURCE_DIR}/../tileplan python ${PROJECT_SOURCE_DIR}/tests/ops_unit.py --executor $<TARGET_FILE:ops_unit>)
            set_tests_properties(ops_unit PROPERTIES TIMEOUT 180)
            add_test(NAME decoder_regression COMMAND ${UV_EXECUTABLE} run --locked --project ${PROJECT_SOURCE_DIR}/../tileplan
                python ${PROJECT_SOURCE_DIR}/tests/decoder_regression.py --executor $<TARGET_FILE:tile_runtime_test>)
            add_test(NAME stack_regression COMMAND ${UV_EXECUTABLE} run --locked --project ${PROJECT_SOURCE_DIR}/../tileplan
                python ${PROJECT_SOURCE_DIR}/tests/stack_regression.py --executor $<TARGET_FILE:tile_runtime>
                --sequence-smoke $<TARGET_FILE:sequence_context_smoke> --ring-smoke $<TARGET_FILE:sequence_ring_smoke>)
            add_test(NAME kernel_regression COMMAND ${UV_EXECUTABLE} run --locked --project ${PROJECT_SOURCE_DIR}/../tileplan
                python ${PROJECT_SOURCE_DIR}/tests/kernel_regression.py --executor $<TARGET_FILE:tile_runtime>
                --block-executor $<TARGET_FILE:tile_runtime_test>)
            set_tests_properties(kernel_regression PROPERTIES TIMEOUT 240)
            add_test(NAME cli_regression COMMAND ${UV_EXECUTABLE} run --locked --project ${PROJECT_SOURCE_DIR}/../tileplan
                python ${PROJECT_SOURCE_DIR}/tests/cli_regression.py --executor $<TARGET_FILE:tile_runtime>)
            set_tests_properties(decoder_regression stack_regression PROPERTIES TIMEOUT 240)
            set_tests_properties(cli_regression PROPERTIES TIMEOUT 30)
            find_program(COMPUTE_SANITIZER_EXECUTABLE compute-sanitizer HINTS ${CUDAToolkit_BIN_DIR})
            if(COMPUTE_SANITIZER_EXECUTABLE)
                add_custom_target(sanitizers COMMAND ${UV_EXECUTABLE} run --locked --project ${PROJECT_SOURCE_DIR}/../tileplan
                    python ${PROJECT_SOURCE_DIR}/tests/sanitizers.py --build-dir ${CMAKE_CURRENT_BINARY_DIR}
                    --sanitizer ${COMPUTE_SANITIZER_EXECUTABLE} USES_TERMINAL
                    DEPENDS tile_runtime tile_runtime_test sequence_context_smoke sequence_ring_smoke input_ring_copy_smoke ops_unit)
            else()
                message(WARNING "compute-sanitizer was not found: the sanitizers target is unavailable")
            endif()
        endif()
    endif()
endif()
