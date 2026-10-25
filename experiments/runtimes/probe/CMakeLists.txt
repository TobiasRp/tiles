cmake_minimum_required(VERSION 3.20)
project(rt_probe CXX)

# Build against an already built llama.cpp tree (see scripts/build_llamacpp.sh).
set(LLAMACPP_SRC "" CACHE PATH "llama.cpp source checkout")
set(LLAMACPP_BUILD "" CACHE PATH "llama.cpp build directory with bin/ libraries")
if(NOT LLAMACPP_SRC OR NOT LLAMACPP_BUILD)
  message(FATAL_ERROR "set LLAMACPP_SRC and LLAMACPP_BUILD")
endif()

find_library(LLAMA_LIB llama PATHS ${LLAMACPP_BUILD}/bin NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB ggml PATHS ${LLAMACPP_BUILD}/bin NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB ggml-base PATHS ${LLAMACPP_BUILD}/bin NO_DEFAULT_PATH REQUIRED)

add_executable(rt_probe rt_probe.cpp)
target_compile_features(rt_probe PRIVATE cxx_std_20)
target_include_directories(rt_probe PRIVATE ${LLAMACPP_SRC}/include ${LLAMACPP_SRC}/ggml/include)
target_link_libraries(rt_probe PRIVATE ${LLAMA_LIB} ${GGML_LIB} ${GGML_BASE_LIB})
set_target_properties(rt_probe PROPERTIES
  BUILD_RPATH ${LLAMACPP_BUILD}/bin
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
