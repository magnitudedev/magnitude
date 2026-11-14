# Tools against upstream llama.cpp at the converter's pinned release (LLAMA_SOURCE, a checkout of the commit
# pyproject.toml pins), CPU only (`magnitude_conversion/llama.py`): verify-publish reads metadata, tokenizes and
# renders, and runs no model; logits runs one prompt through the model.
cmake_minimum_required(VERSION 3.14)
project(magnitude-llama-tools CXX C)

set(LLAMA_SOURCE "" CACHE PATH "llama.cpp checkout at the pinned commit")
if (NOT EXISTS "${LLAMA_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_SOURCE is not a llama.cpp checkout: '${LLAMA_SOURCE}'")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_SOURCE}" llama.cpp EXCLUDE_FROM_ALL)

add_executable(verify-publish verify_publish.cpp)
# the loader and the model are llama.cpp's internal types: the steps llama_model_load runs, without the tensors
target_include_directories(verify-publish PRIVATE "${LLAMA_SOURCE}/src")
target_link_libraries(verify-publish PRIVATE llama-common llama)
target_compile_features(verify-publish PRIVATE cxx_std_17)

add_executable(logits logits.cpp)
target_link_libraries(logits PRIVATE llama)
target_compile_features(logits PRIVATE cxx_std_17)
