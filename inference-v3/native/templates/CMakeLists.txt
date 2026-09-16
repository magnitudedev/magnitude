cmake_minimum_required(VERSION 3.24)
project(templates LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(EXTRACTED "${CMAKE_CURRENT_BINARY_DIR}/extracted")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/prepare.py" "${EXTRACTED}"
    COMMAND_ERROR_IS_FATAL ANY OUTPUT_QUIET)
add_library(templates_core OBJECT
    src/templates-support.cpp
    "${EXTRACTED}/common/chat-auto-parser-generator.cpp"
    "${EXTRACTED}/common/chat-auto-parser-helpers.cpp"
    "${EXTRACTED}/common/chat-diff-analyzer.cpp"
    "${EXTRACTED}/common/chat-peg-parser.cpp"
    "${EXTRACTED}/common/chat.cpp"
    "${EXTRACTED}/common/jinja/caps.cpp"
    "${EXTRACTED}/common/jinja/lexer.cpp"
    "${EXTRACTED}/common/jinja/parser.cpp"
    "${EXTRACTED}/common/jinja/runtime.cpp"
    "${EXTRACTED}/common/jinja/string.cpp"
    "${EXTRACTED}/common/jinja/value.cpp"
    "${EXTRACTED}/common/json-schema-to-grammar.cpp"
    "${EXTRACTED}/common/json-schema.cpp"
    "${EXTRACTED}/common/json.cpp"
    "${EXTRACTED}/common/parsers/cohere2moe.cpp"
    "${EXTRACTED}/common/parsers/deepseek.cpp"
    "${EXTRACTED}/common/parsers/functionary-v3-2.cpp"
    "${EXTRACTED}/common/parsers/gemma4.cpp"
    "${EXTRACTED}/common/parsers/gigachat-v3.cpp"
    "${EXTRACTED}/common/parsers/gpt-oss.cpp"
    "${EXTRACTED}/common/parsers/kimi-k2.cpp"
    "${EXTRACTED}/common/parsers/kimi-k3.cpp"
    "${EXTRACTED}/common/parsers/lfm2.cpp"
    "${EXTRACTED}/common/parsers/minicpm5.cpp"
    "${EXTRACTED}/common/parsers/minimax-m3.cpp"
    "${EXTRACTED}/common/parsers/ministral3.cpp"
    "${EXTRACTED}/common/parsers/muse-glimmer.cpp"
    "${EXTRACTED}/common/parsers/parsers.cpp"
    "${EXTRACTED}/common/parsers/qwen3-coder.cpp"
    "${EXTRACTED}/common/peg-parser.cpp"
    "${EXTRACTED}/common/trie.cpp"
    "${EXTRACTED}/common/unicode.cpp"
)
target_compile_features(templates_core PUBLIC cxx_std_17)
set_target_properties(templates_core PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(templates_core PUBLIC
    "${EXTRACTED}/common" "${EXTRACTED}/vendor" "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE IDENTITY_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*" "${CMAKE_CURRENT_SOURCE_DIR}/include/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/*" "${CMAKE_CURRENT_SOURCE_DIR}/tools/*")
list(APPEND IDENTITY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/manifest.json" "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
list(SORT IDENTITY_FILES)
set(IDENTITY "${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_SYSTEM_NAME};${CMAKE_SYSTEM_PROCESSOR};${CMAKE_BUILD_TYPE}")
foreach(INPUT IN LISTS IDENTITY_FILES)
    file(SHA256 "${INPUT}" INPUT_HASH)
    string(APPEND IDENTITY ";${INPUT_HASH}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${INPUT}")
endforeach()
string(SHA256 BUILD_ID "${IDENTITY}")
add_library(templates SHARED src/abi.cpp src/output-stream.cpp src/schema-validation.cpp)
target_link_libraries(templates PRIVATE templates_core)
target_include_directories(templates PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(templates PRIVATE TEMPLATES_BUILD_ID="${BUILD_ID}")
set_target_properties(templates PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(templates PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map")
    set_property(TARGET templates APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map")
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test-output-text tests/test-output-text.cpp)
    target_link_libraries(test-output-text PRIVATE templates_core)
    target_compile_options(test-output-text PRIVATE -UNDEBUG)
    add_test(NAME output-text COMMAND test-output-text)
    add_executable(test-stream-prefixes tests/test-stream-prefixes.cpp)
    target_link_libraries(test-stream-prefixes PRIVATE templates_core)
    target_compile_options(test-stream-prefixes PRIVATE -UNDEBUG)
    add_test(NAME stream-prefixes COMMAND test-stream-prefixes)
    add_executable(test-peg-parser
        "${EXTRACTED}/tests/test-peg-parser.cpp"
        "${EXTRACTED}/tests/peg-parser/test-basic.cpp"
        "${EXTRACTED}/tests/peg-parser/test-unicode.cpp"
        "${EXTRACTED}/tests/peg-parser/test-json-parser.cpp"
        "${EXTRACTED}/tests/peg-parser/test-gbnf-generation.cpp"
        "${EXTRACTED}/tests/peg-parser/test-json-serialization.cpp"
        "${EXTRACTED}/tests/peg-parser/test-python-dict-parser.cpp"
        "${EXTRACTED}/tests/peg-parser/simple-tokenize.cpp")
    target_link_libraries(test-peg-parser PRIVATE templates_core)
    add_test(NAME peg-parser COMMAND test-peg-parser)
    foreach(test_name jinja chat-peg-parser chat-auto-parser chat-analysis chat-template)
        add_executable(test-${test_name} "${EXTRACTED}/tests/test-${test_name}.cpp")
        target_link_libraries(test-${test_name} PRIVATE templates_core)
        target_compile_options(test-${test_name} PRIVATE -UNDEBUG)
        if(NOT test_name STREQUAL "chat-analysis")
            add_test(NAME ${test_name} COMMAND test-${test_name})
            set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY "${EXTRACTED}")
        endif()
    endforeach()
    target_sources(test-jinja PRIVATE "${EXTRACTED}/common/subproc.cpp")
    target_compile_definitions(test-jinja PRIVATE LLAMA_SUBPROCESS)
    target_sources(test-chat-peg-parser PRIVATE "${EXTRACTED}/tests/peg-parser/simple-tokenize.cpp")
endif()
