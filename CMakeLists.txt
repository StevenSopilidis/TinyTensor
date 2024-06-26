cmake_minimum_required(VERSION 3.14)
project(TinyTensor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    TinyTensor
    # ./main.cpp
    ./TinyTensor/src/Tensor.cpp
    ./TinyTensor/src/Cpu.cpp
    ./TinyTensor/tests/tensor_test.cpp
    ./TinyTensor/tests/cpu_test.cpp
)

find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h)
    add_custom_target(
        format
        COMMAND ${CLANG_FORMAT} -i -style=file ${ALL_SOURCE_FILES}
    )
endif()

find_program(CLANG_TIDY clang-tidy)
if(CLANG_TIDY)
    set(CLANG_TIDY_CHECKS
        "modernize-use-auto"
        "readability-identifier-naming"
        "performance-inefficient-string-concatenation"
        "bugprone-bool-pointer-implicit-conversion"
    )

    set(SOURCE_DIRS
        ./TintTensor/include
        ./TintTensor/src
        ./TintTensor/tests
    )

    file(GLOB_RECURSE MY_SOURCE_FILES
        ${SOURCE_DIRS}/*.cpp
        ${SOURCE_DIRS}/*.h
    )

    add_custom_target(
        tidy
        COMMAND ${CLANG_TIDY} -checks='${CLANG_TIDY_CHECKS}' ${MY_SOURCE_FILES} -p compile_commands.json
        COMMENT "Running clang-tidy"
    )
endif()

target_include_directories(TinyTensor PUBLIC ./TinyTensor/include)

target_link_libraries(
  TinyTensor PUBLIC
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(TinyTensor)
