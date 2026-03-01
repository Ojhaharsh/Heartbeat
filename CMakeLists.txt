cmake_minimum_required(VERSION 3.16)
project(Heartbeat VERSION 0.1.0 LANGUAGES C CXX)

# ==============================================================================
# Project Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ==============================================================================
# Options
# ==============================================================================
option(HEARTBEAT_AVX2 "Enable AVX2 optimizations" ON)
option(HEARTBEAT_BUILD_TESTS "Build unit tests" ON)

# ==============================================================================
# Compiler Flags
# ==============================================================================
if(MSVC)
    add_compile_options(/W4 /utf-8)
    if(HEARTBEAT_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(HEARTBEAT_AVX2)
        add_compile_options(-mavx2 -mfma -mf16c)
    endif()
endif()

# ==============================================================================
# Dependencies
# ==============================================================================

# GGML - Add as submodule
add_subdirectory(extern/ggml EXCLUDE_FROM_ALL)

# KissFFT - Compile the C source
add_library(kissfft STATIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/kissfft/kiss_fft.c)
target_include_directories(kissfft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/kissfft)
target_compile_definitions(kissfft PUBLIC kiss_fft_scalar=float)

# espeak-ng - Find system installation
if(WIN32)
    # Windows: Look in common installation paths
    find_path(ESPEAK_NG_INCLUDE_DIR
        NAMES speak_lib.h
        PATHS
            "C:/Program Files/eSpeak NG/include"
            "C:/Program Files (x86)/eSpeak NG/include"
            "$ENV{ESPEAK_NG_DIR}/include"
    )
    find_library(ESPEAK_NG_LIBRARY
        NAMES espeak-ng libespeak-ng
        PATHS
            "C:/Program Files/eSpeak NG/lib"
            "C:/Program Files (x86)/eSpeak NG/lib"
            "$ENV{ESPEAK_NG_DIR}/lib"
    )
else()
    # Linux/macOS: Use pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ESPEAK_NG espeak-ng)
    endif()
    if(NOT ESPEAK_NG_FOUND)
        find_path(ESPEAK_NG_INCLUDE_DIR NAMES espeak-ng/speak_lib.h)
        find_library(ESPEAK_NG_LIBRARY NAMES espeak-ng)
    endif()
endif()

if(ESPEAK_NG_INCLUDE_DIR AND ESPEAK_NG_LIBRARY)
    add_library(espeak-ng INTERFACE)
    target_include_directories(espeak-ng INTERFACE ${ESPEAK_NG_INCLUDE_DIR})
    target_link_libraries(espeak-ng INTERFACE ${ESPEAK_NG_LIBRARY})
    message(STATUS "Found espeak-ng: ${ESPEAK_NG_LIBRARY}")
else()
    message(WARNING "espeak-ng not found! Phonemizer will be disabled.")
    set(ESPEAK_NG_FOUND FALSE)
endif()

# ==============================================================================
# Runtime DLL Staging (Windows)
# ==============================================================================
function(heartbeat_copy_runtime_dlls target_name)
    if(NOT WIN32)
        return()
    endif()

    if(TARGET ggml AND TARGET ggml-cpu AND TARGET ggml-base)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ggml>
                $<TARGET_FILE_DIR:${target_name}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ggml-cpu>
                $<TARGET_FILE_DIR:${target_name}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ggml-base>
                $<TARGET_FILE_DIR:${target_name}>
            VERBATIM
        )
    endif()
endfunction()

# ==============================================================================
# Heartbeat Library
# ==============================================================================
set(HEARTBEAT_SOURCES
    src/gguf_loader.cpp
    src/phonemizer.cpp
    src/model.cpp
    src/dsp.cpp
    src/heartbeat.cpp
)

set(HEARTBEAT_HEADERS
    include/heartbeat.h
    include/gguf_loader.h
    include/phonemizer.h
    include/model.h
    include/dsp.h
)

add_library(heartbeat_lib STATIC ${HEARTBEAT_SOURCES} ${HEARTBEAT_HEADERS})
target_include_directories(heartbeat_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(heartbeat_lib PUBLIC
    ggml
    kissfft
)

if(ESPEAK_NG_INCLUDE_DIR AND ESPEAK_NG_LIBRARY)
    target_link_libraries(heartbeat_lib PUBLIC espeak-ng)
    target_compile_definitions(heartbeat_lib PUBLIC HEARTBEAT_HAS_ESPEAK)
endif()

# ==============================================================================
# Heartbeat CLI Executable
# ==============================================================================
add_executable(heartbeat src/main.cpp)
target_link_libraries(heartbeat PRIVATE heartbeat_lib)
heartbeat_copy_runtime_dlls(heartbeat)

# ==============================================================================
# Installation
# ==============================================================================
install(TARGETS heartbeat RUNTIME DESTINATION bin)
install(FILES ${HEARTBEAT_HEADERS} DESTINATION include/heartbeat)

# ==============================================================================
# Tests
# ==============================================================================
if(HEARTBEAT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ==============================================================================
# Summary
# ==============================================================================
message(STATUS "")
message(STATUS "=== Heartbeat Build Configuration ===")
message(STATUS "Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "AVX2:           ${HEARTBEAT_AVX2}")
message(STATUS "espeak-ng:      ${ESPEAK_NG_FOUND}")
message(STATUS "=====================================")
