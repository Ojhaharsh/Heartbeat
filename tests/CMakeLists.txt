# Tests CMakeLists.txt
enable_testing()

# Test utilities
add_library(test_utils INTERFACE)
target_include_directories(test_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Phonemizer test
add_executable(test_phonemizer test_phonemizer.cpp)
target_link_libraries(test_phonemizer PRIVATE heartbeat_lib test_utils)
add_test(NAME phonemizer_test COMMAND test_phonemizer)
if(WIN32 AND COMMAND heartbeat_copy_runtime_dlls)
    heartbeat_copy_runtime_dlls(test_phonemizer)
endif()

# DSP test
add_executable(test_dsp test_dsp.cpp)
target_link_libraries(test_dsp PRIVATE heartbeat_lib test_utils)
add_test(NAME dsp_test COMMAND test_dsp)
if(WIN32 AND COMMAND heartbeat_copy_runtime_dlls)
    heartbeat_copy_runtime_dlls(test_dsp)
endif()

# GGUF loader test
add_executable(test_gguf_loader test_gguf_loader.cpp)
target_link_libraries(test_gguf_loader PRIVATE heartbeat_lib test_utils)
add_test(NAME gguf_loader_test COMMAND test_gguf_loader)
if(WIN32 AND COMMAND heartbeat_copy_runtime_dlls)
    heartbeat_copy_runtime_dlls(test_gguf_loader)
endif()

# Integration test
add_executable(test_integration test_integration.cpp)
target_link_libraries(test_integration PRIVATE heartbeat_lib test_utils)
add_test(NAME integration_test COMMAND test_integration)
if(WIN32 AND COMMAND heartbeat_copy_runtime_dlls)
    heartbeat_copy_runtime_dlls(test_integration)
endif()

if(WIN32)
    set(HEARTBEAT_TEST_PATH "PATH=$<TARGET_FILE_DIR:ggml>;$ENV{PATH}")
    set_tests_properties(
        phonemizer_test
        dsp_test
        gguf_loader_test
        integration_test
        PROPERTIES ENVIRONMENT "${HEARTBEAT_TEST_PATH}"
    )
endif()
