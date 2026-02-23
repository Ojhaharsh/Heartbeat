# Tests CMakeLists.txt
enable_testing()

# Test utilities
add_library(test_utils INTERFACE)
target_include_directories(test_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Phonemizer test
add_executable(test_phonemizer test_phonemizer.cpp)
target_link_libraries(test_phonemizer PRIVATE heartbeat_lib test_utils)
add_test(NAME phonemizer_test COMMAND test_phonemizer)

# DSP test
add_executable(test_dsp test_dsp.cpp)
target_link_libraries(test_dsp PRIVATE heartbeat_lib test_utils)
add_test(NAME dsp_test COMMAND test_dsp)

# GGUF loader test
add_executable(test_gguf_loader test_gguf_loader.cpp)
target_link_libraries(test_gguf_loader PRIVATE heartbeat_lib test_utils)
add_test(NAME gguf_loader_test COMMAND test_gguf_loader)

# Integration test
add_executable(test_integration test_integration.cpp)
target_link_libraries(test_integration PRIVATE heartbeat_lib test_utils)
add_test(NAME integration_test COMMAND test_integration)
