/**
 * Test: GGUF Loader functionality
 */

#include "gguf_loader.h"
#include <iostream>
#include <cassert>

using namespace heartbeat;

void test_gguf_constants() {
    std::cout << "Testing GGUF constants..." << std::endl;
    
    assert(gguf::MAGIC == 0x46554747);  // "GGUF"
    assert(gguf::VERSION_2 == 2);
    assert(gguf::VERSION_3 == 3);
    
    std::cout << "  ✓ GGUF constants correct" << std::endl;
}

void test_kokoro_params_defaults() {
    std::cout << "Testing KokoroParams defaults..." << std::endl;
    
    KokoroParams params;
    
    assert(params.vocab_size == 178);
    assert(params.hidden_size == 768);
    assert(params.num_layers == 12);
    assert(params.num_heads == 12);
    assert(params.style_dim == 256);
    assert(params.n_mels == 80);
    assert(params.sample_rate == 24000);
    
    std::cout << "  ✓ KokoroParams defaults correct" << std::endl;
}

void test_gguf_loader_init() {
    std::cout << "Testing GGUFLoader initialization..." << std::endl;
    
    GGUFLoader loader;
    
    assert(!loader.is_loaded());
    assert(loader.vocabulary().empty());
    
    std::cout << "  ✓ GGUFLoader initialization correct" << std::endl;
}

void test_loader_missing_file() {
    std::cout << "Testing loader with missing file..." << std::endl;
    
    GGUFLoader loader;
    bool success = loader.load("nonexistent_file.gguf");
    
    assert(!success);
    assert(!loader.is_loaded());
    
    std::cout << "  ✓ Missing file handled correctly" << std::endl;
}

void test_phoneme_to_id() {
    std::cout << "Testing phoneme_to_id..." << std::endl;
    
    GGUFLoader loader;
    
    // Without loading, should return -1
    int id = loader.phoneme_to_id("test");
    assert(id == -1);
    
    std::cout << "  ✓ phoneme_to_id correct" << std::endl;
}

void test_tensor_names() {
    std::cout << "Testing tensor_names..." << std::endl;
    
    GGUFLoader loader;
    auto names = loader.tensor_names();
    
    // Empty when not loaded
    assert(names.empty());
    
    std::cout << "  ✓ tensor_names correct" << std::endl;
}

int main() {
    std::cout << "\n=== GGUF Loader Tests ===\n" << std::endl;
    
    try {
        test_gguf_constants();
        test_kokoro_params_defaults();
        test_gguf_loader_init();
        test_loader_missing_file();
        test_phoneme_to_id();
        test_tensor_names();
        
        std::cout << "\n✓ All GGUF loader tests passed!\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
