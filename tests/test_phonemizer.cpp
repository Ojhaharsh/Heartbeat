/**
 * Test: Phonemizer functionality
 */

#include "phonemizer.h"
#include <iostream>
#include <cassert>

using namespace heartbeat;

void test_normalize_text() {
    std::cout << "Testing text normalization..." << std::endl;
    
    assert(normalize_text("Hello World") == "hello world");
    assert(normalize_text("UPPERCASE") == "uppercase");
    assert(normalize_text("Mixed Case") == "mixed case");
    
    std::cout << "  ✓ Text normalization passed" << std::endl;
}

void test_split_sentences() {
    std::cout << "Testing sentence splitting..." << std::endl;
    
    auto sentences = split_sentences("Hello. How are you? I'm fine!");
    assert(sentences.size() == 3);
    assert(sentences[0] == "Hello.");
    assert(sentences[1] == "How are you?");
    assert(sentences[2] == "I'm fine!");
    
    std::cout << "  ✓ Sentence splitting passed" << std::endl;
}

void test_phonemizer_init() {
    std::cout << "Testing phonemizer initialization..." << std::endl;
    
    Phonemizer phonemizer;
    
    // Create a simple vocabulary
    std::vector<std::string> vocab = {
        "<pad>", "<s>", "</s>", "<unk>",
        "h", "e", "l", "o", " ", "w", "r", "d"
    };
    
    bool success = phonemizer.initialize(vocab);
    // Even without espeak, basic init should succeed
    assert(success);
    
    std::cout << "  ✓ Phonemizer initialization passed" << std::endl;
}

void test_token_lookup() {
    std::cout << "Testing token lookup..." << std::endl;
    
    Phonemizer phonemizer;
    std::vector<std::string> vocab = {
        "<pad>", "<s>", "</s>", "<unk>",
        "h", "e", "l", "o"
    };
    phonemizer.initialize(vocab);
    
    assert(phonemizer.get_token_id("h") == 4);
    assert(phonemizer.get_token_id("e") == 5);
    assert(phonemizer.get_token_id("l") == 6);
    assert(phonemizer.get_token_id("o") == 7);
    assert(phonemizer.get_token_id("unknown") == 3);  // UNK token
    
    std::cout << "  ✓ Token lookup passed" << std::endl;
}

int main() {
    std::cout << "\n=== Phonemizer Tests ===\n" << std::endl;
    
    try {
        test_normalize_text();
        test_split_sentences();
        test_phonemizer_init();
        test_token_lookup();
        
        std::cout << "\n✓ All phonemizer tests passed!\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
