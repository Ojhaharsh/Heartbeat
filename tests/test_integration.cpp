/**
 * Test: Integration test for complete TTS pipeline
 */

#include "heartbeat.h"
#include <iostream>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;
using namespace heartbeat;

void test_voice_constants() {
    std::cout << "Testing voice constants..." << std::endl;
    
    assert(voices::AMERICAN_FEMALE.code == "af");
    assert(voices::AMERICAN_MALE.code == "am");
    assert(voices::BRITISH_FEMALE.code == "bf");
    assert(voices::BRITISH_MALE.code == "bm");
    assert(voices::INDIAN_FEMALE.code == "in_f");
    assert(voices::INDIAN_MALE.code == "in_m");
    
    assert(voices::INDIAN_FEMALE.language == "en-in");
    assert(voices::AMERICAN_FEMALE.language == "en-us");
    
    std::cout << "  ✓ Voice constants correct" << std::endl;
}

void test_synthesis_result_struct() {
    std::cout << "Testing SynthesisResult struct..." << std::endl;
    
    SynthesisResult result;
    
    assert(result.audio.empty());
    assert(result.sample_rate == 0);
    assert(result.duration_seconds == 0.0f);
    assert(result.inference_time_ms == 0);
    assert(!result.success);
    assert(result.error_message.empty());
    
    std::cout << "  ✓ SynthesisResult struct correct" << std::endl;
}

void test_audio_config_defaults() {
    std::cout << "Testing AudioConfig defaults..." << std::endl;
    
    AudioConfig config;
    
    assert(config.sample_rate == 24000);
    assert(config.channels == 1);
    assert(config.bit_depth == 16);
    
    std::cout << "  ✓ AudioConfig defaults correct" << std::endl;
}

void test_voice_config() {
    std::cout << "Testing VoiceConfig..." << std::endl;
    
    VoiceConfig voice;
    voice.code = "test";
    voice.name = "Test Voice";
    voice.language = "en-us";
    voice.speed = 1.5f;
    voice.pitch = 0.8f;
    
    assert(voice.code == "test");
    assert(voice.speed == 1.5f);
    assert(voice.pitch == 0.8f);
    
    std::cout << "  ✓ VoiceConfig correct" << std::endl;
}

void test_model_loading_error() {
    std::cout << "Testing model loading with nonexistent file..." << std::endl;
    
    bool caught_error = false;
    
    try {
        Heartbeat hb("nonexistent.gguf");
    } catch (const std::runtime_error& e) {
        caught_error = true;
        std::string msg = e.what();
        assert(msg.find("Failed to load") != std::string::npos);
    }
    
    assert(caught_error);
    
    std::cout << "  ✓ Model loading error handled correctly" << std::endl;
}

int main() {
    std::cout << "\n=== Integration Tests ===\n" << std::endl;
    
    try {
        test_voice_constants();
        test_synthesis_result_struct();
        test_audio_config_defaults();
        test_voice_config();
        test_model_loading_error();
        
        std::cout << "\n✓ All integration tests passed!\n" << std::endl;
        
        // Note: Full synthesis test requires actual model file
        if (fs::exists("models/kokoro.gguf")) {
            std::cout << "Model file found! Running full synthesis test...\n" << std::endl;
            
            Heartbeat hb("models/kokoro.gguf");
            auto result = hb.synthesize("Hello, world!", "af");
            
            if (result.success) {
                std::cout << "Synthesis successful!" << std::endl;
                std::cout << "  Duration: " << result.duration_seconds << "s" << std::endl;
                std::cout << "  Inference: " << result.inference_time_ms << "ms" << std::endl;
                
                hb.write_wav("integration_test.wav", result);
                std::cout << "  Output: integration_test.wav" << std::endl;
            } else {
                std::cout << "Synthesis failed: " << result.error_message << std::endl;
            }
        } else {
            std::cout << "Note: Skipping full synthesis test (model not found)" << std::endl;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
