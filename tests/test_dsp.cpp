/**
 * Test: DSP functionality
 */

#include "dsp.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace heartbeat;

const float EPSILON = 1e-5f;

bool approx_equal(float a, float b, float eps = EPSILON) {
    return std::abs(a - b) < eps;
}

void test_hanning_window() {
    std::cout << "Testing Hanning window..." << std::endl;
    
    DSP dsp;
    auto window = dsp.hanning_window(16);
    
    assert(window.size() == 16);
    
    // First and last should be close to 0
    assert(approx_equal(window[0], 0.0f, 0.01f));
    assert(approx_equal(window[15], 0.0f, 0.01f));
    
    // Middle should be close to 1
    assert(window[7] > 0.9f && window[7] <= 1.0f);
    assert(window[8] > 0.9f && window[8] <= 1.0f);
    
    // Should be symmetric
    for (int i = 0; i < 8; i++) {
        assert(approx_equal(window[i], window[15 - i], 0.01f));
    }
    
    std::cout << "  ✓ Hanning window passed" << std::endl;
}

void test_normalize() {
    std::cout << "Testing audio normalization..." << std::endl;
    
    DSP dsp;
    
    std::vector<float> audio = {0.5f, -0.3f, 0.8f, -0.2f, 0.1f};
    auto normalized = dsp.normalize(audio, 0.95f);
    
    // Find max after normalization
    float max_val = 0.0f;
    for (float s : normalized) {
        max_val = std::max(max_val, std::abs(s));
    }
    
    assert(approx_equal(max_val, 0.95f, 0.01f));
    
    std::cout << "  ✓ Audio normalization passed" << std::endl;
}

void test_apply_fades() {
    std::cout << "Testing fade application..." << std::endl;
    
    DSP dsp;
    
    std::vector<float> audio(1000, 1.0f);
    auto faded = dsp.apply_fades(audio, 100);
    
    assert(faded.size() == 1000);
    
    // Start should fade in
    assert(faded[0] < 0.1f);
    assert(faded[50] > 0.4f && faded[50] < 0.6f);
    assert(faded[100] > 0.9f);
    
    // End should fade out
    assert(faded[999] < 0.1f);
    assert(faded[950] > 0.4f && faded[950] < 0.6f);
    assert(faded[899] > 0.9f);
    
    std::cout << "  ✓ Fade application passed" << std::endl;
}

void test_float_to_pcm16() {
    std::cout << "Testing float to PCM16 conversion..." << std::endl;
    
    DSP dsp;
    
    std::vector<float> audio = {0.0f, 0.5f, 1.0f, -0.5f, -1.0f};
    auto pcm = dsp.float_to_pcm16(audio);
    
    assert(pcm.size() == 5);
    assert(pcm[0] == 0);
    assert(pcm[1] == 16383 || pcm[1] == 16384);  // ~0.5 * 32767
    assert(pcm[2] == 32767);
    assert(pcm[3] == -16383 || pcm[3] == -16384);
    assert(pcm[4] == -32767);
    
    std::cout << "  ✓ Float to PCM16 conversion passed" << std::endl;
}

void test_overlap_add() {
    std::cout << "Testing overlap-add..." << std::endl;
    
    DSP dsp;
    
    // Create overlapping frames
    std::vector<std::vector<float>> frames;
    frames.push_back({1.0f, 1.0f, 1.0f, 1.0f});
    frames.push_back({1.0f, 1.0f, 1.0f, 1.0f});
    frames.push_back({1.0f, 1.0f, 1.0f, 1.0f});
    
    auto result = dsp.overlap_add(frames, 2);
    
    // With hop=2 and frame_size=4, we expect length = 2*(3-1) + 4 = 8
    assert(result.size() == 8);
    
    // Overlapping regions should have higher values
    assert(result[2] >= 1.5f);  // Overlap of 2 frames
    assert(result[4] >= 1.5f);  // Overlap of 2 frames
    
    std::cout << "  ✓ Overlap-add passed" << std::endl;
}

void test_wav_writer() {
    std::cout << "Testing WAV writer..." << std::endl;
    
    // Generate a simple sine wave
    std::vector<float> audio(4800);  // 0.2 seconds at 24kHz
    for (size_t i = 0; i < audio.size(); i++) {
        float t = static_cast<float>(i) / 24000.0f;
        audio[i] = 0.5f * std::sin(2.0f * 3.14159f * 440.0f * t);  // 440 Hz
    }
    
    bool success = WavWriter::write("test_output.wav", audio, 24000, 1);
    assert(success);
    
    std::cout << "  ✓ WAV writer passed" << std::endl;
}

int main() {
    std::cout << "\n=== DSP Tests ===\n" << std::endl;
    
    try {
        test_hanning_window();
        test_normalize();
        test_apply_fades();
        test_float_to_pcm16();
        test_overlap_add();
        test_wav_writer();
        
        std::cout << "\n✓ All DSP tests passed!\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
