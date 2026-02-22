#pragma once
#ifndef HEARTBEAT_H
#define HEARTBEAT_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace heartbeat {

// Forward declarations
class HeartbeatImpl;

/**
 * Voice configuration for TTS synthesis.
 */
struct VoiceConfig {
    std::string code;       // Voice code (e.g., "af", "am", "in_f")
    std::string name;       // Human-readable name
    std::string language;   // Language code (e.g., "en-us", "en-in")
    float speed = 1.0f;     // Speaking speed multiplier
    float pitch = 1.0f;     // Pitch adjustment
};

/**
 * Audio output configuration.
 */
struct AudioConfig {
    int sample_rate = 24000;    // Output sample rate in Hz
    int channels = 1;           // Number of channels (mono)
    int bit_depth = 16;         // Bits per sample
};

/**
 * Synthesis result containing audio data and metadata.
 */
struct SynthesisResult {
    std::vector<float> audio;       // Audio samples (normalized -1 to 1)
    int sample_rate;                // Sample rate of audio
    float duration_seconds;         // Duration in seconds
    int64_t inference_time_ms;      // Inference time in milliseconds
    bool success;                   // Whether synthesis succeeded
    std::string error_message;      // Error message if failed
};

/**
 * Main Heartbeat TTS engine class.
 * 
 * Example usage:
 * @code
 *   Heartbeat hb("models/kokoro.gguf");
 *   auto result = hb.synthesize("Hello, world!", "af");
 *   if (result.success) {
 *       hb.write_wav("output.wav", result);
 *   }
 * @endcode
 */
class Heartbeat {
public:
    /**
     * Initialize Heartbeat with a GGUF model file.
     * @param model_path Path to the Kokoro GGUF model file.
     * @throws std::runtime_error if model fails to load.
     */
    explicit Heartbeat(const std::string& model_path);
    
    /**
     * Destructor.
     */
    ~Heartbeat();
    
    // Non-copyable
    Heartbeat(const Heartbeat&) = delete;
    Heartbeat& operator=(const Heartbeat&) = delete;
    
    // Movable
    Heartbeat(Heartbeat&&) noexcept;
    Heartbeat& operator=(Heartbeat&&) noexcept;
    
    /**
     * Synthesize speech from text.
     * @param text Input text to synthesize.
     * @param voice_code Voice code (e.g., "af" for American Female).
     * @return SynthesisResult containing audio data or error.
     */
    SynthesisResult synthesize(const std::string& text, 
                                const std::string& voice_code = "af");
    
    /**
     * Synthesize speech with custom voice configuration.
     * @param text Input text to synthesize.
     * @param voice Voice configuration.
     * @return SynthesisResult containing audio data or error.
     */
    SynthesisResult synthesize(const std::string& text, 
                                const VoiceConfig& voice);
    
    /**
     * Write audio to a WAV file.
     * @param path Output file path.
     * @param result Synthesis result to write.
     * @return true if successful, false otherwise.
     */
    bool write_wav(const std::string& path, const SynthesisResult& result);
    
    /**
     * Get list of available voices.
     * @return Vector of available voice configurations.
     */
    std::vector<VoiceConfig> available_voices() const;
    
    /**
     * Get model information.
     * @return String describing the loaded model.
     */
    std::string model_info() const;
    
    /**
     * Check if phonemizer is available.
     * @return true if espeak-ng is available.
     */
    bool has_phonemizer() const;

private:
    std::unique_ptr<HeartbeatImpl> impl_;
};

// Predefined voice constants
namespace voices {
    static const VoiceConfig AMERICAN_FEMALE   = {"af", "American Female", "en-us", 1.0f, 1.0f};
    static const VoiceConfig AMERICAN_MALE     = {"am", "American Male", "en-us", 1.0f, 0.9f};
    static const VoiceConfig BRITISH_FEMALE    = {"bf", "British Female", "en-gb", 1.0f, 1.0f};
    static const VoiceConfig BRITISH_MALE      = {"bm", "British Male", "en-gb", 1.0f, 0.9f};
    static const VoiceConfig INDIAN_FEMALE     = {"in_f", "Indian Female", "en-in", 1.0f, 1.0f};
    static const VoiceConfig INDIAN_MALE       = {"in_m", "Indian Male", "en-in", 1.0f, 0.9f};
}

} // namespace heartbeat

#endif // HEARTBEAT_H
