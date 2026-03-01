#include "heartbeat.h"
#include "gguf_loader.h"
#include "phonemizer.h"
#include "model.h"
#include "dsp.h"

#include <chrono>
#include <stdexcept>
#include <iostream>
#include <cstdlib>

namespace heartbeat {

/**
 * Internal implementation (PIMPL pattern).
 */
class HeartbeatImpl {
public:
    GGUFLoader loader;
    Phonemizer phonemizer;
    Model model;
    DSP dsp;
    
    bool initialized = false;
    std::string model_path;
};

Heartbeat::Heartbeat(const std::string& model_path)
    : impl_(std::make_unique<HeartbeatImpl>()) {
    
    impl_->model_path = model_path;
    
    // Load GGUF model
    if (!impl_->loader.load(model_path)) {
        throw std::runtime_error("Failed to load model: " + model_path);
    }
    
    // Initialize phonemizer with vocabulary
    if (!impl_->phonemizer.initialize(impl_->loader.vocabulary())) {
        // Phonemizer failed, but we can continue without it
    }
    
    // Load model weights
    if (!impl_->model.load_weights(impl_->loader)) {
        throw std::runtime_error("Failed to load model weights");
    }
    
    // Configure DSP
    DSPConfig dsp_config;
    dsp_config.n_fft = impl_->loader.params().istft_n_fft;
    dsp_config.hop_length = impl_->loader.params().istft_hop_length;
    dsp_config.sample_rate = impl_->loader.params().sample_rate;
    impl_->dsp = DSP(dsp_config);
    
    impl_->initialized = true;
}

Heartbeat::~Heartbeat() = default;

Heartbeat::Heartbeat(Heartbeat&&) noexcept = default;
Heartbeat& Heartbeat::operator=(Heartbeat&&) noexcept = default;

SynthesisResult Heartbeat::synthesize(const std::string& text,
                                       const std::string& voice_code) {
    VoiceConfig config;
    config.code = voice_code;
    
    // Map voice code to language
    if (voice_code.find("in") == 0) {
        config.language = "en-in";
    } else if (voice_code.find("b") == 0) {
        config.language = "en-gb";
    } else {
        config.language = "en-us";
    }
    
    return synthesize(text, config);
}

SynthesisResult Heartbeat::synthesize(const std::string& text,
                                       const VoiceConfig& voice) {
    SynthesisResult result;
    result.success = false;
    const bool verbose = std::getenv("HEARTBEAT_VERBOSE") != nullptr;
    
    if (!impl_->initialized) {
        result.error_message = "Model not initialized";
        if (verbose) std::cerr << "DEBUG: Model not initialized\n";
        return result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Convert text to tokens
        if (verbose) std::cerr << "DEBUG: Converting text to tokens...\n";
        PhonemizerConfig ph_config;
        ph_config.language = voice.language;
        ph_config.rate = voice.speed;
        
        std::vector<int> tokens = impl_->phonemizer.text_to_tokens(text, ph_config);
        if (verbose) std::cerr << "DEBUG: Got " << tokens.size() << " tokens\n";
        
        if (tokens.empty()) {
            result.error_message = "Failed to phonemize text";
            if (verbose) std::cerr << "DEBUG: No tokens generated\n";
            return result;
        }
        
        // Get style vector for voice
        if (verbose) std::cerr << "DEBUG: Getting style vector for voice: " << voice.code << "\n";
        std::vector<float> style = impl_->model.get_style_vector(voice.code);
        if (verbose) std::cerr << "DEBUG: Style vector size: " << style.size() << "\n";
        
        // Run model forward pass
        if (verbose) std::cerr << "DEBUG: Running model forward pass...\n";
        ModelOutput model_out = impl_->model.forward(tokens, style);
        if (verbose) {
            std::cerr << "DEBUG: Model output - mag size: " << model_out.magnitude.size() 
                      << ", phase size: " << model_out.phase.size() << "\n";
        }
        
        if (model_out.magnitude.empty() || (model_out.n_mels > 1 && model_out.phase.empty())) {
            result.error_message = "Model forward pass failed: empty output";
            if (verbose) std::cerr << "DEBUG: Model forward pass returned empty/invalid output\n";
            return result;
        }
        
        // Synthesize audio
        if (model_out.n_mels == 1 && model_out.phase.empty()) {
            if (verbose) std::cerr << "DEBUG: Model output is raw audio, skipping ISTFT\n";
            result.audio = model_out.magnitude;
        } else {
            if (verbose) std::cerr << "DEBUG: Running ISTFT...\n";
            result.audio = impl_->dsp.istft(
                model_out.magnitude,
                model_out.phase,
                model_out.n_mels,
                model_out.n_frames
            );
        }
        if (verbose) std::cerr << "DEBUG: Audio samples: " << result.audio.size() << "\n";
        
        // Post-processing
        if (verbose && result.audio.size() > 0) {
            std::cerr << "DEBUG Samples: ";
            for (int i = 0; i < std::min((int)result.audio.size(), 10); i++) {
                std::cerr << result.audio[i] << " ";
            }
            std::cerr << "...\n";
        }
        result.audio = impl_->dsp.normalize(result.audio, 0.95f);
        if (verbose && result.audio.size() > 0) {
            std::cerr << "DEBUG Normalized Samples: ";
            for (int i = 0; i < std::min((int)result.audio.size(), 10); i++) {
                std::cerr << result.audio[i] << " ";
            }
            std::cerr << "...\n";
        }
        result.audio = impl_->dsp.apply_fades(result.audio, 256);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        );
        
        result.sample_rate = impl_->loader.params().sample_rate;
        result.duration_seconds = static_cast<float>(result.audio.size()) / result.sample_rate;
        result.inference_time_ms = duration.count();
        result.success = true;
        if (verbose) std::cerr << "DEBUG: Synthesis complete!\n";
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Synthesis error: ") + e.what();
        if (verbose) std::cerr << "DEBUG: Exception: " << e.what() << "\n";
    }
    
    return result;
}

bool Heartbeat::write_wav(const std::string& path, const SynthesisResult& result) {
    if (!result.success || result.audio.empty()) {
        return false;
    }
    
    return WavWriter::write(path, result.audio, result.sample_rate, 1);
}

std::vector<VoiceConfig> Heartbeat::available_voices() const {
    return {
        voices::AMERICAN_FEMALE,
        voices::AMERICAN_MALE,
        voices::BRITISH_FEMALE,
        voices::BRITISH_MALE,
        voices::INDIAN_FEMALE,
        voices::INDIAN_MALE,
    };
}

std::string Heartbeat::model_info() const {
    if (!impl_->initialized) {
        return "Model not loaded";
    }
    
    const auto& params = impl_->loader.params();
    
    std::string info;
    info += "Kokoro-82M Native Inference Engine\n";
    info += "  Vocab Size:    " + std::to_string(params.vocab_size) + "\n";
    info += "  Hidden Size:   " + std::to_string(params.hidden_size) + "\n";
    info += "  Layers:        " + std::to_string(params.num_layers) + "\n";
    info += "  Heads:         " + std::to_string(params.num_heads) + "\n";
    info += "  Style Dim:     " + std::to_string(params.style_dim) + "\n";
    info += "  Sample Rate:   " + std::to_string(params.sample_rate) + " Hz\n";
    
    return info;
}

bool Heartbeat::has_phonemizer() const {
    return impl_->phonemizer.is_available();
}

} // namespace heartbeat
