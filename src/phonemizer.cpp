#include "phonemizer.h"

#include <algorithm>
#include <sstream>
#include <regex>
#include <cctype>

#ifdef HEARTBEAT_HAS_ESPEAK
extern "C" {
#include <espeak-ng/speak_lib.h>
}
#endif

namespace heartbeat {

Phonemizer::Phonemizer() {
    // Initialize language mappings
    language_maps_["en-us"] = {
        "en-us",
        {}  // Use default IPA mapping
    };
    
    language_maps_["en-gb"] = {
        "en-gb",
        {}
    };
    
    language_maps_["en-in"] = {
        "en-in",
        {}
    };
}

Phonemizer::~Phonemizer() {
    cleanup_espeak();
}

bool Phonemizer::init_espeak() {
#ifdef HEARTBEAT_HAS_ESPEAK
    if (espeak_initialized_) return true;
    
    // Initialize espeak-ng
    int result = espeak_Initialize(
        AUDIO_OUTPUT_RETRIEVAL,  // Don't play audio
        0,                        // Buffer length (use default)
        nullptr,                  // Path to espeak-ng-data (use default)
        0                         // Options
    );
    
    if (result == -1) {
        return false;
    }
    
    espeak_initialized_ = true;
    return true;
#else
    return false;
#endif
}

void Phonemizer::cleanup_espeak() {
#ifdef HEARTBEAT_HAS_ESPEAK
    if (espeak_initialized_) {
        espeak_Terminate();
        espeak_initialized_ = false;
    }
#endif
}

bool Phonemizer::initialize(const std::vector<std::string>& vocab) {
    vocab_ = vocab;
    vocab_map_.clear();
    
    for (size_t i = 0; i < vocab.size(); i++) {
        vocab_map_[vocab[i]] = static_cast<int>(i);
    }
    
    // Find special tokens
    auto find_token = [this](const std::vector<std::string>& candidates) -> int {
        for (const auto& token : candidates) {
            auto it = vocab_map_.find(token);
            if (it != vocab_map_.end()) {
                return it->second;
            }
        }
        return -1;
    };
    
    pad_token_id_ = find_token({"<pad>", "[PAD]", "<PAD>"});
    bos_token_id_ = find_token({"<s>", "[CLS]", "<BOS>"});
    eos_token_id_ = find_token({"</s>", "[SEP]", "<EOS>"});
    unk_token_id_ = find_token({"<unk>", "[UNK]", "<UNK>"});
    
    if (pad_token_id_ < 0) pad_token_id_ = 0;
    if (unk_token_id_ < 0) unk_token_id_ = 0;
    
    // Try to initialize espeak
    initialized_ = init_espeak();
    
    // Even without espeak, we can do basic phonemization
    if (!initialized_ && !vocab_.empty()) {
        initialized_ = true;
    }
    
    return initialized_;
}

std::string Phonemizer::text_to_ipa(const std::string& text, 
                                     const std::string& language) {
#ifdef HEARTBEAT_HAS_ESPEAK
    if (!espeak_initialized_ && !init_espeak()) {
        // Fallback: return text as-is
        return text;
    }
    
    // Set voice for language
    auto it = language_maps_.find(language);
    std::string voice = (it != language_maps_.end()) ? it->second.espeak_voice : "en-us";
    espeak_SetVoiceByName(voice.c_str());
    
    // Convert text to phonemes
    const char* text_ptr = text.c_str();
    std::string result;
    
    while (*text_ptr) {
        const char* phonemes = espeak_TextToPhonemes(
            reinterpret_cast<const void**>(&text_ptr),
            espeakCHARS_UTF8,
            espeakPHONEMES_IPA
        );
        
        if (phonemes) {
            result += phonemes;
        }
    }
    
    return result;
#else
    // Fallback: basic letter-to-phoneme (very simplified)
    return text;
#endif
}

std::vector<Phoneme> Phonemizer::ipa_to_phonemes(const std::string& ipa) {
    std::vector<Phoneme> phonemes;
    
    // Simple character-by-character parsing
    // In a real implementation, we'd need proper IPA parsing
    std::string current;
    
    for (size_t i = 0; i < ipa.size(); ) {
        // Check for multi-character phonemes first
        bool found = false;
        
        // Try 3-char, then 2-char, then 1-char
        for (int len = 3; len >= 1 && !found; len--) {
            if (i + len <= ipa.size()) {
                std::string candidate = ipa.substr(i, len);
                auto it = vocab_map_.find(candidate);
                
                if (it != vocab_map_.end()) {
                    phonemes.push_back({candidate, it->second, 0.0f});
                    i += len;
                    found = true;
                }
            }
        }
        
        if (!found) {
            // Unknown character - use UNK or skip
            if (unk_token_id_ >= 0 && !std::isspace(static_cast<unsigned char>(ipa[i]))) {
                phonemes.push_back({"<unk>", unk_token_id_, 0.0f});
            }
            i++;
        }
    }
    
    return phonemes;
}

std::vector<int> Phonemizer::text_to_tokens(const std::string& text,
                                             const PhonemizerConfig& config) {
    std::vector<int> tokens;
    
    // Normalize text
    std::string normalized = normalize_text(text);
    
    // If we have no vocabulary, use simple character-level tokenization
    if (vocab_map_.empty()) {
        // Use character indices as tokens (simple fallback)
        // Kokoro expects phoneme tokens 1-177, we'll map characters to this range
        for (char c : normalized) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                // Map a-z to tokens 1-26
                int token = (c - 'a') + 1;
                tokens.push_back(token);
            } else if (c == ' ') {
                // Space token
                tokens.push_back(0);
            }
        }
        return tokens;
    }
    
    // Add BOS token
    if (bos_token_id_ >= 0) {
        tokens.push_back(bos_token_id_);
    }
    
    // Convert to IPA
    std::string ipa = text_to_ipa(normalized, config.language);
    
    // Parse IPA to phonemes
    auto phonemes = ipa_to_phonemes(ipa);
    
    // Extract token IDs
    for (const auto& ph : phonemes) {
        if (ph.token_id >= 0) {
            tokens.push_back(ph.token_id);
        }
    }
    
    // Add EOS token
    if (eos_token_id_ >= 0) {
        tokens.push_back(eos_token_id_);
    }
    
    return tokens;
}

int Phonemizer::get_token_id(const std::string& phoneme) const {
    auto it = vocab_map_.find(phoneme);
    return (it != vocab_map_.end()) ? it->second : unk_token_id_;
}

std::vector<std::string> Phonemizer::available_languages() const {
    std::vector<std::string> languages;
    for (const auto& [lang, _] : language_maps_) {
        languages.push_back(lang);
    }
    return languages;
}

void Phonemizer::set_language(const std::string& language) {
    current_language_ = language;
    
#ifdef HEARTBEAT_HAS_ESPEAK
    if (espeak_initialized_) {
        auto it = language_maps_.find(language);
        std::string voice = (it != language_maps_.end()) ? it->second.espeak_voice : "en-us";
        espeak_SetVoiceByName(voice.c_str());
    }
#endif
}

// Utility functions

std::string normalize_text(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    
    for (char c : text) {
        // Convert to lowercase (basic ASCII)
        if (c >= 'A' && c <= 'Z') {
            result += (c + 32);
        } else {
            result += c;
        }
    }
    
    // TODO: Expand numbers, abbreviations, etc.
    // This is simplified for now
    
    return result;
}

std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string current;
    
    for (char c : text) {
        current += c;
        
        // Check for sentence boundaries
        if (c == '.' || c == '!' || c == '?' || c == ';') {
            // Trim whitespace
            while (!current.empty() && std::isspace(current.front())) {
                current.erase(0, 1);
            }
            while (!current.empty() && std::isspace(current.back())) {
                current.pop_back();
            }
            
            if (!current.empty()) {
                sentences.push_back(current);
                current.clear();
            }
        }
    }
    
    // Don't forget the last sentence
    if (!current.empty()) {
        while (!current.empty() && std::isspace(current.front())) {
            current.erase(0, 1);
        }
        while (!current.empty() && std::isspace(current.back())) {
            current.pop_back();
        }
        if (!current.empty()) {
            sentences.push_back(current);
        }
    }
    
    return sentences;
}

} // namespace heartbeat
