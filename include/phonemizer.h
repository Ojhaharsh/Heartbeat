#pragma once
#ifndef PHONEMIZER_H
#define PHONEMIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace heartbeat {

/**
 * Phoneme with timing information.
 */
struct Phoneme {
    std::string symbol;     // IPA symbol
    int token_id;           // Vocabulary token ID
    float duration_hint;    // Optional duration hint (0 = auto)
};

/**
 * Phonemizer configuration.
 */
struct PhonemizerConfig {
    std::string language = "en-us";     // espeak-ng language code
    float rate = 1.0f;                  // Speaking rate
    bool preserve_punctuation = true;   // Keep punctuation markers
};

/**
 * Text-to-phoneme converter using espeak-ng.
 * 
 * Converts text input to IPA phonemes and maps them to
 * vocabulary token IDs for the Kokoro model.
 */
class Phonemizer {
public:
    Phonemizer();
    ~Phonemizer();
    
    /**
     * Initialize the phonemizer with vocabulary mapping.
     * @param vocab Vector of phoneme tokens from GGUF vocabulary.
     * @return true if initialization successful.
     */
    bool initialize(const std::vector<std::string>& vocab);
    
    /**
     * Check if phonemizer is available.
     * @return true if espeak-ng is available and initialized.
     */
    bool is_available() const { return initialized_; }
    
    /**
     * Convert text to phoneme token IDs.
     * @param text Input text.
     * @param config Phonemizer configuration.
     * @return Vector of token IDs.
     */
    std::vector<int> text_to_tokens(const std::string& text,
                                     const PhonemizerConfig& config = {});
    
    /**
     * Convert text to IPA phonemes.
     * @param text Input text.
     * @param language espeak-ng language code.
     * @return IPA phoneme string.
     */
    std::string text_to_ipa(const std::string& text,
                            const std::string& language = "en-us");
    
    /**
     * Get token ID for a phoneme symbol.
     * @param phoneme IPA phoneme symbol.
     * @return Token ID or -1 if not found.
     */
    int get_token_id(const std::string& phoneme) const;
    
    /**
     * Get available languages.
     */
    std::vector<std::string> available_languages() const;
    
    /**
     * Set voice for language.
     * @param language Language code (e.g., "en-us", "en-in").
     */
    void set_language(const std::string& language);

private:
    bool init_espeak();
    void cleanup_espeak();
    std::vector<Phoneme> ipa_to_phonemes(const std::string& ipa);
    
    bool initialized_ = false;
    bool espeak_initialized_ = false;
    
    std::unordered_map<std::string, int> vocab_map_;
    std::vector<std::string> vocab_;
    
    // Special token IDs
    int pad_token_id_ = 0;
    int bos_token_id_ = 1;
    int eos_token_id_ = 2;
    int unk_token_id_ = 3;
    
    // Language-specific mappings
    struct LanguageMap {
        std::string espeak_voice;
        std::unordered_map<std::string, std::string> phoneme_map;
    };
    std::unordered_map<std::string, LanguageMap> language_maps_;
    std::string current_language_ = "en-us";
};

// Utility functions

/**
 * Normalize text for phonemization.
 * Handles numbers, abbreviations, etc.
 */
std::string normalize_text(const std::string& text);

/**
 * Split text into sentences for better prosody.
 */
std::vector<std::string> split_sentences(const std::string& text);

} // namespace heartbeat

#endif // PHONEMIZER_H
