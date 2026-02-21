#pragma once
#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <variant>

// Forward declaration for GGML
struct ggml_context;
struct ggml_tensor;

namespace heartbeat {

/**
 * GGUF file magic numbers and constants.
 */
namespace gguf {
    constexpr uint32_t MAGIC = 0x46554747;  // "GGUF" in little-endian
    constexpr uint32_t VERSION_2 = 2;
    constexpr uint32_t VERSION_3 = 3;
    
    enum class ValueType : uint32_t {
        UINT8   = 0,
        INT8    = 1,
        UINT16  = 2,
        INT16   = 3,
        UINT32  = 4,
        INT32   = 5,
        FLOAT32 = 6,
        BOOL    = 7,
        STRING  = 8,
        ARRAY   = 9,
        UINT64  = 10,
        INT64   = 11,
        FLOAT64 = 12,
    };
    
    enum class TensorType : uint32_t {
        F32  = 0,
        F16  = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        BF16 = 30,
    };
}

/**
 * Variant type for GGUF metadata values.
 */
using GGUFValue = std::variant<
    uint8_t, int8_t,
    uint16_t, int16_t,
    uint32_t, int32_t,
    uint64_t, int64_t,
    float, double,
    bool, std::string,
    std::vector<std::string>
>;

/**
 * Tensor metadata from GGUF file.
 */
struct TensorInfo {
    std::string name;
    std::vector<int64_t> dimensions;
    gguf::TensorType type;
    uint64_t offset;        // Offset in data section
    size_t size_bytes;      // Total size in bytes
    ggml_tensor* tensor;    // Loaded GGML tensor (null until load)
};

/**
 * Kokoro model hyperparameters extracted from GGUF.
 */
struct KokoroParams {
    // Text Encoder (PL-BERT)
    int vocab_size = 178;
    int hidden_size = 768;
    int num_layers = 12;
    int num_heads = 12;
    int intermediate_size = 3072;
    int max_position = 512;
    
    // Style
    int style_dim = 256;
    int n_styles = 1;
    
    // Decoder
    int n_mels = 80;
    int decoder_channels = 512;
    int decoder_layers = 8;
    
    // ISTFT
    int istft_n_fft = 16;
    int istft_hop_length = 4;
    
    // Audio
    int sample_rate = 24000;
};

/**
 * GGUF file loader and model container.
 */
class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();
    
    /**
     * Load a GGUF file from disk.
     * @param path Path to the GGUF file.
     * @return true if successful.
     */
    bool load(const std::string& path);
    
    /**
     * Check if model is loaded.
     */
    bool is_loaded() const { return loaded_; }
    
    /**
     * Get model parameters.
     */
    const KokoroParams& params() const { return params_; }
    
    /**
     * Get a metadata value by key.
     * @param key Metadata key.
     * @return Pointer to value or nullptr if not found.
     */
    const GGUFValue* get_metadata(const std::string& key) const;
    
    /**
     * Get a tensor by name.
     * @param name Tensor name.
     * @return Pointer to GGML tensor or nullptr if not found.
     */
    ggml_tensor* get_tensor(const std::string& name) const;
    
    /**
     * Get phoneme vocabulary.
     * @return Vector of phoneme tokens.
     */
    const std::vector<std::string>& vocabulary() const { return vocab_; }
    
    /**
     * Get phoneme to ID mapping.
     */
    int phoneme_to_id(const std::string& phoneme) const;
    
    /**
     * Get GGML context.
     */
    ggml_context* context() { return ctx_; }
    
    /**
     * Get all tensor names.
     */
    std::vector<std::string> tensor_names() const;

private:
    bool parse_header(const uint8_t* data, size_t size);
    bool parse_metadata(const uint8_t* data, size_t& offset);
    bool parse_tensors(const uint8_t* data, size_t& offset, size_t alignment);
    bool load_tensor_data(const uint8_t* data, size_t data_offset);
    void extract_params();
    
    bool loaded_ = false;
    uint32_t version_ = 0;
    
    ggml_context* ctx_ = nullptr;
    std::vector<uint8_t> file_data_;
    
    std::unordered_map<std::string, GGUFValue> metadata_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    
    KokoroParams params_;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> vocab_map_;
};

} // namespace heartbeat

#endif // GGUF_LOADER_H
