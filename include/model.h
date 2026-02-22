#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "gguf_loader.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <array>

// Forward declarations
struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer;

namespace heartbeat {

// ========================================
// ALBERT Layer Weights (Shared across all layers)
// ========================================

struct ALBERTLayer {
    // Full layer layer norm
    ggml_tensor* full_ln_weight = nullptr;
    ggml_tensor* full_ln_bias = nullptr;
    
    // Self-attention
    ggml_tensor* q_weight = nullptr;
    ggml_tensor* q_bias = nullptr;
    ggml_tensor* k_weight = nullptr;
    ggml_tensor* k_bias = nullptr;
    ggml_tensor* v_weight = nullptr;
    ggml_tensor* v_bias = nullptr;
    ggml_tensor* o_weight = nullptr;
    ggml_tensor* o_bias = nullptr;
    
    // Attention layer norm
    ggml_tensor* attn_ln_weight = nullptr;
    ggml_tensor* attn_ln_bias = nullptr;
    
    // FFN
    ggml_tensor* ffn_weight = nullptr;
    ggml_tensor* ffn_bias = nullptr;
    ggml_tensor* ffn_out_weight = nullptr;
    ggml_tensor* ffn_out_bias = nullptr;
};

// ========================================
// Weight-Normalized Convolution
// ========================================

struct WeightNormConv {
    ggml_tensor* weight_g = nullptr;
    ggml_tensor* weight_v = nullptr;
    ggml_tensor* bias = nullptr;
};

// ========================================
// CNN Block (Conv + Normalization)
// ========================================

struct CNNBlock {
    ggml_tensor* conv_weight_g = nullptr;
    ggml_tensor* conv_weight_v = nullptr;
    ggml_tensor* conv_bias = nullptr;
    ggml_tensor* norm_gamma = nullptr;
    ggml_tensor* norm_beta = nullptr;
};

// ========================================
// LSTM Weights
// ========================================

struct LSTMWeights {
    // Forward LSTM
    ggml_tensor* weight_ih = nullptr;
    ggml_tensor* weight_hh = nullptr;
    ggml_tensor* bias_ih = nullptr;
    ggml_tensor* bias_hh = nullptr;
    
    // Reverse LSTM (for BiLSTM)  
    ggml_tensor* weight_ih_r = nullptr;
    ggml_tensor* weight_hh_r = nullptr;
    ggml_tensor* bias_ih_r = nullptr;
    ggml_tensor* bias_hh_r = nullptr;
};

struct LinearWeights {
    ggml_tensor* weight = nullptr;
    ggml_tensor* bias = nullptr;
};

struct LSTMBlock {
    LSTMWeights lstm;
    LinearWeights linear;
};

struct PredictorWeights {
    // Text Encoder: 3 blocks of (BiLSTM + Linear) => 6 layers total
    std::array<LSTMBlock, 3> text_encoder; 
    
    // Shared LSTM for F0/N
    LSTMWeights shared_lstm;
    
    // Duration
    LSTMWeights duration_lstm;
    LinearWeights duration_proj;
    
    // F0 Predictor (3 AdainResBlk1d)
    std::array<AdainResBlk1dWeights, 3> F0;
    LinearWeights F0_proj;
    
    // N Predictor (3 AdainResBlk1d)
    std::array<AdainResBlk1dWeights, 3> N;
    LinearWeights N_proj;
};

// ========================================
// AdaIN (Adaptive Instance Normalization)
// ========================================

struct AdaINWeights {
    ggml_tensor* fc_weight = nullptr;
    ggml_tensor* fc_bias = nullptr;
};

// ========================================
// HiFi-GAN Upsampling Block
// ========================================

struct UpsampleBlock {
    ggml_tensor* weight_g = nullptr;
    ggml_tensor* weight_v = nullptr;
    ggml_tensor* bias = nullptr;
};

// ========================================
// HiFi-GAN Residual Block
// ========================================

struct ResBlock {
    // Each resblock has 3 conv layers in convs1 and convs2
    std::array<WeightNormConv, 3> convs1;
    std::array<WeightNormConv, 3> convs2;
    
    // AdaIN layers (3 for each conv set)
    std::array<AdaINWeights, 3> adain1;
    std::array<AdaINWeights, 3> adain2;
    
    // Alpha parameters for residual scaling (Snake activation)
    std::array<ggml_tensor*, 3> alpha1 = {nullptr, nullptr, nullptr};
    std::array<ggml_tensor*, 3> alpha2 = {nullptr, nullptr, nullptr};
};

// ========================================
// AdainResBlk1d (Used in early decoder stages)
// ========================================

struct AdainResBlk1dWeights {
    WeightNormConv conv1;
    WeightNormConv conv2;
    AdaINWeights norm1;
    AdaINWeights norm2;
    WeightNormConv conv1x1; // shortcut
    WeightNormConv pool;    // for upsampling
};

// ========================================
// Complete Model Weights
// ========================================

struct ModelWeights {
    // ========== BERT Embeddings ==========
    ggml_tensor* word_embeddings = nullptr;
    ggml_tensor* position_embeddings = nullptr;
    ggml_tensor* token_type_embeddings = nullptr;
    ggml_tensor* embed_ln_weight = nullptr;
    ggml_tensor* embed_ln_bias = nullptr;
    
    // ========== ALBERT Projection ==========
    ggml_tensor* bert_proj_weight = nullptr;
    ggml_tensor* bert_proj_bias = nullptr;
    
    // ========== ALBERT Shared Layer ==========
    ALBERTLayer albert_layer;
    
    // ========== Secondary Text Encoder ==========
    // Note: Kokoro uses Predictor's text encoder instead?
    // Or is this separate?
    // Legacy support kept but might be unused if we rely on Predictor
    ggml_tensor* text_enc_embedding = nullptr;
    std::array<CNNBlock, 3> text_enc_cnn;
    LSTMWeights text_enc_lstm;
    
    // Projection from BERT hidden (768) to Predictor input (512)
    ggml_tensor* text_enc_proj_weight = nullptr;
    ggml_tensor* text_enc_proj_bias = nullptr;
    
    // ========== Predictor (Style + Duration + F0 + N) ==========
    PredictorWeights predictor;
    
    // ========== ISTFTNet Decoder Stage ==========
    // Initial encoding/decoding layers
    AdainResBlk1dWeights decoder_encode;
    std::array<AdainResBlk1dWeights, 4> decoder_decode;
    
    // F0 and N conditioning
    WeightNormConv decoder_F0_conv;
    WeightNormConv decoder_N_conv;
    WeightNormConv decoder_asr_res;
    
    // Generator Stage
    struct GeneratorWeights {
        LinearWeights m_source_linear;
        std::array<ggml_tensor*, 2> noise_convs_weight;
        std::array<ggml_tensor*, 2> noise_convs_bias;
        std::array<ResBlock, 2> noise_res;
        std::array<WeightNormConv, 2> ups;
        std::array<ResBlock, 6> resblocks; 
        WeightNormConv conv_post;
    } generator;
};

// ========================================
// Forward Pass Output
// ========================================

struct ModelOutput {
    std::vector<float> magnitude;   // [n_frames, n_fft]
    std::vector<float> phase;       // [n_frames, n_fft]
    int n_mels = 0;
    int n_frames = 0;
};

// ========================================
// Model Class
// ========================================

class Model {
public:
    Model();
    ~Model();
    
    /**
     * Load model weights from GGUF loader.
     * @param loader Initialized GGUF loader.
     * @return true if successful.
     */
    bool load_weights(GGUFLoader& loader);
    
    /**
     * Run forward pass.
     * @param token_ids Input token IDs from phonemizer.
     * @param style_vector Style embedding vector.
     * @return ModelOutput containing magnitude and phase spectrograms.
     */
    ModelOutput forward(const std::vector<int>& token_ids,
                        const std::vector<float>& style_vector);
    
    /**
     * Get style vector for a voice code.
     * @param voice_code Voice identifier (e.g., "af", "in_f").
     * @return Style embedding vector.
     */
    std::vector<float> get_style_vector(const std::string& voice_code);
    
    /**
     * Check if model is ready.
     */
    bool is_ready() const { return loaded_; }
    
    /**
     * Get model parameters.
     */
    const KokoroParams& params() const { return params_; }
    
private:
    bool loaded_ = false;
    KokoroParams params_;
    ModelWeights weights_;
    
    // ALBERT Graph Helpers
    ggml_tensor* build_embeddings(ggml_context* ctx, const std::vector<int>& tokens);
    ggml_tensor* build_attention(ggml_context* ctx, ggml_tensor* hidden, const ALBERTLayer& layer);
    ggml_tensor* build_ffn(ggml_context* ctx, ggml_tensor* hidden, const ALBERTLayer& layer);
    ggml_tensor* build_encoder(ggml_context* ctx, ggml_tensor* hidden);
    
    // Predictor Helpers
    ggml_tensor* build_duration_predictor(ggml_context* ctx, ggml_tensor* hidden, ggml_tensor* style);
    
    // Decoder Helpers
    ggml_tensor* build_decoder(ggml_context* ctx, ggml_tensor* en, ggml_tensor* style);
    ggml_tensor* build_adain(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style, const AdaINWeights& w);
    
    // Reference to loader for tensor access
    GGUFLoader* loader_ = nullptr;
    
    // Compute context
    ggml_context* compute_ctx_ = nullptr;
    
    // Style vectors per voice
    std::unordered_map<std::string, std::vector<float>> voice_styles_;
    
    // Default style vector
    std::vector<float> default_style_;
};

} // namespace heartbeat

#endif // MODEL_H
