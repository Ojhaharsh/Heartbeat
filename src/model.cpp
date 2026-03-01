/**
 * Kokoro-82M Model Implementation
 * 
 * Architecture: ALBERT (Shared-Weight BERT) + HiFi-GAN Decoder
 * 
 * Components:
 * 1. BERT Encoder (ALBERT-style with albert_layer_groups)
 * 2. Text Encoder (CNN + BiLSTM)
 * 3. Duration Encoder + Prediction Encoder
 * 4. HiFi-GAN Decoder with AdaIN
 */

#include "model.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <thread>

extern "C" {
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
}

namespace heartbeat {

// Forward declarations
static ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, 
                               ggml_tensor* weight, ggml_tensor* bias);

static bool heartbeat_verbose() {
    return std::getenv("HEARTBEAT_VERBOSE") != nullptr;
}

static int heartbeat_num_threads() {
    if (const char* env = std::getenv("HEARTBEAT_THREADS")) {
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end != env && parsed > 0) {
            return static_cast<int>(std::clamp(parsed, 1L, 64L));
        }
    }

    const unsigned hw = std::thread::hardware_concurrency();
    return static_cast<int>(std::clamp(hw == 0 ? 4u : hw, 1u, 64u));
}

Model::Model() = default;

Model::~Model() {
    if (compute_ctx_) {
        ggml_free(compute_ctx_);
    }
}

bool Model::load_weights(GGUFLoader& loader) {
    if (!loader.is_loaded()) {
        if (heartbeat_verbose()) {
            std::cerr << "DEBUG: Loader not loaded\n";
        }
        return false;
    }
    
    params_ = loader.params();
    
    const bool verbose = heartbeat_verbose();

    // Print available tensor names for debugging (verbose mode only)
    auto tensor_names = loader.tensor_names();
    if (verbose) {
        std::cerr << "DEBUG: Available tensors: " << tensor_names.size() << "\n";
        for (const auto& n : tensor_names) {
            std::cerr << "TENSOR: " << n << "\n";
        }
    }
    
    // Store reference to loader for direct tensor access during forward pass
    loader_ = &loader;
    
    // Verify manual lookup (removed debug loop)
    // Use identified tensor name: "bert_encoder.weight"
    weights_.text_enc_proj_weight = loader.get_tensor("bert_encoder.weight");
    weights_.text_enc_proj_bias = loader.get_tensor("bert_encoder.bias");
    
    if (verbose) {
        if (weights_.text_enc_proj_weight) std::cerr << "DEBUG: SUCCESS loading text_enc_proj (bert_encoder)\n";
        else std::cerr << "DEBUG: FAILED loading text_enc_proj (bert_encoder)\n";
    }
    
    // Helper to get tensor with fallback name patterns
    
    // Helper to get tensor with fallback name patterns
    auto get_tensor = [&loader](const std::vector<std::string>& names) -> ggml_tensor* {
        for (const auto& name : names) {
            auto* t = loader.get_tensor(name);
            if (t) {
                return t;
            }
        }
        return nullptr;
    };
    
    // Helper for LSTM weights
    auto get_lstm = [&](const std::string& prefix) -> LSTMWeights {
        LSTMWeights w;
        w.weight_ih = get_tensor({prefix + ".weight_ih_l0"});
        w.weight_hh = get_tensor({prefix + ".weight_hh_l0"});
        w.bias_ih = get_tensor({prefix + ".bias_ih_l0"});
        w.bias_hh = get_tensor({prefix + ".bias_hh_l0"});
        w.weight_ih_r = get_tensor({prefix + ".weight_ih_l0_reverse"});
        w.weight_hh_r = get_tensor({prefix + ".weight_hh_l0_reverse"});
        w.bias_ih_r = get_tensor({prefix + ".bias_ih_l0_reverse"});
        w.bias_hh_r = get_tensor({prefix + ".bias_hh_l0_reverse"});
        return w;
    };

    // Helper for Linear weights
    auto get_linear = [&](const std::string& prefix) -> LinearWeights {
        LinearWeights w;
        w.weight = get_tensor({prefix + ".weight", prefix + ".linear_layer.weight"});
        w.bias = get_tensor({prefix + ".bias", prefix + ".linear_layer.bias"});
        return w;
    };
    
    // Helper for AdaIN weights
    auto get_adain = [&](const std::string& prefix) -> AdaINWeights {
        AdaINWeights w;
        w.fc_weight = get_tensor({prefix + ".fc.weight"});
        w.fc_bias = get_tensor({prefix + ".fc.bias"});
        return w;
    };

    // Helper for AdainResBlk1d weights
    auto get_adain_res_blk = [&](const std::string& prefix) -> AdainResBlk1dWeights {
        AdainResBlk1dWeights w;
        w.conv1.weight_g = get_tensor({prefix + ".conv1.weight_g"});
        w.conv1.weight_v = get_tensor({prefix + ".conv1.weight_v"});
        w.conv1.bias = get_tensor({prefix + ".conv1.bias"});
        w.conv2.weight_g = get_tensor({prefix + ".conv2.weight_g"});
        w.conv2.weight_v = get_tensor({prefix + ".conv2.weight_v"});
        w.conv2.bias = get_tensor({prefix + ".conv2.bias"});
        w.norm1 = get_adain(prefix + ".norm1");
        w.norm2 = get_adain(prefix + ".norm2");
        w.conv1x1.weight_g = get_tensor({prefix + ".conv1x1.weight_g"});
        w.conv1x1.weight_v = get_tensor({prefix + ".conv1x1.weight_v"});
        w.pool.weight_g = get_tensor({prefix + ".pool.weight_g"});
        w.pool.weight_v = get_tensor({prefix + ".pool.weight_v"});
        w.pool.bias = get_tensor({prefix + ".pool.bias"});
        return w;
    };

    // Helper for ResBlock weights
    auto get_res_block = [&](const std::string& prefix) -> ResBlock {
        ResBlock w;
        for (int j = 0; j < 3; j++) {
            std::string c1 = prefix + ".convs1." + std::to_string(j);
            std::string c2 = prefix + ".convs2." + std::to_string(j);
            w.convs1[j].weight_g = get_tensor({c1 + ".weight_g"});
            w.convs1[j].weight_v = get_tensor({c1 + ".weight_v"});
            w.convs1[j].bias = get_tensor({c1 + ".bias"});
            w.convs2[j].weight_g = get_tensor({c2 + ".weight_g"});
            w.convs2[j].weight_v = get_tensor({c2 + ".weight_v"});
            w.convs2[j].bias = get_tensor({c2 + ".bias"});
            w.adain1[j] = get_adain(prefix + ".adain1." + std::to_string(j));
            w.adain2[j] = get_adain(prefix + ".adain2." + std::to_string(j));
            w.alpha1[j] = get_tensor({prefix + ".alpha1." + std::to_string(j)});
            w.alpha2[j] = get_tensor({prefix + ".alpha2." + std::to_string(j)});
        }
        return w;
    };
    
    // Count loaded tensors
    int loaded_count = 0;
    
    // ========================================
    // BERT Text Encoder (ALBERT-style)
    // ========================================
    
    // BERT Embeddings (shared for    // word_embeddings
    weights_.word_embeddings = get_tensor({
        "text_encoder.embedding.weight", 
        "bert.embeddings.word_embeddings.weight"
    });
    
    // Debug discovery for text encoder architecture (verbose mode only)
    if (verbose) {
        std::cerr << "--- TENSOR DISCOVERY ---\n";
        std::vector<std::string> names = loader.tensor_names();
        for (const auto& name : names) {
            ggml_tensor* t = loader.get_tensor(name);
            if (name.find("text_encoder") != std::string::npos || 
                name.find("bert") != std::string::npos ||
                name.find("predictor") != std::string::npos ||
                name.find("decoder") != std::string::npos) {
                std::cerr << "Found: " << name << " [" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "] type=" << t->type << "\n";
            }
        }
        std::cerr << "------------------------\n";
    }

    if (weights_.word_embeddings) {
        if (verbose) {
            std::cerr << "DEBUG: Word Embeddings loaded. Dim: " << weights_.word_embeddings->ne[0] << " x " << weights_.word_embeddings->ne[1] << "\n";
        }
        loaded_count++;
    }
    
    weights_.position_embeddings = get_tensor({
        "text_encoder.pos_emb.weight", 
        "bert.embeddings.position_embeddings.weight"
    });
    if (weights_.position_embeddings) loaded_count++;
    
    weights_.token_type_embeddings = get_tensor({
        "text_encoder.token_type_emb.weight",
        "bert.embeddings.token_type_embeddings.weight"
    });
    if (weights_.token_type_embeddings) loaded_count++;
    
    weights_.embed_ln_weight = get_tensor({
        "text_encoder.embed_ln.weight",
        "bert.embeddings.LayerNorm.weight"
    });
    if (weights_.embed_ln_weight) loaded_count++;
    
    weights_.embed_ln_bias = get_tensor({
        "text_encoder.embed_ln.bias",
        "bert.embeddings.LayerNorm.bias"
    });
    if (weights_.embed_ln_bias) loaded_count++;
    
    // ALBERT: Hidden size mapping (embedding_hidden_mapping_in)
    weights_.bert_proj_weight = get_tensor({
        "bert.encoder.embedding_hidden_mapping_in.weight"
    });
    if (weights_.bert_proj_weight) loaded_count++;
    
    weights_.bert_proj_bias = get_tensor({
        "bert.encoder.embedding_hidden_mapping_in.bias"
    });
    if (weights_.bert_proj_bias) loaded_count++;
    
    // ALBERT: Shared layer weights (albert_layer_groups.0.albert_layers.0)
    // ALBERT shares weights across all 12 layers using a single set
    std::string albert_prefix = "bert.encoder.albert_layer_groups.0.albert_layers.0";
    
    // Shared layer - full layer norm
    weights_.albert_layer.full_ln_weight = get_tensor({albert_prefix + ".full_layer_layer_norm.weight"});
    weights_.albert_layer.full_ln_bias = get_tensor({albert_prefix + ".full_layer_layer_norm.bias"});
    
    // Shared layer - attention
    weights_.albert_layer.q_weight = get_tensor({albert_prefix + ".attention.query.weight"});
    weights_.albert_layer.q_bias = get_tensor({albert_prefix + ".attention.query.bias"});
    weights_.albert_layer.k_weight = get_tensor({albert_prefix + ".attention.key.weight"});
    weights_.albert_layer.k_bias = get_tensor({albert_prefix + ".attention.key.bias"});
    weights_.albert_layer.v_weight = get_tensor({albert_prefix + ".attention.value.weight"});
    weights_.albert_layer.v_bias = get_tensor({albert_prefix + ".attention.value.bias"});
    weights_.albert_layer.o_weight = get_tensor({albert_prefix + ".attention.dense.weight"});
    weights_.albert_layer.o_bias = get_tensor({albert_prefix + ".attention.dense.bias"});
    weights_.albert_layer.attn_ln_weight = get_tensor({albert_prefix + ".attention.LayerNorm.weight"});
    weights_.albert_layer.attn_ln_bias = get_tensor({albert_prefix + ".attention.LayerNorm.bias"});
    
    // Shared layer - FFN
    weights_.albert_layer.ffn_weight = get_tensor({albert_prefix + ".ffn.weight"});
    weights_.albert_layer.ffn_bias = get_tensor({albert_prefix + ".ffn.bias"});
    weights_.albert_layer.ffn_out_weight = get_tensor({albert_prefix + ".ffn_output.weight"});
    weights_.albert_layer.ffn_out_bias = get_tensor({albert_prefix + ".ffn_output.bias"});
    
    if (weights_.albert_layer.q_weight) {
        loaded_count += 14; // Count all albert layer weights
    }
    
    // ========================================
    // Secondary Text Encoder (CNN + LSTM)
    // ========================================
    
    weights_.text_enc_embedding = get_tensor({"text_encoder.embedding.weight"});
    if (weights_.text_enc_embedding) {
        if (verbose) {
            std::cerr << "DEBUG: Secondary Text Encoder Embedding loaded.\n";
        }
        loaded_count++;
    }
    
    // CNN layers (3 conv blocks)
    for (int i = 0; i < 3; i++) {
        std::string cnn_prefix = "text_encoder.cnn." + std::to_string(i);
        weights_.text_enc_cnn[i].conv_weight_g = get_tensor({cnn_prefix + ".0.weight_g"});
        weights_.text_enc_cnn[i].conv_weight_v = get_tensor({cnn_prefix + ".0.weight_v"});
        weights_.text_enc_cnn[i].conv_bias = get_tensor({cnn_prefix + ".0.bias"});
        
        // Norm layers are 0.1 for indices 0 and 2, but what about 1?
        // Let's check discovery again: 
        // Found: text_encoder.cnn.0.1.gamma
        // Found: text_encoder.cnn.1.1.gamma
        // Found: text_encoder.cnn.2.1.gamma
        weights_.text_enc_cnn[i].norm_gamma = get_tensor({cnn_prefix + ".1.gamma"});
        weights_.text_enc_cnn[i].norm_beta = get_tensor({cnn_prefix + ".1.beta"});
        
        if (weights_.text_enc_cnn[i].conv_weight_v) {
            if (verbose) {
                std::cerr << "DEBUG: Loaded text_encoder.cnn." << i << "\n";
            }
            loaded_count += 5;
        }
    }
    
    // LSTM weights (Secondary Encoder)
    weights_.text_enc_lstm = get_lstm("text_encoder.lstm");
    if (weights_.text_enc_lstm.weight_ih) {
        if (verbose) {
            std::cerr << "DEBUG: Loaded text_encoder.lstm\n";
        }
        loaded_count += 8;
    }
    
    // Logic moved to top
    if (verbose && weights_.text_enc_proj_weight) std::cerr << "DEBUG: Verified text_enc_proj is loaded.\n";
    
    // ========================================
    // Predictor (Style / Duration / F0 / N)
    // ========================================
    
    // Text Encoder LSTMs (6 layers)
    for (int i = 0; i < 3; i++) {
        std::string lstm_prefix = "predictor.text_encoder.lstms." + std::to_string(2*i);
        weights_.predictor.text_encoder[i].lstm = get_lstm(lstm_prefix);
        
        std::string linear_prefix = "predictor.text_encoder.lstms." + std::to_string(2*i+1);
        weights_.predictor.text_encoder[i].linear = get_linear(linear_prefix);
    }
    
    // Shared LSTM for F0/N
    weights_.predictor.shared_lstm = get_lstm("predictor.shared");
    
    // Duration LSTM
    weights_.predictor.duration_lstm = get_lstm("predictor.lstm");
    
    // Duration Projection
    weights_.predictor.duration_proj = get_linear("predictor.duration_proj");

    // F0 Predictor (3 AdainResBlk1d)
    for (int i = 0; i < 3; i++) {
        weights_.predictor.F0[i] = get_adain_res_blk("predictor.F0." + std::to_string(i));
    }
    weights_.predictor.F0_proj = get_linear("predictor.F0_proj");

    // N Predictor (3 AdainResBlk1d)
    for (int i = 0; i < 3; i++) {
        weights_.predictor.N[i] = get_adain_res_blk("predictor.N." + std::to_string(i));
    }
    weights_.predictor.N_proj = get_linear("predictor.N_proj");

    
    // ========================================
    // ISTFTNet Decoder Stage
    // ========================================
    
    // Initial encoding/decoding layers
    weights_.decoder_encode = get_adain_res_blk("decoder.encode");
    for (int i = 0; i < 4; i++) {
        weights_.decoder_decode[i] = get_adain_res_blk("decoder.decode." + std::to_string(i));
    }
    
    // F0 and N conditioning
    weights_.decoder_F0_conv.weight_g = get_tensor({"decoder.F0_conv.weight_g"});
    weights_.decoder_F0_conv.weight_v = get_tensor({"decoder.F0_conv.weight_v"});
    weights_.decoder_F0_conv.bias = get_tensor({"decoder.F0_conv.bias"});
    
    weights_.decoder_N_conv.weight_g = get_tensor({"decoder.N_conv.weight_g"});
    weights_.decoder_N_conv.weight_v = get_tensor({"decoder.N_conv.weight_v"});
    weights_.decoder_N_conv.bias = get_tensor({"decoder.N_conv.bias"});
    
    weights_.decoder_asr_res.weight_g = get_tensor({"decoder.asr_res.0.weight_g"});
    weights_.decoder_asr_res.weight_v = get_tensor({"decoder.asr_res.0.weight_v"});
    weights_.decoder_asr_res.bias = get_tensor({"decoder.asr_res.0.bias"});
    
    // Generator Stage
    weights_.generator.m_source_linear = get_linear("decoder.generator.m_source.l_linear");
    for (int i = 0; i < 2; i++) {
        weights_.generator.noise_convs_weight[i] = get_tensor({"decoder.generator.noise_convs." + std::to_string(i) + ".weight"});
        weights_.generator.noise_convs_bias[i] = get_tensor({"decoder.generator.noise_convs." + std::to_string(i) + ".bias"});
        weights_.generator.noise_res[i] = get_res_block("decoder.generator.noise_res." + std::to_string(i));
        weights_.generator.ups[i].weight_g = get_tensor({"decoder.generator.ups." + std::to_string(i) + ".weight_g"});
        weights_.generator.ups[i].weight_v = get_tensor({"decoder.generator.ups." + std::to_string(i) + ".weight_v"});
        weights_.generator.ups[i].bias = get_tensor({"decoder.generator.ups." + std::to_string(i) + ".bias"});
    }
    
    for (int i = 0; i < 6; i++) {
        weights_.generator.resblocks[i] = get_res_block("decoder.generator.resblocks." + std::to_string(i));
    }
    
    weights_.generator.conv_post.weight_g = get_tensor({"decoder.generator.conv_post.weight_g"});
    weights_.generator.conv_post.weight_v = get_tensor({"decoder.generator.conv_post.weight_v"});
    weights_.generator.conv_post.bias = get_tensor({"decoder.generator.conv_post.bias"});
    
    // ========================================
    // Voice Style Vectors (loaded separately)
    // ========================================
    
    // ========================================
    // Voice Style Vectors (loaded separately)
    // ========================================
    
    // Default style vector (use model style_dim)
    const size_t default_style_dim = params_.style_dim > 0 ? static_cast<size_t>(params_.style_dim) : 128;
    default_style_.resize(default_style_dim, 0.0f);
    for (size_t i = 0; i < default_style_dim; i++) {
        default_style_[i] = (i % 2 == 0) ? 0.1f : -0.1f;
    }
    
    if (verbose) {
        std::cerr << "DEBUG: Loaded " << loaded_count << " weight tensors\n";
    }
    
    loaded_ = true;
    return true;
}

std::vector<float> Model::get_style_vector(const std::string& voice_code) {
    auto it = voice_styles_.find(voice_code);
    if (it != voice_styles_.end()) {
        return it->second;
    }
    
    // Return default style vector
    return default_style_;
}

// ========================================
// GGML Graph Building Helpers
// ========================================

static ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, 
                               ggml_tensor* weight, ggml_tensor* bias) {
    // Layer normalization: (x - mean) / sqrt(var + eps) * weight + bias
    // ggml_norm returns (x - mean) / sqrt(var + eps)
    x = ggml_norm(ctx, x, 1e-5f);
    if (weight) {
        x = ggml_mul(ctx, x, weight);
    }
    if (bias) {
        x = ggml_add(ctx, x, bias);
    }
    return x;
}

static ggml_tensor* apply_weight_norm(ggml_context* ctx, ggml_tensor* g, ggml_tensor* v) {
    if (!g || !v) return v;
    
    // v: [K, OC, IC] for trans-conv, [K, IC, OC] for conv
    // g: [OC] usually. But for Kokoro's trans-convs, g appears to be [IC].
    
    int n_dims = ggml_n_dims(v);
    int g_dim = (int)g->ne[0];
    int oc = (int)v->ne[n_dims - 1]; // Last dim is OC for normal conv [K, IC, OC]
    int ic = (int)v->ne[n_dims - 2]; // Second last is IC
    
    // Special case for TransposeConv where weight-norm matches IC (dim 0 in PyTorch)
    // discovery: ups.0.v=[20, 256, 512], ups.0.g=[512]. So ic=512 matches g.
    bool scale_ic = (g_dim == v->ne[2] && v->ne[2] != v->ne[1]);
    
    ggml_tensor* v_sq = ggml_sqr(ctx, v);
    ggml_tensor* v_norm;
    
    if (scale_ic) {
        // Norm over [K, OC] to scale each IC
        // ne0=K, ne1=OC, ne2=IC. Reshape to [K*OC, IC]
        v_norm = ggml_sum_rows(ctx, ggml_reshape_2d(ctx, v_sq, v->ne[0] * v->ne[1], v->ne[2]));
        v_norm = ggml_sqrt(ctx, v_norm);
        
        // Scale v by g / v_norm along ne2
        // reshape g and v_norm to match [1, 1, IC] for repeat
        ggml_tensor* g_3d = ggml_reshape_3d(ctx, g, 1, 1, v->ne[2]);
        ggml_tensor* vn_3d = ggml_reshape_3d(ctx, v_norm, 1, 1, v->ne[2]);
        
        ggml_tensor* scale = ggml_div(ctx, g_3d, vn_3d);
        return ggml_mul(ctx, v, ggml_repeat(ctx, scale, v));
    } else {
        // Standard OC scaling (ne[v->n_dims-1])
        // Reshape to [K*IC, OC]
        v_norm = ggml_sum_rows(ctx, ggml_reshape_2d(ctx, v_sq, v->ne[0] * v->ne[1], v->ne[2]));
        v_norm = ggml_sqrt(ctx, v_norm);
        
        // Scale v by g / v_norm
        ggml_tensor* g_3d = ggml_reshape_3d(ctx, g, 1, 1, v->ne[2]);
        ggml_tensor* vn_3d = ggml_reshape_3d(ctx, v_norm, 1, 1, v->ne[2]);
        
        ggml_tensor* scale = ggml_div(ctx, g_3d, vn_3d);
        return ggml_mul(ctx, v, ggml_repeat(ctx, scale, v));
    }
}

static ggml_tensor* weight_norm_conv1d(ggml_context* ctx,
                                        ggml_tensor* x,
                                        ggml_tensor* w_g,
                                        ggml_tensor* w_v,
                                        ggml_tensor* bias,
                                        int stride = 1,
                                        int padding = 1,
                                        int dilation = 1) {
    if (!x || !w_v) return x;
    
    ggml_tensor* w = apply_weight_norm(ctx, w_g, w_v);
    
    // Ensure contiguous and cast to F16
    ggml_tensor* w_cont = ggml_cont(ctx, w);
    ggml_tensor* w_f16 = ggml_cast(ctx, w_cont, GGML_TYPE_F16);
    
    ggml_tensor* h = ggml_conv_1d(ctx, w_f16, x, stride, padding, dilation);
    
    if (bias) {
        h = ggml_add(ctx, h, ggml_repeat(ctx, ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1), h));
    }
    
    return h;
}

static ggml_tensor* weight_norm_conv_transpose_1d(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* w_g,
                                                ggml_tensor* w_v,
                                                ggml_tensor* bias,
                                                int stride) {
    if (!x || !w_v) return x;
    
    ggml_tensor* w = apply_weight_norm(ctx, w_g, w_v);
    
    // Ensure weights are contiguous
    ggml_tensor* w_cont = ggml_cont(ctx, w);
    ggml_tensor* w_f16 = ggml_cast(ctx, w_cont, GGML_TYPE_F16);
    
    // ggml_conv_transpose_1d: kernel [K, OC, IC], data [L, IC, N]
    ggml_tensor* h = ggml_conv_transpose_1d(ctx, w_f16, x, stride, 0, 1);
    
    if (bias) {
        h = ggml_add(ctx, h, ggml_repeat(ctx, ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1), h));
    }
    
    return h;
}


static ggml_tensor* build_snake(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    if (!alpha) return ggml_leaky_relu(ctx, x, 0.1, true); // Fallback
    
    // alpha is [1, C, 1]
    // result = x + (1/alpha) * sin^2(alpha * x)
    ggml_tensor* alpha_rep = ggml_repeat(ctx, alpha, x);
    ggml_tensor* ax = ggml_mul(ctx, x, alpha_rep);
    ggml_tensor* sin_ax = ggml_sin(ctx, ax);
    ggml_tensor* sin2_ax = ggml_sqr(ctx, sin_ax);
    
    // 1/alpha
    ggml_tensor* inv_alpha = ggml_div(ctx, ggml_new_f32(ctx, 1.0f), alpha);
    ggml_tensor* inv_alpha_rep = ggml_repeat(ctx, inv_alpha, x);
    
    ggml_tensor* term2 = ggml_mul(ctx, sin2_ax, inv_alpha_rep);
    return ggml_add(ctx, x, term2);
}

static ggml_tensor* build_adain(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style, const AdaINWeights& w) {
    if (!x || !style || !w.fc_weight) return x;
    
    // 1. Instance Norm
    // x is [L, C, N]
    // ggml_norm does (x-mean)/std globally or per row? 
    // For InstanceNorm1d, we need mean/std per channel and per batch.
    // In GGML, x [L, C, N] -> reshape to [L, C*N]? No.
    // Let's use ggml_norm(ctx, x, 1e-5f) which is usually sufficient if dimensions are aligned.
    // Actually, we need to be careful. Let's follow StyleTTS2 implementation.
    
    ggml_tensor* normalized = ggml_norm(ctx, x, 1e-5f);
    
    // 2. AdaIN scale/bias from style
    // h = fc(s) -> [2*C]
    ggml_tensor* h = ggml_mul_mat(ctx, w.fc_weight, style);
    if (w.fc_bias) h = ggml_add(ctx, h, w.fc_bias);
    
    // Split h into gamma and beta
    int channels = (int)x->ne[1];
    if (h->ne[0] < 2 * channels) {
        return x;
    }
    ggml_tensor* gamma = ggml_view_1d(ctx, h, channels, 0);
    ggml_tensor* beta = ggml_view_1d(ctx, h, channels, channels * sizeof(float));
    
    // Reshape to [1, C, 1] for broadcasting
    ggml_tensor* gamma_3d = ggml_reshape_3d(ctx, gamma, 1, channels, 1);
    ggml_tensor* beta_3d = ggml_reshape_3d(ctx, beta, 1, channels, 1);
    
    // result = (1 + gamma) * norm(x) + beta
    ggml_tensor* one = ggml_new_f32(ctx, 1.0f);
    ggml_tensor* one_plus_gamma = ggml_add(ctx, ggml_repeat(ctx, one, gamma_3d), gamma_3d);
    
    ggml_tensor* out = ggml_mul(ctx, normalized, ggml_repeat(ctx, one_plus_gamma, normalized));
    out = ggml_add(ctx, out, ggml_repeat(ctx, beta_3d, out));
    
    return out;
}

static ggml_tensor* build_adainresblk1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style, const AdainResBlk1dWeights& w, bool upsample) {
    // shortcut
    ggml_tensor* shortcut = x;
    if (upsample) {
        // Nearest neighbor upsample (scale factor 2)
        // Inistftnet.py: F.interpolate(x, scale_factor=2, mode='nearest')
        // GGML doesn't have interpolate. We can use repeat or manual expansion.
        // For simplicity, let's use a 1D trans-conv with stride 2 if pool exists.
        if (w.pool.weight_v) {
            shortcut = weight_norm_conv_transpose_1d(ctx, shortcut, w.pool.weight_g, w.pool.weight_v, w.pool.bias, 2);
        }
    }
    if (w.conv1x1.weight_v) {
        shortcut = weight_norm_conv1d(ctx, shortcut, w.conv1x1.weight_g, w.conv1x1.weight_v, nullptr, 1, 0, 1);
    }
    
    // residual branch
    ggml_tensor* res = build_adain(ctx, x, style, w.norm1);
    res = ggml_leaky_relu(ctx, res, 0.2, true);
    
    if (upsample && w.pool.weight_v) {
        res = weight_norm_conv_transpose_1d(ctx, res, w.pool.weight_g, w.pool.weight_v, w.pool.bias, 2);
    }
    
    res = weight_norm_conv1d(ctx, res, w.conv1.weight_g, w.conv1.weight_v, w.conv1.bias, 1, 1, 1);
    res = build_adain(ctx, res, style, w.norm2);
    res = ggml_leaky_relu(ctx, res, 0.2, true);
    res = weight_norm_conv1d(ctx, res, w.conv2.weight_g, w.conv2.weight_v, w.conv2.bias, 1, 1, 1);
    
    // add shortcut and scale by rsqrt(2)
    ggml_tensor* out = ggml_add(ctx, res, shortcut);
    out = ggml_scale(ctx, out, 1.0f / sqrtf(2.0f));
    return out;
}

static ggml_tensor* build_resbreak_istftnet(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style, const ResBlock& w) {
    // Three stages of (AdaIN -> Snake -> Conv)
    for (int j = 0; j < 3; j++) {
        ggml_tensor* xt = build_adain(ctx, x, style, w.adain1[j]);
        xt = build_snake(ctx, xt, w.alpha1[j]);
        xt = weight_norm_conv1d(ctx, xt, w.convs1[j].weight_g, w.convs1[j].weight_v, w.convs1[j].bias, 1, (int)(w.convs1[j].weight_v->ne[0]-1)/2, 1); // Auto padding?
        
        xt = build_adain(ctx, xt, style, w.adain2[j]);
        xt = build_snake(ctx, xt, w.alpha2[j]);
        xt = weight_norm_conv1d(ctx, xt, w.convs2[j].weight_g, w.convs2[j].weight_v, w.convs2[j].bias, 1, (int)(w.convs2[j].weight_v->ne[0]-1)/2, 1);
        
        x = ggml_add(ctx, x, xt);
    }
    return x;
}

static ggml_tensor* linear_forward(ggml_context* ctx, ggml_tensor* x, const LinearWeights& w) {
    if (!x || !w.weight) {
        return x;
    }

    ggml_tensor* y = ggml_mul_mat(ctx, w.weight, x);
    if (w.bias) {
        y = ggml_add(ctx, y, ggml_repeat(ctx, w.bias, y));
    }
    return y;
}

// ========================================
// Graph Building Functions
// ========================================

ggml_tensor* Model::build_embeddings(ggml_context* ctx, 
                                      const std::vector<int>& tokens) {
    int seq_len = static_cast<int>(tokens.size());
    if (heartbeat_verbose()) {
        std::cerr << "DEBUG: build_embeddings seq_len=" << seq_len << "\n";
    }
    
    // 1. Word Embeddings
    ggml_tensor* token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    std::memcpy(token_ids->data, tokens.data(), seq_len * sizeof(int));
    
    // if (!weights_.word_embeddings) std::cerr << "ERROR: word_embeddings is NULL\n";
    // std::cerr << "DEBUG: get_rows word_embeddings\n";
    ggml_tensor* hidden = ggml_get_rows(ctx, weights_.word_embeddings, token_ids);
    
    // 2. Position Embeddings (if available)
    if (weights_.position_embeddings) {
        ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
        int* pos_data = (int*)pos_ids->data;
        for (int i = 0; i < seq_len; i++) pos_data[i] = i;
        
        ggml_tensor* pos_emb = ggml_get_rows(ctx, weights_.position_embeddings, pos_ids);
        hidden = ggml_add(ctx, hidden, pos_emb);
    }
    
    // 3. Token Type Embeddings (if available) - usually 0 for single sentence
    if (weights_.token_type_embeddings) {
        // Just use type 0 for all
        ggml_tensor* type_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
        ggml_set_zero(type_ids);
        
        ggml_tensor* type_emb = ggml_get_rows(ctx, weights_.token_type_embeddings, type_ids);
        hidden = ggml_add(ctx, hidden, type_emb);
    }
    
    // 4. Layer Norm
    if (weights_.embed_ln_weight && weights_.embed_ln_bias) {
        hidden = layer_norm(ctx, hidden, weights_.embed_ln_weight, weights_.embed_ln_bias);
    }
    
    return hidden;
}

ggml_tensor* Model::build_attention(ggml_context* ctx,
                                     ggml_tensor* hidden,
                                     const ALBERTLayer& layer) {
    // Hidden: [seq, hidden] (768)
    
    // 1. Projections
    // ggml_mul_mat(W, x) -> W * x^T.
    // W: [768, 768]. x: [seq, 768] (physically 768 contiguous).
    // result: [768, seq].
    
    ggml_tensor* Q = ggml_mul_mat(ctx, layer.q_weight, hidden);
    if (layer.q_bias) Q = ggml_add(ctx, Q, layer.q_bias);
    
    ggml_tensor* K = ggml_mul_mat(ctx, layer.k_weight, hidden);
    if (layer.k_bias) K = ggml_add(ctx, K, layer.k_bias);
    
    ggml_tensor* V = ggml_mul_mat(ctx, layer.v_weight, hidden);
    if (layer.v_bias) V = ggml_add(ctx, V, layer.v_bias);
    
    ggml_tensor* scores = ggml_mul_mat(ctx, K, Q);
    
    // DEBUG: Check scores
    // scores = ggml_check_nan(ctx, scores); // Not available in all GGML versions?
    
    float scale = 1.0f / sqrtf((float)768 / 12.0f);
    scores = ggml_scale(ctx, scores, scale);
    // Softmax
    scores = ggml_soft_max(ctx, scores);
    
    ggml_tensor* V_trans = ggml_transpose(ctx, V);
    // Explicitly make contiguous to avoid GGML assertion failure in mul_mat
    ggml_tensor* V_trans_cont = ggml_cont(ctx, V_trans);
    
    ggml_tensor* context = ggml_mul_mat(ctx, V_trans_cont, scores);
    
    ggml_tensor* out = ggml_mul_mat(ctx, layer.o_weight, context);
    if (layer.o_bias) out = ggml_add(ctx, out, layer.o_bias);
    
    // Add NaN check for output
    // float* d = (float*)out->data; // Cannot check during graph building
    
    out = ggml_add(ctx, hidden, out);
    out = layer_norm(ctx, out, layer.attn_ln_weight, layer.attn_ln_bias);
    
    return out;
}

ggml_tensor* Model::build_ffn(ggml_context* ctx,
                               ggml_tensor* hidden,
                               const ALBERTLayer& layer) {
    ggml_tensor* inter = ggml_mul_mat(ctx, layer.ffn_weight, hidden);
    if (layer.ffn_bias) inter = ggml_add(ctx, inter, layer.ffn_bias);
    
    inter = ggml_gelu(ctx, inter);
    
    ggml_tensor* out = ggml_mul_mat(ctx, layer.ffn_out_weight, inter);
    if (layer.ffn_out_bias) out = ggml_add(ctx, out, layer.ffn_out_bias);
    
    out = ggml_add(ctx, hidden, out);
    out = layer_norm(ctx, out, layer.full_ln_weight, layer.full_ln_bias);
    
    return out;
}

static ggml_tensor* lstm_forward(ggml_context* ctx, 
                                 ggml_tensor* x, 
                                 const LSTMWeights& w) {
    if (!x || !w.weight_ih || !w.weight_hh) return x;
    
    // x: [InDim, SeqLen]
    // w.weight_ih: [InDim, 4*HDim]
    // w.weight_hh: [HDim, 4*HDim]
    
    int h_dim = (int)w.weight_hh->ne[0]; // Hidden dimension (e.g. 256)
    
    // Standard LSTM cell logic (Simplified for graph performance)
    ggml_tensor* gates = ggml_mul_mat(ctx, w.weight_ih, x);
    if (w.bias_ih) gates = ggml_add(ctx, gates, w.bias_ih);
    
    // Split gates into i, f, g, o using h_dim
    ggml_tensor* i = ggml_view_2d(ctx, gates, h_dim, x->ne[1], gates->nb[1], 0);
    ggml_tensor* f = ggml_view_2d(ctx, gates, h_dim, x->ne[1], gates->nb[1], h_dim * sizeof(float));
    ggml_tensor* g = ggml_view_2d(ctx, gates, h_dim, x->ne[1], gates->nb[1], 2 * h_dim * sizeof(float));
    ggml_tensor* o = ggml_view_2d(ctx, gates, h_dim, x->ne[1], gates->nb[1], 3 * h_dim * sizeof(float));
    
    i = ggml_sigmoid(ctx, i);
    f = ggml_sigmoid(ctx, f);
    g = ggml_tanh(ctx, g);
    o = ggml_sigmoid(ctx, o);
    
    // c = f*c_prev + i*g. Assume c_prev = 0.
    ggml_tensor* c = ggml_mul(ctx, i, g);
    ggml_tensor* h = ggml_mul(ctx, o, ggml_tanh(ctx, c));
    
    if (w.weight_ih_r) {
        ggml_tensor* gates_r = ggml_mul_mat(ctx, w.weight_ih_r, x);
        if (w.bias_ih_r) gates_r = ggml_add(ctx, gates_r, w.bias_ih_r);
        
        ggml_tensor* i_r = ggml_view_2d(ctx, gates_r, h_dim, x->ne[1], gates_r->nb[1], 0);
        ggml_tensor* g_r = ggml_view_2d(ctx, gates_r, h_dim, x->ne[1], gates_r->nb[1], 2 * h_dim * sizeof(float));
        ggml_tensor* o_r = ggml_view_2d(ctx, gates_r, h_dim, x->ne[1], gates_r->nb[1], 3 * h_dim * sizeof(float));
        
        ggml_tensor* h_r = ggml_mul(ctx, ggml_sigmoid(ctx, o_r), ggml_tanh(ctx, ggml_mul(ctx, ggml_sigmoid(ctx, i_r), ggml_tanh(ctx, g_r))));
        
        // Concat forward and reverse [2*h_dim, SeqLen]
        // This will be [512, SeqLen] if h_dim=256
        return ggml_concat(ctx, h, h_r, 0);
    }
    
    return h;
}

static ggml_tensor* build_cnn_lstm_encoder(ggml_context* ctx, 
                                           ggml_tensor* hidden, 
                                           const ModelWeights& weights) {
    // hidden: [Dim, SeqLen] (usually [512, Seq])
    
    // GGML conv_1d expects [Length, IC, N]. Transpose and make contiguous.
    hidden = ggml_cont(ctx, ggml_transpose(ctx, hidden));
    
    // 1. CNN Layers
    for (int i = 0; i < 3; i++) {
        const auto& block = weights.text_enc_cnn[i];
        if (!block.conv_weight_v) break;
        
        // padding = kernel_size // 2 (kernel is 5)
        hidden = weight_norm_conv1d(ctx, hidden, block.conv_weight_g, block.conv_weight_v, block.conv_bias, 1, 2, 1);
        
        // Normalization (InstanceNorm or LayerNorm)
        if (block.norm_gamma) {
            // Norm over channels (ne1 in current [L, C] layout)
            // Transpose back to [C, L] and make contiguous for norm
            hidden = ggml_cont(ctx, ggml_transpose(ctx, hidden)); 
            hidden = layer_norm(ctx, hidden, block.norm_gamma, block.norm_beta);
            // Transpose back to [L, C] and make contiguous for next conv
            hidden = ggml_cont(ctx, ggml_transpose(ctx, hidden)); 
        }
        
        hidden = ggml_gelu(ctx, hidden);
    }
    
    // Transpose back for LSTM: [SeqLen, Dim] -> [Dim, SeqLen]
    hidden = ggml_cont(ctx, ggml_transpose(ctx, hidden));
    
    // 2. LSTM layer
    if (weights.text_enc_lstm.weight_ih) {
        hidden = lstm_forward(ctx, hidden, weights.text_enc_lstm);
    }
    
    return hidden;
}

ggml_tensor* Model::build_encoder(ggml_context* ctx, ggml_tensor* hidden) {
    // If CNN-LSTM weights exist, use them. Otherwise fallback to ALBERT.
    if (weights_.text_enc_cnn[0].conv_weight_v) {
        if (heartbeat_verbose()) {
            std::cerr << "DEBUG: Using CNN-LSTM Text Encoder Path\n";
        }
        return build_cnn_lstm_encoder(ctx, hidden, weights_);
    }
    
    if (heartbeat_verbose()) {
        std::cerr << "DEBUG: Using ALBERT Transformer Text Encoder Path\n";
    }
    if (weights_.bert_proj_weight) {
        hidden = ggml_mul_mat(ctx, weights_.bert_proj_weight, hidden);
        if (weights_.bert_proj_bias) {
            hidden = ggml_add(ctx, hidden, weights_.bert_proj_bias);
        }
    }
    
    for (int i = 0; i < 12; i++) {
        hidden = build_attention(ctx, hidden, weights_.albert_layer);
        hidden = build_ffn(ctx, hidden, weights_.albert_layer);
    }
    
    // Project to Predictor Dimension (768 -> 512)
    if (weights_.text_enc_proj_weight) {
        hidden = ggml_mul_mat(ctx, weights_.text_enc_proj_weight, hidden);
        if (weights_.text_enc_proj_bias) {
            hidden = ggml_add(ctx, hidden, weights_.text_enc_proj_bias);
        }
    }
    
    return hidden;
}

// ========================================
// Decoder Helpers (StyleTTS2 / HiFiGAN)
// ========================================

static ggml_tensor* predictor_lstm_forward(ggml_context* ctx, 
                                           ggml_tensor* x, 
                                           ggml_tensor* style_rep,
                                           const LSTMWeights& w) {
    if (!x || !w.weight_ih) return x;
    
    // x: [Dim, SeqLen] (512)
    // style_rep: [StyleDim, SeqLen] (128)
    // Concat to [640, SeqLen]
    ggml_tensor* input = ggml_concat(ctx, x, style_rep, 0);
    
    return lstm_forward(ctx, input, w);
}

ggml_tensor* Model::build_duration_predictor(ggml_context* ctx, 
                                             ggml_tensor* hidden, 
                                             ggml_tensor* style) {
    int seq_len = (int)hidden->ne[1];
    int style_dim = (int)style->ne[0];
    
    // Broadcast style
    ggml_tensor* style_rep = ggml_repeat(ctx, style, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style_dim, seq_len));

    ggml_tensor* x = hidden; // [512, Seq]
    
    // 3 Blocks of (Predictor LSTM + Linear)
    for (int i = 0; i < 3; i++) {
        // 1. LSTM (0, 2, 4)
        if (weights_.predictor.text_encoder[i].lstm.weight_ih) {
            x = predictor_lstm_forward(ctx, x, style_rep, weights_.predictor.text_encoder[i].lstm);
        }
        
        // 2. Linear (1, 3, 5)
        if (weights_.predictor.text_encoder[i].linear.weight) {
            // Linear layers in predictor also appear to take [Style]?
            // Usually not, but if weight ne[0] is 640, then it does.
            // In our discovery, they were 128 elements?
            // Let's check linear weight dimension and only project if it matches.
            ggml_tensor* w_lin = weights_.predictor.text_encoder[i].linear.weight;
            if (w_lin->ne[0] == x->ne[0]) {
                x = linear_forward(ctx, x, weights_.predictor.text_encoder[i].linear);
            } else {
                std::cerr << "WARNING: Skipping Linear layer " << 2*i+1 << " due to mismatch: " << w_lin->ne[0] << " vs " << x->ne[0] << "\n";
            }
        }
    }
    
    // Shared LSTM
    if (weights_.predictor.shared_lstm.weight_ih) {
        x = predictor_lstm_forward(ctx, x, style_rep, weights_.predictor.shared_lstm);
    }
    
    // Duration LSTM
    if (weights_.predictor.duration_lstm.weight_ih) {
        x = predictor_lstm_forward(ctx, x, style_rep, weights_.predictor.duration_lstm);
    }
    
    // Final Projection to Duration
    if (weights_.predictor.duration_proj.weight) {
        x = linear_forward(ctx, x, weights_.predictor.duration_proj);
    }
    
    return x;
}


ModelOutput Model::forward(const std::vector<int>& tokens,
                            const std::vector<float>& style) {
    ModelOutput output;
    output.n_mels = params_.n_mels;
    output.n_frames = 0;
    
    if (!loaded_) {
        if (heartbeat_verbose()) {
            std::cerr << "DEBUG: Model not loaded in forward\n";
        }
        return output;
    }
    
    int seq_len = static_cast<int>(tokens.size());
    const bool verbose = heartbeat_verbose();
    if (verbose) {
        std::cerr << "DEBUG: Forward pass with seq_len=" << seq_len << "\n";
    }
    const int n_threads = heartbeat_num_threads();
    
    // 1. Initialize GGML context for this inference
    // Increased to 12GB for Decoder activations (high res audio)
    // Audio tensors are large [1, T]. Intermediate [C, T].
    size_t ctx_size = 12288LL * 1024 * 1024;
    
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx0 = ggml_init(params);
    if (!ctx0) {
        std::cerr << "ERROR: Failed to allocate GGML context\n";
        return output;
    }
    

    // 5. Initialize style tensor
    int style_dim = params_.style_dim > 0 ? params_.style_dim : 128;
    if (!style.empty()) {
        style_dim = static_cast<int>(style.size());
    }
    if (style_dim <= 0) {
        style_dim = 128;
    }

    ggml_tensor* style_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, style_dim);
    ggml_set_zero(style_tensor);
    if (!style.empty()) {
        std::memcpy(style_tensor->data, style.data(), static_cast<size_t>(style_dim) * sizeof(float));
    }
    
    // 6. Build Text Encoder (ALBERT)
    if (verbose) {
        std::cerr << "DEBUG: Building Embeddings...\n";
    }
    // 5. Build Embeddings
    ggml_tensor* hidden = build_embeddings(ctx0, tokens); // [seq, 128 or 256]
    
    // DEBUG: Compute Embeddings
    if (verbose) {
        std::cerr << "DEBUG: Computing Embeddings Graph...\n";
        struct ggml_cgraph* gf_1 = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf_1, hidden);
        ggml_graph_compute_with_ctx(ctx0, gf_1, n_threads);
        
        float* d = (float*)hidden->data;
        std::cerr << "DEBUG: Embeddings [0]=" << d[0] << " is_nan=" << std::isnan(d[0]) << "\n";
    }

    if (verbose) {
        std::cerr << "DEBUG: Building Encoder...\n";
    }
    hidden = build_encoder(ctx0, hidden); // [768, seq]
    
    // DEBUG: Compute Encoder
    if (verbose) {
        std::cerr << "DEBUG: Computing Encoder Graph...\n";
        struct ggml_cgraph* gf_2 = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf_2, hidden);
        ggml_graph_compute_with_ctx(ctx0, gf_2, n_threads);
        
        float* d = (float*)hidden->data;
        std::cerr << "DEBUG: Encoder [0]=" << d[0] << " is_nan=" << std::isnan(d[0]) << "\n";
    }
    
    // 7. Predict Durations
    if (verbose) {
        std::cerr << "DEBUG: Building Duration Predictor...\n";
    }
    ggml_tensor* log_duration = build_duration_predictor(ctx0, hidden, style_tensor);
    
    // DEBUG: Compute Durations
    if (verbose) {
        struct ggml_cgraph* gf_dur = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf_dur, log_duration);
        ggml_graph_compute_with_ctx(ctx0, gf_dur, n_threads);
        
        float* d = (float*)log_duration->data;
        std::cerr << "DEBUG: Log Duration [0]=" << d[0] << " seq_len=" << log_duration->ne[1] << "\n";
        
        float total_dur = 0;
        for (int i = 0; i < log_duration->ne[1]; i++) {
            total_dur += expf(d[i]);
        }
        std::cerr << "DEBUG: Total Predicted Frames: " << total_dur << "\n";
    }
    
    // 8. Compute Graph (Part 1: Duration)
    if (verbose) {
        std::cerr << "DEBUG: Computing Duration Graph...\n";
    }
    struct ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, log_duration);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);
    if (verbose) {
        std::cerr << "DEBUG: Duration Graph Computed.\n";
    }
    
    // 9. Process Duration Output
    std::vector<int> durations;
    int total_frames = 0;
    
    float* dur_data = (float*)log_duration->data;
    // log_duration can be [bins, seq] (preferred) or [1, seq]
    const int dur_bins = static_cast<int>(log_duration->ne[0]);
    const int dur_tokens = static_cast<int>(log_duration->ne[1]);
    const int token_count = std::min(seq_len, dur_tokens);
    
    if (verbose) {
        std::cerr << "DEBUG: Duration shape: bins=" << dur_bins << ", tokens=" << dur_tokens << "\n";
        std::cerr << "DEBUG: Duration values (log): ";
        for (int i = 0; i < std::min(dur_tokens * dur_bins, 10); i++) std::cerr << dur_data[i] << " ";
        std::cerr << "...\n";
    }
    
    for (int t = 0; t < token_count; t++) {
        float d = 1.0f;

        if (dur_bins > 1) {
            // Match Kokoro reference behavior:
            // duration = sigmoid(logits).sum(axis=-1)
            float sum_sigmoid = 0.0f;
            const int base = t * dur_bins;
            for (int b = 0; b < dur_bins; b++) {
                float v = dur_data[base + b];
                if (!std::isfinite(v)) {
                    v = 0.0f;
                }
                sum_sigmoid += 1.0f / (1.0f + std::exp(-v));
            }
            d = sum_sigmoid;
        } else {
            // Legacy scalar-log-duration fallback.
            float v = dur_data[t];
            if (!std::isfinite(v)) {
                v = 0.0f;
            }
            d = std::exp(v);
        }

        d = std::clamp(d, 1.0f, 100.0f);
        int d_int = static_cast<int>(std::round(d));
        if (d_int < 1) {
            d_int = 1;
        }

        durations.push_back(d_int);
        total_frames += d_int;
    }

    // If duration output had fewer tokens than input, pad remaining with 1 frame.
    for (int t = token_count; t < seq_len; t++) {
        durations.push_back(1);
        total_frames += 1;
    }
    
    // Safety cap for total frames
    if (total_frames > 24000 * 30) { // Limit to 30 seconds
        total_frames = 24000 * 30;
    }
    if (total_frames <= 0) total_frames = 1;
    
    if (verbose) {
        std::cerr << "DEBUG: Predicted total frames: " << total_frames << "\n";
    }
    
    // 9. Up-sample Hidden States (CPU Side)
    // hidden is [512, seq_len]
    // durations is vector<int> size seq_len
    
    // Create new tensor input for decoder
    // size [512, total_frames]
    
    if (verbose) {
        std::cerr << "DEBUG: Upsampling hidden states to " << total_frames << " frames...\n";
    }
    // Create decoder input in GGML's native [Length, Channels, Batch] layout
    // ggml_new_tensor_3d args: (ctx, type, ne0, ne1, ne2)
    // For [L, C, N]: ne0=L, ne1=C, ne2=N
    ggml_tensor* dec_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, total_frames, 512);
    
    // Safety check: total_frames > 0
    if (total_frames <= 0) total_frames = 1;
    
    float* src_ptr = (float*)hidden->data;
    float* dst_ptr = (float*)dec_input->data;
    int hidden_dim = 512; 
    
    // Upsample: for each phoneme, repeat its hidden state 'duration' times
    // Layout: [total_frames, 512] stored as rows
    // Each row is one frame with 512 channels
    int current_frame =0;
    const int expand_tokens = std::min(seq_len, static_cast<int>(durations.size()));
    for (int i = 0; i < expand_tokens && current_frame < total_frames; i++) { 
        int d = durations[i];
        if (d > 0) {
            float* s = src_ptr + i * hidden_dim;
            for (int k = 0; k < d && current_frame < total_frames; k++) {
                // Copy 512 channels for this frame  
                // GGML stores [L, C] as L rows of C elements
                // Frame k is at offset k*512
                 std::memcpy(dst_ptr + current_frame * hidden_dim, 
                             s, 
                             hidden_dim * sizeof(float));
                 current_frame++;
            }
        }
    }
    
    // 10. Build Decoder
    if (verbose) {
        std::cerr << "DEBUG: Building Decoder Graph...\n";
    }
    ggml_tensor* audio_tensor = build_decoder(ctx0, dec_input, style_tensor);
    
    // 11. Compute Decoder
    {
        if (verbose) {
            std::cerr << "DEBUG: Computing Decoder (High VRAM usage)...\n";
        }
        struct ggml_cgraph* gf_dec = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf_dec, audio_tensor);
        ggml_graph_compute_with_ctx(ctx0, gf_dec, n_threads);
        if (verbose) {
            std::cerr << "DEBUG: Decoder Graph Computed.\n";
        }
    }
    
    // 12. Extract Audio
    // audio_tensor layout is [Length, Channels=22, Batch=1]
    // Hypothesis: 22 channels are sub-band waveforms that need to be summed
    // This is common in multi-band vocoders (e.g., multi-band excitation)
    int n_frames = (int)audio_tensor->ne[0];  // Length/time dimension
    int n_channels = (int)audio_tensor->ne[1]; // Should be 22
    
    if (verbose) {
        std::cerr << "DEBUG: Decoder output shape: [" << n_frames << ", " << n_channels << ", " << audio_tensor->ne[2] << "]\n";
        std::cerr << "DEBUG: Summing " << n_channels << " sub-band channels\n";
    }
    
    output.n_frames = n_frames;
    output.n_mels = 1;  // Raw audio
    output.magnitude.resize(n_frames);
    output.phase.resize(0);
    
    float* audio_data = (float*)audio_tensor->data;
    
    // Sum all channels to create final waveform
    for (int i = 0; i < n_frames; i++) {
        float sum = 0.0f;
        for (int c = 0; c < n_channels; c++) {
            sum += audio_data[c * n_frames + i];
        }
        output.magnitude[i] = sum;  // Direct sum, no averaging
    }
    
    
    ggml_free(ctx0);
    compute_ctx_ = nullptr;
    
    return output;
}

// Helper for ResBlock
static ggml_tensor* build_resblock_impl(ggml_context* ctx, 
                                        ggml_tensor* x, 
                                        ggml_tensor* style, 
                                        const ResBlock& w) {
    static const int dilations[3] = {1, 3, 5};
    
    for (int j = 0; j < 3; j++) {
        if (!w.convs1[j].weight_v || !w.convs2[j].weight_v) {
            continue;
        }

        ggml_tensor* input = x; 
        
        // 1. AdaIN 1
        ggml_tensor* h = build_adain(ctx, x, style, w.adain1[j]);
        
        // 2. LeakyReLU
        h = ggml_leaky_relu(ctx, h, 0.1f, true); 
        
        // 3. Conv1
        int K1 = (int)w.convs1[j].weight_v->ne[0];
        int P1 = dilations[j] * (K1 - 1) / 2;
        h = weight_norm_conv1d(ctx, h, w.convs1[j].weight_g, w.convs1[j].weight_v, w.convs1[j].bias, 1, P1, dilations[j]);
        
        // 4. AdaIN 2
        h = build_adain(ctx, h, style, w.adain2[j]);
        
        // 5. LeakyReLU
        h = ggml_leaky_relu(ctx, h, 0.1f, true);
        
        // 6. Conv2
        int K2 = (int)w.convs2[j].weight_v->ne[0];
        int P2 = (K2 - 1) / 2;
        h = weight_norm_conv1d(ctx, h, w.convs2[j].weight_g, w.convs2[j].weight_v, w.convs2[j].bias, 1, P2, 1);
        
        // Residual Add with Alpha
        if (w.alpha2[j]) {
             // x = alpha1 * input + alpha2 * h ? OR input + alpha * h?
             // Usually it's input + alpha * h.
             h = ggml_mul(ctx, h, ggml_repeat(ctx, w.alpha2[j], h));
             x = ggml_add(ctx, input, h);
        } else {
             x = ggml_add(ctx, input, h);
        }
    }
    return x;
}




ggml_tensor* Model::build_decoder(ggml_context* ctx, ggml_tensor* en, ggml_tensor* style) {
    // en: [TotalFrames, Channels(512)] (from upsampling step - already correct!)
    // Just need to add batch dimension: [TotalFrames, Channels, 1]
    ggml_tensor* x = ggml_reshape_3d(ctx, en, en->ne[0], en->ne[1], 1);

    // Upsample stages (Kokoro generator has 2 upsample blocks)
    static constexpr int kUpsampleRates[2] = {10, 6};
    for (int i = 0; i < 2; i++) {
        if (weights_.generator.ups[i].weight_v) {
            x = weight_norm_conv_transpose_1d(
                ctx,
                x,
                weights_.generator.ups[i].weight_g,
                weights_.generator.ups[i].weight_v,
                weights_.generator.ups[i].bias,
                kUpsampleRates[i]
            );
        }

        x = ggml_leaky_relu(ctx, x, 0.1f, true);

        if (weights_.generator.noise_res[i].convs1[0].weight_v) {
            x = build_resblock_impl(ctx, x, style, weights_.generator.noise_res[i]);
        }

        ggml_tensor* mrf_sum = nullptr;
        int mrf_count = 0;
        const int mrf_start = i * 3;
        for (int j = 0; j < 3; j++) {
            const int blk_idx = mrf_start + j;
            if (blk_idx >= 6 || !weights_.generator.resblocks[blk_idx].convs1[0].weight_v) {
                continue;
            }

            ggml_tensor* r_out = build_resblock_impl(ctx, x, style, weights_.generator.resblocks[blk_idx]);
            mrf_sum = mrf_sum ? ggml_add(ctx, mrf_sum, r_out) : r_out;
            mrf_count++;
        }

        if (mrf_sum && mrf_count > 0) {
            x = ggml_scale(ctx, mrf_sum, 1.0f / static_cast<float>(mrf_count));
        }
    }

    // Final activation and post conv
    x = ggml_leaky_relu(ctx, x, 0.1f, true);

    if (weights_.generator.conv_post.weight_v) {
        x = weight_norm_conv1d(ctx, x,
                               weights_.generator.conv_post.weight_g,
                               weights_.generator.conv_post.weight_v,
                               weights_.generator.conv_post.bias, 1, 3);
    }

    // Output waveform range
    x = ggml_tanh(ctx, x);
    
    return x;
}

} // namespace heartbeat

