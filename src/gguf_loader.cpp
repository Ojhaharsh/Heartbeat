#include "gguf_loader.h"

#include <fstream>
#include <cstring>
#include <stdexcept>
#include <algorithm>

// GGML headers
extern "C" {
#include "ggml.h"
}

namespace heartbeat {

// Helper to read values from buffer
template<typename T>
T read_value(const uint8_t* data, size_t& offset) {
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

std::string read_string(const uint8_t* data, size_t& offset) {
    uint64_t len = read_value<uint64_t>(data, offset);
    std::string str(reinterpret_cast<const char*>(data + offset), len);
    offset += len;
    return str;
}

GGUFValue read_gguf_value(const uint8_t* data, size_t& offset, gguf::ValueType type);

std::vector<std::string> read_string_array(const uint8_t* data, size_t& offset) {
    auto arr_type = read_value<uint32_t>(data, offset);
    uint64_t arr_len = read_value<uint64_t>(data, offset);
    
    std::vector<std::string> result;
    result.reserve(arr_len);
    
    for (uint64_t i = 0; i < arr_len; i++) {
        if (arr_type == static_cast<uint32_t>(gguf::ValueType::STRING)) {
            result.push_back(read_string(data, offset));
        } else {
            // Skip non-string array elements
            offset += 8;  // Approximate, adjust based on type
        }
    }
    return result;
}

GGUFValue read_gguf_value(const uint8_t* data, size_t& offset, gguf::ValueType type) {
    switch (type) {
        case gguf::ValueType::UINT8:
            return read_value<uint8_t>(data, offset);
        case gguf::ValueType::INT8:
            return read_value<int8_t>(data, offset);
        case gguf::ValueType::UINT16:
            return read_value<uint16_t>(data, offset);
        case gguf::ValueType::INT16:
            return read_value<int16_t>(data, offset);
        case gguf::ValueType::UINT32:
            return read_value<uint32_t>(data, offset);
        case gguf::ValueType::INT32:
            return read_value<int32_t>(data, offset);
        case gguf::ValueType::UINT64:
            return read_value<uint64_t>(data, offset);
        case gguf::ValueType::INT64:
            return read_value<int64_t>(data, offset);
        case gguf::ValueType::FLOAT32:
            return read_value<float>(data, offset);
        case gguf::ValueType::FLOAT64:
            return read_value<double>(data, offset);
        case gguf::ValueType::BOOL:
            return static_cast<bool>(read_value<uint8_t>(data, offset));
        case gguf::ValueType::STRING:
            return read_string(data, offset);
        case gguf::ValueType::ARRAY:
            return read_string_array(data, offset);
        default:
            throw std::runtime_error("Unknown GGUF value type");
    }
}

size_t get_tensor_type_size(gguf::TensorType type) {
    switch (type) {
        case gguf::TensorType::F32:  return 4;
        case gguf::TensorType::F16:  return 2;
        case gguf::TensorType::BF16: return 2;
        case gguf::TensorType::Q4_0: return 1;  // ~0.5 bytes per element
        case gguf::TensorType::Q4_1: return 1;
        case gguf::TensorType::Q8_0: return 1;
        default: return 4;
    }
}

ggml_type to_ggml_type(gguf::TensorType type) {
    switch (type) {
        case gguf::TensorType::F32:  return GGML_TYPE_F32;
        case gguf::TensorType::F16:  return GGML_TYPE_F16;
        case gguf::TensorType::Q4_0: return GGML_TYPE_Q4_0;
        case gguf::TensorType::Q4_1: return GGML_TYPE_Q4_1;
        case gguf::TensorType::Q8_0: return GGML_TYPE_Q8_0;
        default: return GGML_TYPE_F32;
    }
}

// GGUFLoader implementation

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

bool GGUFLoader::load(const std::string& path) {
    // Read entire file
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    file_data_.resize(file_size);
    if (!file.read(reinterpret_cast<char*>(file_data_.data()), file_size)) {
        return false;
    }
    file.close();
    
    const uint8_t* data = file_data_.data();
    size_t offset = 0;
    
    // Parse header
    if (!parse_header(data, file_size)) {
        return false;
    }
    offset = 8;  // After magic + version
    
    // Read counts
    uint64_t n_tensors = read_value<uint64_t>(data, offset);
    uint64_t n_kv = read_value<uint64_t>(data, offset);
    
    // Parse metadata
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(data, offset);
        auto value_type = static_cast<gguf::ValueType>(read_value<uint32_t>(data, offset));
        GGUFValue value = read_gguf_value(data, offset, value_type);
        metadata_[key] = value;
    }
    
    // Parse tensor infos
    for (uint64_t i = 0; i < n_tensors; i++) {
        TensorInfo info;
        info.name = read_string(data, offset);
        
        uint32_t n_dims = read_value<uint32_t>(data, offset);
        info.dimensions.resize(n_dims);
        for (uint32_t j = 0; j < n_dims; j++) {
            info.dimensions[j] = read_value<uint64_t>(data, offset);
        }
        
        info.type = static_cast<gguf::TensorType>(read_value<uint32_t>(data, offset));
        info.offset = read_value<uint64_t>(data, offset);
        info.tensor = nullptr;
        
        // Calculate size
        size_t n_elements = 1;
        for (auto dim : info.dimensions) {
            n_elements *= dim;
        }
        info.size_bytes = n_elements * get_tensor_type_size(info.type);
        
        tensors_[info.name] = info;
    }
    
    // Align to tensor data
    size_t alignment = 32;
    if (auto* align_val = get_metadata("general.alignment")) {
        if (auto* val = std::get_if<uint32_t>(align_val)) {
            alignment = *val;
        }
    }
    
    size_t data_offset = (offset + alignment - 1) & ~(alignment - 1);
    
    // Load tensor data into GGML
    if (!load_tensor_data(data, data_offset)) {
        return false;
    }
    
    // Extract model parameters
    extract_params();
    
    loaded_ = true;
    return true;
}

bool GGUFLoader::parse_header(const uint8_t* data, size_t size) {
    if (size < 8) return false;
    
    uint32_t magic;
    std::memcpy(&magic, data, 4);
    if (magic != gguf::MAGIC) {
        return false;
    }
    
    std::memcpy(&version_, data + 4, 4);
    if (version_ != gguf::VERSION_2 && version_ != gguf::VERSION_3) {
        return false;
    }
    
    return true;
}

bool GGUFLoader::load_tensor_data(const uint8_t* data, size_t data_offset) {
    // Calculate total memory needed
    size_t total_size = 0;
    for (const auto& [name, info] : tensors_) {
        total_size += info.size_bytes + 64;  // Padding
    }
    
    // Create GGML context
    struct ggml_init_params params = {
        .mem_size = total_size + 64 * 1024 * 1024,  // Extra 64 MB for overhead
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    
    ctx_ = ggml_init(params);
    if (!ctx_) {
        return false;
    }
    
    // Create and load each tensor
    for (auto& [name, info] : tensors_) {
        // Convert dimensions
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (size_t i = 0; i < info.dimensions.size() && i < GGML_MAX_DIMS; i++) {
            ne[i] = info.dimensions[i];
        }
        
        // Create tensor
        ggml_type gtype = to_ggml_type(info.type);
        ggml_tensor* tensor = ggml_new_tensor(ctx_, gtype, info.dimensions.size(), ne);
        
        if (!tensor) {
            continue;
        }
        
        ggml_set_name(tensor, name.c_str());
        
        // Copy data
        const uint8_t* tensor_data = data + data_offset + info.offset;
        std::memcpy(tensor->data, tensor_data, ggml_nbytes(tensor));
        
        info.tensor = tensor;
    }
    
    return true;
}

void GGUFLoader::extract_params() {
    // Extract hyperparameters from metadata
    auto get_int = [this](const std::string& key, int default_val) -> int {
        if (auto* val = get_metadata(key)) {
            if (auto* v = std::get_if<uint32_t>(val)) return *v;
            if (auto* v = std::get_if<int32_t>(val)) return *v;
        }
        return default_val;
    };
    
    params_.vocab_size = get_int("kokoro.vocab_size", 178);
    params_.hidden_size = get_int("kokoro.hidden_size", 768);
    params_.num_layers = get_int("kokoro.num_layers", 12);
    params_.num_heads = get_int("kokoro.num_heads", 12);
    params_.style_dim = get_int("kokoro.style_dim", 256);
    params_.n_mels = get_int("kokoro.n_mels", 80);
    params_.istft_n_fft = get_int("kokoro.istft_n_fft", 16);
    params_.istft_hop_length = get_int("kokoro.istft_hop_length", 4);
    params_.sample_rate = get_int("kokoro.sample_rate", 24000);
    
    // Extract vocabulary
    if (auto* vocab_val = get_metadata("tokenizer.ggml.tokens")) {
        if (auto* tokens = std::get_if<std::vector<std::string>>(vocab_val)) {
            vocab_ = *tokens;
            for (size_t i = 0; i < vocab_.size(); i++) {
                vocab_map_[vocab_[i]] = static_cast<int>(i);
            }
        }
    }
}

const GGUFValue* GGUFLoader::get_metadata(const std::string& key) const {
    auto it = metadata_.find(key);
    return (it != metadata_.end()) ? &it->second : nullptr;
}

ggml_tensor* GGUFLoader::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? it->second.tensor : nullptr;
}

int GGUFLoader::phoneme_to_id(const std::string& phoneme) const {
    auto it = vocab_map_.find(phoneme);
    return (it != vocab_map_.end()) ? it->second : -1;
}

std::vector<std::string> GGUFLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        names.push_back(name);
    }
    return names;
}

} // namespace heartbeat
