#include "dsp.h"

#include <cmath>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <cstdlib>

// KissFFT integration
#define KISS_FFT_SCALAR float
#include "kiss_fft.h"

namespace heartbeat {

DSP::DSP(const DSPConfig& config) : config_(config) {
    // Pre-compute Hanning window
    window_ = hanning_window(config_.n_fft);
    init_fft();
}


DSP::DSP(DSP&& other) noexcept : config_(other.config_), window_(std::move(other.window_)), fft_cfg_(other.fft_cfg_), fft_fwd_cfg_(other.fft_fwd_cfg_) {
    other.fft_cfg_ = nullptr;
    other.fft_fwd_cfg_ = nullptr;
}

DSP& DSP::operator=(DSP&& other) noexcept {
    if (this != &other) {
        cleanup_fft();
        
        config_ = other.config_;
        window_ = std::move(other.window_);
        fft_cfg_ = other.fft_cfg_;
        fft_fwd_cfg_ = other.fft_fwd_cfg_;
        
        other.fft_cfg_ = nullptr;
        other.fft_fwd_cfg_ = nullptr;
    }
    return *this;
}

DSP::~DSP() {
    cleanup_fft();
}

void DSP::init_fft() {
    const bool verbose = std::getenv("HEARTBEAT_VERBOSE") != nullptr;
    if (verbose) std::cerr << "DEBUG DSP: init_fft called for n_fft=" << config_.n_fft << "\n";
    // Initialize inverse FFT configuration
    fft_cfg_ = kiss_fft_alloc(config_.n_fft, 1, nullptr, nullptr);  // 1 = inverse
    // Initialize forward FFT configuration
    fft_fwd_cfg_ = kiss_fft_alloc(config_.n_fft, 0, nullptr, nullptr);  // 0 = forward
    if (!fft_cfg_) {
        if (verbose) std::cerr << "DEBUG DSP: Failed to allocate inverse kiss_fft_cfg!\n";
    }
    if (!fft_fwd_cfg_) {
        if (verbose) std::cerr << "DEBUG DSP: Failed to allocate forward kiss_fft_cfg!\n";
    }
    if (verbose && fft_cfg_ && fft_fwd_cfg_) {
        std::cerr << "DEBUG DSP: kiss_fft_cfg allocated (inv=" << fft_cfg_ << ", fwd=" << fft_fwd_cfg_ << ")\n";
    }
}

void DSP::cleanup_fft() {
    if (fft_cfg_) {
        kiss_fft_free(fft_cfg_);
        fft_cfg_ = nullptr;
    }
    if (fft_fwd_cfg_) {
        kiss_fft_free(fft_fwd_cfg_);
        fft_fwd_cfg_ = nullptr;
    }
}

std::vector<float> DSP::hanning_window(int size) {
    std::vector<float> window(size);
    constexpr float PI = 3.14159265358979323846f;
    
    // Use periodic Hanning window (matching PyTorch torch.hann_window(periodic=True))
    // periodic: divide by size, not (size-1)
    for (int i = 0; i < size; i++) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * PI * i / size));
    }
    
    return window;
}

void DSP::ifft(const std::complex<float>* in, std::complex<float>* out, int n) {
    // std::cerr << "DEBUG DSP: ifft called calling with n=" << n << "\n";
    if (!fft_cfg_) {
        if (std::getenv("HEARTBEAT_VERBOSE") != nullptr) {
            std::cerr << "DEBUG DSP: ifft called with null cfg!\n";
        }
        return;
    }
    auto cfg = static_cast<kiss_fft_cfg>(fft_cfg_);
    
    // KissFFT uses its own complex type
    std::vector<kiss_fft_cpx> kin(n), kout(n);
    
    for (int i = 0; i < n; i++) {
        kin[i].r = in[i].real();
        kin[i].i = in[i].imag();
    }
    
    kiss_fft(cfg, kin.data(), kout.data());
    
    // Scale by 1/N for inverse FFT
    float scale = 1.0f / n;
    for (int i = 0; i < n; i++) {
        out[i] = std::complex<float>(kout[i].r * scale, kout[i].i * scale);
    }
}

void DSP::fft(const std::complex<float>* in, std::complex<float>* out, int n) {
    if (!fft_fwd_cfg_) {
        if (std::getenv("HEARTBEAT_VERBOSE") != nullptr) {
            std::cerr << "DEBUG DSP: fft called with null forward cfg!\n";
        }
        return;
    }
    auto cfg = static_cast<kiss_fft_cfg>(fft_fwd_cfg_);
    std::vector<kiss_fft_cpx> kin(n), kout(n);
    for (int i = 0; i < n; i++) {
        kin[i].r = in[i].real();
        kin[i].i = in[i].imag();
    }
    kiss_fft(cfg, kin.data(), kout.data());
    for (int i = 0; i < n; i++) {
        out[i] = std::complex<float>(kout[i].r, kout[i].i);
    }
}

int DSP::stft(const std::vector<float>& signal,
              std::vector<float>& out_spec,
              std::vector<float>& out_phase,
              int& out_n_frames) {
    const int n_fft = config_.n_fft;
    const int hop = config_.hop_length;
    const int half_bins = n_fft / 2 + 1;

    if (window_.empty()) {
        window_ = hanning_window(n_fft);
    }

    // center=True: pad signal by n_fft/2 on each side (matching torch.stft default)
    const int pad = n_fft / 2;
    std::vector<float> padded(signal.size() + 2 * pad, 0.0f);
    std::memcpy(padded.data() + pad, signal.data(), signal.size() * sizeof(float));
    // Reflect padding (matching PyTorch reflect mode)
    for (int i = 0; i < pad && i < (int)signal.size(); i++) {
        padded[pad - 1 - i] = signal[i + 1 < (int)signal.size() ? i + 1 : 0];
        padded[pad + signal.size() + i] = signal[signal.size() > 1 ? signal.size() - 2 - i : 0];
    }

    const int padded_len = static_cast<int>(padded.size());
    out_n_frames = (padded_len - n_fft) / hop + 1;
    if (out_n_frames <= 0) {
        out_spec.clear();
        out_phase.clear();
        return half_bins;
    }

    out_spec.resize(out_n_frames * half_bins);
    out_phase.resize(out_n_frames * half_bins);

    std::vector<std::complex<float>> fft_in(n_fft);
    std::vector<std::complex<float>> fft_out(n_fft);

    for (int t = 0; t < out_n_frames; t++) {
        const int offset = t * hop;
        for (int k = 0; k < n_fft; k++) {
            fft_in[k] = std::complex<float>(padded[offset + k] * window_[k], 0.0f);
        }
        fft(fft_in.data(), fft_out.data(), n_fft);
        for (int f = 0; f < half_bins; f++) {
            const int idx = t * half_bins + f;
            out_spec[idx] = std::abs(fft_out[f]);
            out_phase[idx] = std::arg(fft_out[f]);
        }
    }

    return half_bins;
}

std::vector<float> DSP::istft(const std::vector<float>& magnitude,
                               const std::vector<float>& phase,
                               int n_mels,
                               int n_frames) {
    // Construct complex spectrogram from magnitude and phase
    std::vector<std::complex<float>> spectrogram(n_mels * n_frames);
    
    for (int t = 0; t < n_frames; t++) {
        for (int f = 0; f < n_mels; f++) {
            int idx = t * n_mels + f;
            float mag = magnitude[idx];
            float ph = phase[idx];
            // Complex = magnitude * e^(i * phase)
            spectrogram[idx] = std::polar(mag, ph);
        }
    }
    
    return istft(spectrogram, n_frames);
}

std::vector<float> DSP::istft(const std::vector<std::complex<float>>& spectrogram,
                               int n_frames) {
    int n_fft = config_.n_fft;
    int hop_length = config_.hop_length;
    
    if (n_fft <= 0 || hop_length <= 0) return {};
    
    if (window_.empty()) {
        window_ = hanning_window(n_fft);
    }
    
    // Output audio length
    int audio_length = (n_frames - 1) * hop_length + n_fft;
    std::vector<float> audio(audio_length, 0.0f);
    std::vector<float> window_sum(audio_length, 0.0f);
    
    // IFFT buffer
    std::vector<std::complex<float>> fft_in(n_fft);
    std::vector<std::complex<float>> fft_out(n_fft);
    const int bins_per_frame = n_frames > 0 ? static_cast<int>(spectrogram.size() / n_frames) : 0;
    const int half_bins = n_fft / 2 + 1;
    
    for (int t = 0; t < n_frames; t++) {
        std::fill(fft_in.begin(), fft_in.end(), std::complex<float>(0.0f, 0.0f));

        // Reconstruct full complex spectrum from one-sided bins when possible.
        if (bins_per_frame == half_bins) {
            for (int f = 0; f < half_bins; f++) {
                fft_in[f] = spectrogram[t * bins_per_frame + f];
            }

            // Hermitian symmetry for real-valued waveform.
            const int nyquist = n_fft / 2;
            for (int f = 1; f < nyquist; f++) {
                fft_in[n_fft - f] = std::conj(fft_in[f]);
            }
            if (n_fft % 2 == 0) {
                fft_in[nyquist] = std::complex<float>(fft_in[nyquist].real(), 0.0f);
            }
        } else {
            // Fallback if input already appears full-band.
            const int copy_bins = std::min(n_fft, bins_per_frame);
            for (int f = 0; f < copy_bins; f++) {
                fft_in[f] = spectrogram[t * bins_per_frame + f];
            }
        }
        
        // Inverse FFT
        ifft(fft_in.data(), fft_out.data(), n_fft);
        
        // Overlap-add with window
        int offset = t * hop_length;
        for (int k = 0; k < n_fft && (offset + k) < audio_length; k++) {
            audio[offset + k] += fft_out[k].real() * window_[k];
            window_sum[offset + k] += window_[k] * window_[k];
        }
    }
    
    // Normalize by window sum (avoid division by zero)
    for (int i = 0; i < audio_length; i++) {
        if (window_sum[i] > 1e-8f) {
            audio[i] /= window_sum[i];
        }
    }
    
    // torch.istft with center=True crops n_fft // 2 from both sides of the output
    int pad = n_fft / 2;
    if (audio_length > 2 * pad) {
        std::vector<float> cropped(audio.begin() + pad, audio.end() - pad);
        return cropped;
    }
    
    return audio;
}

std::vector<float> DSP::apply_window(const std::vector<float>& signal, int window_size) {
    auto window = hanning_window(window_size);
    std::vector<float> result(signal.size());
    
    for (size_t i = 0; i < signal.size(); i++) {
        result[i] = signal[i] * window[i % window_size];
    }
    
    return result;
}

std::vector<float> DSP::overlap_add(const std::vector<std::vector<float>>& frames,
                                     int hop_length) {
    if (frames.empty()) return {};
    
    int frame_size = static_cast<int>(frames[0].size());
    int n_frames = static_cast<int>(frames.size());
    int output_length = (n_frames - 1) * hop_length + frame_size;
    
    std::vector<float> output(output_length, 0.0f);
    
    for (int t = 0; t < n_frames; t++) {
        int offset = t * hop_length;
        for (int k = 0; k < frame_size; k++) {
            output[offset + k] += frames[t][k];
        }
    }
    
    return output;
}

std::vector<float> DSP::normalize(const std::vector<float>& audio, float target_peak) {
    if (audio.empty()) return audio;
    
    // First: Center the signal by subtracting the DC offset (mean)
    // Neural vocoders can output high DC biases which completely squash the AC peaks.
    double sum = 0.0;
    for (float sample : audio) {
        sum += sample;
    }
    float mean = static_cast<float>(sum / audio.size());
    
    std::vector<float> ac_audio(audio.size());
    for (size_t i = 0; i < audio.size(); i++) {
        ac_audio[i] = audio[i] - mean;
    }
    
    // Use a robust peak estimate so isolated spikes don't collapse all speech to ~0.
    std::vector<float> abs_vals;
    abs_vals.reserve(ac_audio.size());
    for (float sample : ac_audio) {
        abs_vals.push_back(std::abs(sample));
    }

    // 99.5th percentile absolute value.
    const size_t p_idx = static_cast<size_t>(0.995f * static_cast<float>(abs_vals.size() - 1));
    std::nth_element(abs_vals.begin(), abs_vals.begin() + p_idx, abs_vals.end());
    float peak = abs_vals[p_idx];
    if (peak < 1e-8f) {
        peak = *std::max_element(abs_vals.begin(), abs_vals.end());
    }
    if (peak < 1e-8f) return audio;

    float scale = target_peak / peak;
    std::vector<float> result(audio.size());
    
    for (size_t i = 0; i < audio.size(); i++) {
        result[i] = std::clamp(ac_audio[i] * scale, -1.0f, 1.0f);
    }
    
    return result;
}

std::vector<float> DSP::apply_fades(const std::vector<float>& audio, int fade_samples) {
    if (audio.size() < static_cast<size_t>(fade_samples * 2)) {
        return audio;
    }
    
    std::vector<float> result = audio;
    
    // Fade in
    for (int i = 0; i < fade_samples; i++) {
        float factor = static_cast<float>(i) / fade_samples;
        result[i] *= factor;
    }
    
    // Fade out
    int end = static_cast<int>(result.size());
    for (int i = 0; i < fade_samples; i++) {
        float factor = static_cast<float>(i) / fade_samples;
        result[end - 1 - i] *= factor;
    }
    
    return result;
}

std::vector<int16_t> DSP::float_to_pcm16(const std::vector<float>& audio) {
    std::vector<int16_t> pcm(audio.size());
    
    for (size_t i = 0; i < audio.size(); i++) {
        float sample = std::clamp(audio[i], -1.0f, 1.0f);
        pcm[i] = static_cast<int16_t>(sample * 32767.0f);
    }
    
    return pcm;
}

// WAV Writer

bool WavWriter::write(const std::string& path,
                      const std::vector<float>& audio,
                      int sample_rate,
                      int channels) {
    DSP dsp;
    auto pcm = dsp.float_to_pcm16(audio);
    return write_pcm16(path, pcm, sample_rate, channels);
}

bool WavWriter::write_pcm16(const std::string& path,
                             const std::vector<int16_t>& audio,
                             int sample_rate,
                             int channels) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    
    WavHeader header;
    uint32_t data_size = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
    
    header.file_size = data_size + sizeof(WavHeader) - 8;
    header.num_channels = static_cast<uint16_t>(channels);
    header.sample_rate = static_cast<uint32_t>(sample_rate);
    header.bits_per_sample = 16;
    header.block_align = static_cast<uint16_t>(channels * 2);
    header.byte_rate = sample_rate * channels * 2;
    header.data_size = data_size;
    
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(audio.data()), data_size);
    
    return file.good();
}

} // namespace heartbeat
