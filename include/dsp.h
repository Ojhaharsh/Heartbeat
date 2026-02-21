#pragma once
#ifndef DSP_H
#define DSP_H

#include <vector>
#include <complex>
#include <cstdint>

namespace heartbeat {

/**
 * Audio processing configuration.
 */
struct DSPConfig {
    int n_fft = 16;             // FFT size
    int hop_length = 4;         // Hop length for OLA
    int sample_rate = 24000;    // Output sample rate
    float pre_emphasis = 0.0f;  // Pre-emphasis coefficient
};

/**
 * Digital Signal Processing module for audio synthesis.
 * Implements Inverse Short-Time Fourier Transform (ISTFT)
 * for converting neural network output to audio waveforms.
 */
class DSP {
public:
    explicit DSP(const DSPConfig& config = {});
    ~DSP();
    
    // Disable copy
    DSP(const DSP&) = delete;
    DSP& operator=(const DSP&) = delete;
    
    // Enable move
    DSP(DSP&& other) noexcept;
    DSP& operator=(DSP&& other) noexcept;
    
    /**
     * Perform ISTFT to synthesize audio from magnitude and phase.
     * @param magnitude Magnitude spectrogram [n_mels, n_frames].
     * @param phase Phase spectrogram [n_mels, n_frames].
     * @param n_mels Number of mel bins.
     * @param n_frames Number of time frames.
     * @return Audio samples normalized to [-1, 1].
     */
    std::vector<float> istft(const std::vector<float>& magnitude,
                              const std::vector<float>& phase,
                              int n_mels,
                              int n_frames);
    
    /**
     * Perform ISTFT with complex spectrogram input.
     * @param spectrogram Complex spectrogram [n_fft, n_frames].
     * @param n_frames Number of time frames.
     * @return Audio samples.
     */
    std::vector<float> istft(const std::vector<std::complex<float>>& spectrogram,
                              int n_frames);
    
    /**
     * Apply synthesis window (Hanning).
     * @param signal Input signal.
     * @param window_size Size of window.
     * @return Windowed signal.
     */
    std::vector<float> apply_window(const std::vector<float>& signal,
                                     int window_size);
    
    /**
     * Perform overlap-add synthesis.
     * @param frames Vector of overlapping frames.
     * @param hop_length Hop length between frames.
     * @return Reconstructed signal.
     */
    std::vector<float> overlap_add(const std::vector<std::vector<float>>& frames,
                                    int hop_length);
    
    /**
     * Normalize audio to target peak level.
     * @param audio Input audio samples.
     * @param target_peak Target peak level (0.0 to 1.0).
     * @return Normalized audio.
     */
    std::vector<float> normalize(const std::vector<float>& audio,
                                  float target_peak = 0.95f);
    
    /**
     * Apply fade in/out to avoid clicks.
     * @param audio Audio samples.
     * @param fade_samples Number of samples for fade.
     * @return Audio with fades applied.
     */
    std::vector<float> apply_fades(const std::vector<float>& audio,
                                    int fade_samples = 256);
    
    /**
     * Convert float samples to 16-bit PCM.
     * @param audio Float audio samples [-1, 1].
     * @return 16-bit PCM samples.
     */
    std::vector<int16_t> float_to_pcm16(const std::vector<float>& audio);
    
    /**
     * Get Hanning window.
     * @param size Window size.
     * @return Hanning window coefficients.
     */
    std::vector<float> hanning_window(int size);
    
    /**
     * Get current configuration.
     */
    const DSPConfig& config() const { return config_; }

private:
    void init_fft();
    void cleanup_fft();
    
    void ifft(const std::complex<float>* in, 
              std::complex<float>* out, 
              int n);
    
    DSPConfig config_;
    std::vector<float> window_;
    
    // KissFFT state (opaque pointer)
    void* fft_cfg_ = nullptr;
};

/**
 * WAV file writer.
 */
class WavWriter {
public:
    /**
     * Write audio to WAV file.
     * @param path Output file path.
     * @param audio Audio samples [-1, 1].
     * @param sample_rate Sample rate in Hz.
     * @param channels Number of channels.
     * @return true if successful.
     */
    static bool write(const std::string& path,
                      const std::vector<float>& audio,
                      int sample_rate = 24000,
                      int channels = 1);
    
    /**
     * Write PCM16 audio to WAV file.
     */
    static bool write_pcm16(const std::string& path,
                            const std::vector<int16_t>& audio,
                            int sample_rate = 24000,
                            int channels = 1);
};

/**
 * WAV file header structure.
 */
#pragma pack(push, 1)
struct WavHeader {
    char riff_tag[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size;
    char wave_tag[4] = {'W', 'A', 'V', 'E'};
    char fmt_tag[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;  // PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data_tag[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};
#pragma pack(pop)

} // namespace heartbeat

#endif // DSP_H
