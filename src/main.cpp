/**
 * Heartbeat CLI - Native Kokoro-82M TTS Inference
 * 
 * Usage:
 *   heartbeat --text "Hello, world!" --output hello.wav
 *   heartbeat --text "Welcome!" --voice in_f --output welcome.wav
 *   heartbeat --benchmark --text "Performance test."
 */

#include "heartbeat.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

void print_usage() {
    std::cout << R"(
 _   _                 _   _                _   
| | | | ___  __ _ _ __| |_| |__   ___  __ _| |_ 
| |_| |/ _ \/ _` | '__| __| '_ \ / _ \/ _` | __|
|  _  |  __/ (_| | |  | |_| |_) |  __/ (_| | |_ 
|_| |_|\___|\__,_|_|   \__|_.__/ \___|\__,_|\__|

Native Kokoro-82M TTS Engine

USAGE:
    heartbeat [OPTIONS]

OPTIONS:
    --model <path>      Path to GGUF model (default: models/kokoro.gguf)
    --text <text>       Text to synthesize (required)
    --output <path>     Output WAV file (default: output.wav)
    --voice <code>      Voice code (default: af)
                        Available: af, am, bf, bm, in_f, in_m
    --list-voices       List available voices
    --benchmark         Run in benchmark mode
    --info              Show model information
    --help              Show this help message

EXAMPLES:
    heartbeat --text "Hello, world!"
    heartbeat --text "Welcome to India!" --voice in_f --output welcome.wav
    heartbeat --benchmark --text "The quick brown fox."

)";
}

void print_voices() {
    std::cout << "\nAvailable Voices:\n";
    std::cout << "  Code    Description\n";
    std::cout << "  ----    -----------\n";
    std::cout << "  af      American Female\n";
    std::cout << "  am      American Male\n";
    std::cout << "  bf      British Female\n";
    std::cout << "  bm      British Male\n";
    std::cout << "  in_f    Indian Female\n";
    std::cout << "  in_m    Indian Male\n";
    std::cout << "\n";
}

struct Args {
    std::string model_path = "models/kokoro.gguf";
    std::string text;
    std::string output_path = "output.wav";
    std::string voice = "af";
    bool benchmark = false;
    bool list_voices = false;
    bool show_info = false;
    bool show_help = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            args.show_help = true;
        } else if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--text" && i + 1 < argc) {
            args.text = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (arg == "--voice" && i + 1 < argc) {
            args.voice = argv[++i];
        } else if (arg == "--benchmark") {
            args.benchmark = true;
        } else if (arg == "--list-voices") {
            args.list_voices = true;
        } else if (arg == "--info") {
            args.show_info = true;
        }
    }
    
    return args;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    
    if (args.show_help || (args.text.empty() && !args.list_voices && !args.show_info)) {
        print_usage();
        return 0;
    }
    
    if (args.list_voices) {
        print_voices();
        return 0;
    }
    
    // Check model file exists
    if (!fs::exists(args.model_path)) {
        std::cerr << "Error: Model file not found: " << args.model_path << "\n";
        std::cerr << "\nTo download the model, run:\n";
        std::cerr << "  python scripts/download_model.py\n";
        std::cerr << "  python scripts/export_kokoro.py\n";
        return 1;
    }
    
    try {
        std::cout << "Loading model: " << args.model_path << "\n";
        
        heartbeat::Heartbeat hb(args.model_path);
        
        if (args.show_info) {
            std::cout << "\n" << hb.model_info();
            std::cout << "\nPhonemizer: " << (hb.has_phonemizer() ? "Available" : "Not available") << "\n";
            return 0;
        }
        
        std::cout << "Voice: " << args.voice << "\n";
        std::cout << "Text: \"" << args.text << "\"\n";
        std::cout << "\nSynthesizing...\n";
        
        auto result = hb.synthesize(args.text, args.voice);
        
        if (!result.success) {
            std::cerr << "Synthesis failed: " << result.error_message << "\n";
            return 1;
        }
        
        std::cout << "\n";
        std::cout << "Duration:  " << result.duration_seconds << " seconds\n";
        std::cout << "Inference: " << result.inference_time_ms << " ms\n";
        
        if (args.benchmark) {
            // Calculate real-time factor
            float rtf = static_cast<float>(result.inference_time_ms) / 
                       (result.duration_seconds * 1000.0f);
            std::cout << "RTF:       " << rtf << "x\n";
            
            // Run multiple iterations for benchmarking
            const int iterations = 5;
            int64_t total_time = 0;
            
            std::cout << "\nRunning " << iterations << " benchmark iterations...\n";
            
            for (int i = 0; i < iterations; i++) {
                auto start = std::chrono::high_resolution_clock::now();
                auto bench_result = hb.synthesize(args.text, args.voice);
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                total_time += duration.count();
                
                std::cout << "  Iteration " << (i + 1) << ": " << duration.count() << " ms\n";
            }
            
            float avg_time = static_cast<float>(total_time) / iterations;
            float avg_rtf = avg_time / (result.duration_seconds * 1000.0f);
            
            std::cout << "\nAverage: " << avg_time << " ms (RTF: " << avg_rtf << "x)\n";
        } else {
            // Write output
            if (hb.write_wav(args.output_path, result)) {
                std::cout << "\nSaved to: " << args.output_path << "\n";
            } else {
                std::cerr << "Failed to write WAV file\n";
                return 1;
            }
        }
        
        std::cout << "\n✓ Done!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
