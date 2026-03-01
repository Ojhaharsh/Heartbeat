#!/usr/bin/env python3
"""
Export Kokoro-82M PyTorch model to GGUF format.

This script converts the Kokoro TTS model weights from PyTorch (.pth)
to GGUF format for use with the Heartbeat native inference engine.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    from gguf import GGUFWriter
    import gguf
except ImportError:
    print("GGUF library is required. Install with: pip install gguf")
    sys.exit(1)


# =============================================================================
# GGUF Tensor Type Constants
# =============================================================================
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8


# =============================================================================
# Tensor Name Mapping
# =============================================================================
def map_tensor_name(pytorch_name: str) -> str:
    """Map PyTorch tensor names to GGUF conventions."""
    
    # BERT encoder mappings
    mappings = {
        "bert.embeddings.word_embeddings.weight": "text_encoder.emb.weight",
        "bert.embeddings.position_embeddings.weight": "text_encoder.pos_emb.weight",
        "bert.embeddings.token_type_embeddings.weight": "text_encoder.token_type_emb.weight",
        "bert.embeddings.LayerNorm.weight": "text_encoder.embed_ln.weight",
        "bert.embeddings.LayerNorm.bias": "text_encoder.embed_ln.bias",
    }
    
    if pytorch_name in mappings:
        return mappings[pytorch_name]
    
    # Encoder layer mappings
    if "bert.encoder.layer." in pytorch_name:
        # Extract layer number
        parts = pytorch_name.split(".")
        layer_idx = parts[3]
        rest = ".".join(parts[4:])
        
        layer_mappings = {
            "attention.self.query.weight": "attention.q.weight",
            "attention.self.query.bias": "attention.q.bias",
            "attention.self.key.weight": "attention.k.weight",
            "attention.self.key.bias": "attention.k.bias",
            "attention.self.value.weight": "attention.v.weight",
            "attention.self.value.bias": "attention.v.bias",
            "attention.output.dense.weight": "attention.o.weight",
            "attention.output.dense.bias": "attention.o.bias",
            "attention.output.LayerNorm.weight": "ln1.weight",
            "attention.output.LayerNorm.bias": "ln1.bias",
            "intermediate.dense.weight": "ffn.up.weight",
            "intermediate.dense.bias": "ffn.up.bias",
            "output.dense.weight": "ffn.down.weight",
            "output.dense.bias": "ffn.down.bias",
            "output.LayerNorm.weight": "ln2.weight",
            "output.LayerNorm.bias": "ln2.bias",
        }
        
        new_rest = layer_mappings.get(rest, rest)
        return f"text_encoder.layer.{layer_idx}.{new_rest}"
    
    # Style encoder mappings
    if pytorch_name.startswith("style_encoder"):
        return pytorch_name
    
    # Decoder mappings
    if pytorch_name.startswith("decoder"):
        return pytorch_name
    
    # Duration predictor
    if pytorch_name.startswith("duration_predictor"):
        return pytorch_name
    
    # Default: keep original name
    return pytorch_name


def should_transpose(name: str) -> bool:
    """Check if tensor should be transposed for GGML matmul."""
    # Linear layer weights need transposition
    # if "weight" in name and any(x in name for x in [".q.", ".k.", ".v.", ".o.", ".ffn.", ".proj.", ".dense."]):
    #     return True
    return False


# =============================================================================
# Export Functions
# =============================================================================
def compute_checksum(data: bytes) -> str:
    """Compute SHA256 checksum of tensor data."""
    return hashlib.sha256(data).hexdigest()


def get_tensor_type(name: str, dtype: torch.dtype, quantize: bool = False) -> int:
    """Determine GGUF tensor type."""
    if quantize and "weight" in name and "LayerNorm" not in name and "ln" not in name:
        # Quantize large weight matrices
        return GGML_TYPE_Q8_0
    
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return GGML_TYPE_F16
    
    return GGML_TYPE_F32


def export_kokoro(
    input_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    quantize: bool = False,
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Export Kokoro model to GGUF format.
    
    Args:
        input_path: Path to kokoro-v0_19.pth
        output_path: Output GGUF file path
        config_path: Optional path to config.json
        quantize: Whether to quantize weights
        verify: Compute and store checksums for verification
    
    Returns:
        Dictionary with export statistics
    """
    print("=" * 60)
    print("Kokoro-82M GGUF Exporter")
    print("=" * 60)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Model not found: {input_path}")
    
    # Load PyTorch checkpoint
    print(f"\n[INFO] Loading: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # Handle Kokoro v1.0 nested format: {'bert': {...}, 'predictor': {...}, 'decoder': {...}, ...}
    state_dict = {}
    
    if isinstance(checkpoint, dict):
        # Check if this is the nested format with component sub-dicts
        component_names = ['bert', 'bert_encoder', 'predictor', 'decoder', 'text_encoder']
        is_nested = any(k in checkpoint for k in component_names)
        
        if is_nested:
            # Flatten nested structure
            for component, tensors in checkpoint.items():
                if isinstance(tensors, dict):
                    for name, tensor in tensors.items():
                        # Remove 'module.' prefix if present
                        clean_name = name.replace("module.", "")
                        full_name = f"{component}.{clean_name}"
                        state_dict[full_name] = tensor
                else:
                    state_dict[component] = tensors
            print(f"   Found nested format with {len(checkpoint)} components")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint
    
    print(f"   Loaded {len(state_dict)} tensors")
    
    # Load config if available
    config = {}
    if config_path:
        config_file = Path(config_path)
    else:
        config_file = input_path.parent / "config.json"
    
    if config_file.exists():
        print(f"[INFO] Loading config: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
    
    # Extract hyperparameters (supports both legacy and Kokoro v1.x config layouts)
    plbert_cfg = config.get("plbert", {})
    istft_cfg = config.get("istftnet", config.get("gen", {}))

    vocab_size = int(config.get("n_token", config.get("vocab_size", 178)))
    hidden_size = int(plbert_cfg.get("hidden_size", config.get("hidden_size", 768)))
    num_layers = int(plbert_cfg.get("num_hidden_layers", config.get("num_hidden_layers", 12)))
    num_heads = int(plbert_cfg.get("num_attention_heads", config.get("num_attention_heads", 12)))
    intermediate_size = int(plbert_cfg.get("intermediate_size", config.get("intermediate_size", 3072)))
    style_dim = int(config.get("style_dim", 128))
    n_mels = int(config.get("n_mels", 80))
    sample_rate = int(config.get("sample_rate", 24000))

    # ISTFT parameters
    istft_n_fft = int(istft_cfg.get("gen_istft_n_fft", istft_cfg.get("istft_n_fft", 16)))
    istft_hop_length = int(istft_cfg.get("gen_istft_hop_size", istft_cfg.get("istft_hop_length", 4)))
    
    print(f"\n[INFO] Model Configuration:")
    print(f"   Vocab Size:    {vocab_size}")
    print(f"   Hidden Size:   {hidden_size}")
    print(f"   Layers:        {num_layers}")
    print(f"   Heads:         {num_heads}")
    print(f"   Style Dim:     {style_dim}")
    print(f"   ISTFT n_fft:   {istft_n_fft}")
    print(f"   ISTFT hop:     {istft_hop_length}")
    print(f"   Sample Rate:   {sample_rate} Hz")
    
    # Create GGUF writer
    print(f"\n[INFO] Writing: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = GGUFWriter(str(output_path), "kokoro")
    
    # Write metadata
    writer.add_name("Kokoro-82M")
    writer.add_description("Native Kokoro TTS model for Heartbeat engine")
    writer.add_author("hexgrad")
    
    # Write hyperparameters
    writer.add_uint32("kokoro.vocab_size", vocab_size)
    writer.add_uint32("kokoro.hidden_size", hidden_size)
    writer.add_uint32("kokoro.num_layers", num_layers)
    writer.add_uint32("kokoro.num_heads", num_heads)
    writer.add_uint32("kokoro.intermediate_size", intermediate_size)
    writer.add_uint32("kokoro.style_dim", style_dim)
    writer.add_uint32("kokoro.n_mels", n_mels)
    writer.add_uint32("kokoro.sample_rate", sample_rate)
    writer.add_uint32("kokoro.istft_n_fft", istft_n_fft)
    writer.add_uint32("kokoro.istft_hop_length", istft_hop_length)
    
    # Write vocabulary if available
    vocab_file = input_path.parent / "vocab.json"
    tokens = None
    if vocab_file.exists():
        with open(vocab_file, encoding="utf-8") as f:
            vocab = json.load(f)
        if isinstance(vocab, dict):
            # If vocab is token -> id map, place tokens by ID.
            if all(isinstance(v, (int, float)) for v in vocab.values()):
                max_id = max(int(v) for v in vocab.values()) if vocab else -1
                token_count = max(vocab_size, max_id + 1)
                tokens = [f"<unused_{i}>" for i in range(token_count)]
                for token, idx in vocab.items():
                    idx = int(idx)
                    if 0 <= idx < token_count:
                        tokens[idx] = str(token)
            else:
                tokens = [str(k) for k in vocab.keys()]
        else:
            tokens = [str(t) for t in vocab]
    elif isinstance(config.get("vocab"), dict):
        vocab_map = config["vocab"]
        max_id = max(int(v) for v in vocab_map.values()) if vocab_map else -1
        token_count = max(vocab_size, max_id + 1)
        tokens = [f"<unused_{i}>" for i in range(token_count)]
        for token, idx in vocab_map.items():
            idx = int(idx)
            if 0 <= idx < token_count:
                tokens[idx] = str(token)

    if tokens:
        writer.add_token_list(tokens)
        print(f"   Added {len(tokens)} vocabulary tokens")
    
    # Export tensors
    print(f"\n[INFO] Exporting tensors...")
    
    tensor_count = 0
    total_bytes = 0
    checksums = {}
    
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        
        # Map name
        gguf_name = map_tensor_name(name)
        
        # Convert to numpy
        data = tensor.detach().float().numpy()
        
        # Transpose if needed
        if should_transpose(name) and len(data.shape) == 2:
            data = data.T
        
        # Determine type
        tensor_type = get_tensor_type(gguf_name, tensor.dtype, quantize)
        
        # Convert to appropriate type
        if tensor_type == GGML_TYPE_F16:
            data = data.astype(np.float16)
        elif tensor_type == GGML_TYPE_Q8_0:
            # For Q8_0, we need special handling
            # For now, just use F32
            tensor_type = GGML_TYPE_F32
            data = data.astype(np.float32)
        else:
            data = data.astype(np.float32)
        
        # Add tensor
        writer.add_tensor(gguf_name, data)
        
        # Compute checksum
        if verify:
            checksums[gguf_name] = compute_checksum(data.tobytes())
        
        tensor_count += 1
        total_bytes += data.nbytes
        
        if tensor_count % 50 == 0:
            print(f"   Processed {tensor_count} tensors...")
    
    # Finalize
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    output_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n[INFO] Export complete!")
    print(f"   Tensors:     {tensor_count}")
    print(f"   Total data:  {total_bytes / (1024 * 1024):.1f} MB")
    print(f"   Output file: {output_size:.1f} MB")
    
    # Save checksums
    if verify:
        checksum_path = output_path.with_suffix(".checksums.json")
        with open(checksum_path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)
        print(f"   Checksums:   {checksum_path}")
    
    return {
        "tensor_count": tensor_count,
        "total_bytes": total_bytes,
        "output_size": output_size,
        "checksums": checksums if verify else None,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Export Kokoro-82M model to GGUF format"
    )
    parser.add_argument(
        "--input", "-i",
        default="models/kokoro-v1_0.pth",
        help="Input PyTorch model path"
    )
    parser.add_argument(
        "--output", "-o",
        default="models/kokoro.gguf",
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to config.json (optional)"
    )
    parser.add_argument(
        "--quantize", "-q",
        action="store_true",
        help="Quantize weights to reduce size"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Generate checksums for verification"
    )
    
    args = parser.parse_args()
    
    try:
        stats = export_kokoro(
            input_path=args.input,
            output_path=args.output,
            config_path=args.config,
            quantize=args.quantize,
            verify=args.verify,
        )
        print("\n[SUCCESS] Export complete!")
        return 0
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
