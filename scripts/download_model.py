#!/usr/bin/env python3
"""
Download Kokoro-82M model files from Hugging Face.
"""

import os
import sys
import hashlib
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "hexgrad/Kokoro-82M"
MODEL_DIR = Path(__file__).parent.parent / "models"

# Core model files
CORE_FILES = [
    "kokoro-v1_0.pth",  # Main model weights (327 MB)
    "config.json",       # Model configuration
]

# Voice files with correct names from HuggingFace
VOICE_FILES = [
    # American English
    ("voices/af_bella.pt", "American Female (Bella)"),
    ("voices/af_heart.pt", "American Female (Heart)"),
    ("voices/am_adam.pt", "American Male (Adam)"),
    ("voices/am_michael.pt", "American Male (Michael)"),
    # British English
    ("voices/bf_emma.pt", "British Female (Emma)"),
    ("voices/bf_alice.pt", "British Female (Alice)"),
    ("voices/bm_george.pt", "British Male (George)"),
    ("voices/bm_daniel.pt", "British Male (Daniel)"),
    # Indian English
    ("voices/if_sara.pt", "Indian Female (Sara)"),
    ("voices/im_nicola.pt", "Indian Male (Nicola)"),
]


def download_file(filename: str) -> Path:
    """Download a single file from Hugging Face."""
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=MODEL_DIR,
    )
    return Path(local_path)


def get_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    print("=" * 60)
    print("Kokoro-82M Model Downloader")
    print("=" * 60)
    
    # Create models directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nModel directory: {MODEL_DIR.absolute()}")
    
    downloaded = []
    failed = []
    
    # Download core files
    print("\n[1/2] Downloading core model files...")
    for filename in CORE_FILES:
        try:
            print(f"  Downloading {filename}...")
            local_path = download_file(filename)
            file_size = local_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {filename} ({file_size:.1f} MB)")
            downloaded.append(filename)
        except Exception as e:
            print(f"  [FAIL] {filename}: {e}")
            failed.append(filename)
    
    # Download voice files
    print("\n[2/2] Downloading voice files...")
    for filename, description in VOICE_FILES:
        try:
            print(f"  Downloading {description}...")
            local_path = download_file(filename)
            file_size = local_path.stat().st_size / 1024
            print(f"  [OK] {filename} ({file_size:.0f} KB)")
            downloaded.append(filename)
        except Exception as e:
            print(f"  [FAIL] {filename}: {e}")
            failed.append(filename)
    
    # Summary
    print("\n" + "=" * 60)
    total = len(CORE_FILES) + len(VOICE_FILES)
    print(f"Downloaded: {len(downloaded)}/{total} files")
    
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    
    # Check if we can proceed
    core_ok = all(f in downloaded for f in CORE_FILES)
    
    if core_ok:
        print("\n[OK] Core model ready! Run:")
        print("   python scripts/export_kokoro.py")
        
        # List available voices
        voices = [v[0] for v in VOICE_FILES if v[0] in downloaded]
        if voices:
            print(f"\nVoices downloaded: {len(voices)}")
            for v in voices:
                name = v.replace("voices/", "").replace(".pt", "")
                print(f"  - {name}")
    else:
        print("\n[!] Core model files missing. Cannot proceed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
