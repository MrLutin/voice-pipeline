#!/usr/bin/env python3
"""
Test XTTS v2 voice cloning only (without Whisper/Claude).
Useful for testing your voice sample.

Usage:
    python test_tts_only.py --voice samples/my_voice.wav --text "Bonjour!"
"""

import argparse
import os
import torch
from TTS.api import TTS

def main():
    parser = argparse.ArgumentParser(description="Test XTTS v2 voice cloning")
    parser.add_argument("--voice", "-v", required=True, help="Voice sample WAV file")
    parser.add_argument("--text", "-t", default="Bonjour, je suis une voix clonée!", help="Text to synthesize")
    parser.add_argument("--output", "-o", default="output/test_tts.wav", help="Output file")
    parser.add_argument("--language", "-l", default="fr", help="Language code")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    
    print("📥 Loading XTTS v2...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"🔊 Synthesizing: '{args.text}'")
    tts.tts_to_file(
        text=args.text,
        speaker_wav=args.voice,
        language=args.language,
        file_path=args.output
    )
    
    print(f"✅ Saved: {args.output}")

if __name__ == "__main__":
    main()
