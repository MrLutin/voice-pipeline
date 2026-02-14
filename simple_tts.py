#!/usr/bin/env python3
"""
Simple TTS using Piper (no voice cloning, but more stable)
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from piper.voice import PiperVoice
except ImportError:
    print("❌ piper-tts not installed. Run: pip install piper-tts")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Simple Piper TTS")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    parser.add_argument("--output", "-o", default="output/output.wav", help="Output file")
    parser.add_argument("--voice", "-v", default="fr_FR-gilles-medium", help="Piper voice model")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"🔊 Loading Piper voice: {args.voice}")
    voice = PiperVoice.load(args.voice)
    
    print(f"🗣️ Synthesizing: '{args.text}'")
    with open(args.output, "wb") as f:
        voice.synthesize(args.text, f)
    
    print(f"✅ Saved: {args.output}")

if __name__ == "__main__":
    main()
