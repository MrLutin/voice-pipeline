#!/usr/bin/env python3
"""
Test XTTS v2 with tuned parameters for more natural voice.
"""

import argparse
import os
import torch
from TTS.api import TTS

def main():
    parser = argparse.ArgumentParser(description="Test XTTS v2 with tuned params")
    parser.add_argument("--voice", "-v", required=True, help="Voice sample WAV file")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    parser.add_argument("--output", "-o", default="output/tuned_output.wav", help="Output file")
    parser.add_argument("--language", "-l", default="fr", help="Language code")
    parser.add_argument("--temperature", type=float, default=0.65, help="Temperature (0.1-1.0, lower=more stable)")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty")
    parser.add_argument("--repetition-penalty", type=float, default=2.0, help="Repetition penalty")
    parser.add_argument("--top-k", type=int, default=50, help="Top K sampling")
    parser.add_argument("--top-p", type=float, default=0.85, help="Top P sampling")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    
    print("📥 Loading XTTS v2...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"🔊 Synthesizing with tuned params:")
    print(f"   Temperature: {args.temperature}")
    print(f"   Length penalty: {args.length_penalty}")
    print(f"   Repetition penalty: {args.repetition_penalty}")
    print(f"   Top K: {args.top_k}, Top P: {args.top_p}")
    print(f"   Speed: {args.speed}")
    
    tts.tts_to_file(
        text=args.text,
        speaker_wav=args.voice,
        language=args.language,
        file_path=args.output,
        temperature=args.temperature,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
        speed=args.speed
    )
    
    print(f"✅ Saved: {args.output}")

if __name__ == "__main__":
    main()
