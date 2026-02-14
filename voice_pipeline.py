#!/usr/bin/env python3
"""
Voice Pipeline: Whisper → Claude → XTTS v2
==========================================
Experimental voice-to-voice pipeline with custom voice cloning.

Usage:
    python voice_pipeline.py --input audio.wav
    python voice_pipeline.py --record
    python voice_pipeline.py --text "Salut!"
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Load config
CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)

CONFIG = load_config()

# Check dependencies
def check_deps():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import whisper
    except ImportError:
        missing.append("openai-whisper")
    try:
        from TTS.api import TTS
    except ImportError:
        missing.append("coqui-tts")
    try:
        import anthropic
    except ImportError:
        missing.append("anthropic")
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"Run: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import torch
import whisper
from TTS.api import TTS
import anthropic
import soundfile as sf
import numpy as np


class VoicePipeline:
    def __init__(self, voice_sample: str = None, device: str = None):
        """Initialize the voice pipeline.
        
        Args:
            voice_sample: Path to voice sample WAV (6-30 sec). Uses config default if None.
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.config = CONFIG
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Voice sample: argument > config
        if voice_sample:
            self.voice_sample = voice_sample
        else:
            self.voice_sample = str(Path(__file__).parent / self.config["tts"]["voice_sample"])
        
        print(f"🔧 Device: {self.device}")
        print(f"🎤 Voice sample: {self.voice_sample}")
        
        # Validate voice sample
        if not os.path.exists(self.voice_sample):
            raise FileNotFoundError(f"Voice sample not found: {self.voice_sample}")
        
        print("📥 Loading Whisper model...")
        whisper_model = self.config["whisper"]["model"]
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)
        
        print("📥 Loading XTTS v2 model...")
        tts_model = self.config["tts"]["model"]
        self.tts = TTS(tts_model).to(self.device)
        
        print("📥 Initializing Claude client...")
        self.claude = anthropic.Anthropic()
        
        print("✅ Pipeline ready!")
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper."""
        print("🎧 Transcribing audio...")
        language = self.config["whisper"]["language"]
        result = self.whisper_model.transcribe(audio_path, language=language)
        text = result["text"].strip()
        print(f"📝 Transcription: {text}")
        return text
    
    def generate_response(self, text: str, system_prompt: str = None) -> str:
        """Generate a response using Claude."""
        print("🧠 Generating response with Claude...")
        
        claude_config = self.config["claude"]
        prompt = system_prompt or claude_config["system_prompt"]
        
        message = self.claude.messages.create(
            model=claude_config["model"],
            max_tokens=claude_config["max_tokens"],
            system=prompt,
            messages=[{"role": "user", "content": text}]
        )
        
        response = message.content[0].text
        print(f"💬 Response: {response}")
        return response
    
    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize text to speech using XTTS v2 with voice cloning."""
        print("🔊 Synthesizing speech...")
        
        tts_config = self.config["tts"]
        language = tts_config["language"]
        
        self.tts.tts_to_file(
            text=text,
            speaker_wav=self.voice_sample,
            language=language,
            file_path=output_path
        )
        
        print(f"✅ Audio saved: {output_path}")
        return output_path
    
    def process(self, audio_path: str, output_path: str = None) -> str:
        """Run the full pipeline: Whisper → Claude → XTTS v2.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output audio (auto-generated if None)
            
        Returns:
            Path to the output audio file
        """
        if output_path is None:
            output_path = f"output/response_{Path(audio_path).stem}.wav"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Step 1: Transcribe
        text = self.transcribe(audio_path)
        
        # Step 2: Generate response
        response = self.generate_response(text)
        
        # Step 3: Synthesize
        self.synthesize(response, output_path)
        
        return output_path
    
    def text_to_voice(self, text: str, output_path: str = None) -> str:
        """Convert text directly to voice (skip Whisper).
        
        Args:
            text: Input text
            output_path: Path for output audio
            
        Returns:
            Path to the output audio file
        """
        if output_path is None:
            output_path = "output/response.wav"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Step 1: Generate response
        response = self.generate_response(text)
        
        # Step 2: Synthesize
        self.synthesize(response, output_path)
        
        return output_path


def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> str:
    """Record audio from microphone."""
    import sounddevice as sd
    
    print(f"🎤 Recording for {duration} seconds...")
    print("   Press Enter to start...")
    input()
    
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    
    # Save to temp file
    temp_path = tempfile.mktemp(suffix=".wav")
    sf.write(temp_path, audio, sample_rate)
    
    print(f"✅ Recorded to: {temp_path}")
    return temp_path


def main():
    parser = argparse.ArgumentParser(description="Voice Pipeline: Whisper → Claude → XTTS v2")
    parser.add_argument("--input", "-i", help="Input audio file")
    parser.add_argument("--text", "-t", help="Input text (skip Whisper)")
    parser.add_argument("--voice", "-v", help="Voice sample WAV file (default: from config)")
    parser.add_argument("--output", "-o", help="Output audio file")
    parser.add_argument("--record", "-r", action="store_true", help="Record from microphone")
    parser.add_argument("--duration", "-d", type=float, default=5.0, help="Recording duration (seconds)")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device (auto-detect if not set)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.text and not args.record:
        parser.error("Specify --input, --text, or --record")
    
    # Initialize pipeline
    pipeline = VoicePipeline(voice_sample=args.voice, device=args.device)
    
    # Process
    if args.text:
        output = pipeline.text_to_voice(args.text, args.output)
    elif args.record:
        audio_path = record_audio(duration=args.duration)
        output = pipeline.process(audio_path, args.output)
        os.unlink(audio_path)  # Cleanup temp file
    else:
        output = pipeline.process(args.input, args.output)
    
    print(f"\n🎉 Done! Output: {output}")


if __name__ == "__main__":
    main()
