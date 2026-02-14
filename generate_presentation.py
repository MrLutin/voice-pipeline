#!/usr/bin/env python3
"""
Generate presentation by processing sentences separately and concatenating.
"""

import os
import torch
import numpy as np
import soundfile as sf
from TTS.api import TTS

# Sentences to synthesize
sentences = [
    "Bonjour, je suis Morwintar, votre assistant IA personnel.",
    "Je peux coder en Python, Java, PHP et JavaScript.",
    "Je gère vos serveurs, automatise vos tâches, et communique sur Discord, Telegram et Twitter.",
    "Je génère des images, clone des voix, et recherche sur le web.",
    "Je suis autonome, j'ai ma propre personnalité, et j'apprends constamment.",
    "Ensemble, nous pouvons accomplir l'impossible.",
    "À votre service, Monsieur Lutin."
]

voice_sample = "samples/morlutin_voice.wav"
output_file = "output/presentation_smooth.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device}")

print("📥 Loading XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

os.makedirs("output", exist_ok=True)

all_audio = []
sample_rate = None

for i, sentence in enumerate(sentences):
    print(f"🔊 [{i+1}/{len(sentences)}] {sentence[:50]}...")
    
    temp_file = f"output/temp_{i}.wav"
    tts.tts_to_file(
        text=sentence,
        speaker_wav=voice_sample,
        language="fr",
        file_path=temp_file,
        temperature=0.5,
        top_p=0.8,
        speed=1.0
    )
    
    audio, sr = sf.read(temp_file)
    if sample_rate is None:
        sample_rate = sr
    all_audio.append(audio)
    
    # Add small pause between sentences
    pause = np.zeros(int(sr * 0.3))  # 0.3 second pause
    all_audio.append(pause)
    
    os.remove(temp_file)

# Concatenate all audio
final_audio = np.concatenate(all_audio)
sf.write(output_file, final_audio, sample_rate)

print(f"✅ Saved: {output_file}")
