# Voice Pipeline 🎤→🧠→🔊

Pipeline expérimental: **Whisper → Claude → XTTS v2**

Transforme de l'audio en réponse vocale avec une voix clonée personnalisée.

## 🏗️ Architecture

```
Audio Input → [Whisper STT] → Texte → [Claude AI] → Réponse → [XTTS v2 TTS] → Audio Output
                                                                    ↑
                                                            Voice Sample
```

## 📋 Prérequis

- **Python** 3.10+
- **GPU** recommandé (NVIDIA avec CUDA)
- **~6GB VRAM** pour XTTS v2
- **Clé API Anthropic** (`ANTHROPIC_API_KEY`)

## 🚀 Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 🎙️ Préparer le sample vocal

Place ton sample vocal dans `samples/`:
- Format: **WAV** (16kHz ou 22kHz recommandé)
- Durée: **6-30 secondes**
- Qualité: Audio propre, sans bruit de fond
- Contenu: Parole naturelle, phrases variées

## 📖 Utilisation

### Depuis un fichier audio
```bash
python voice_pipeline.py --input question.wav --voice samples/ma_voix.wav
```

### Depuis du texte (skip Whisper)
```bash
python voice_pipeline.py --text "Salut, comment ça va?" --voice samples/ma_voix.wav
```

### Enregistrer depuis le micro
```bash
python voice_pipeline.py --record --voice samples/ma_voix.wav --duration 5
```

### Options
```
--input, -i     Fichier audio d'entrée
--text, -t      Texte d'entrée (skip Whisper)
--voice, -v     Sample vocal pour le cloning (requis)
--output, -o    Fichier audio de sortie
--record, -r    Enregistrer depuis le micro
--duration, -d  Durée d'enregistrement (défaut: 5s)
--device        Force cuda ou cpu
```

## 📁 Structure

```
voice-pipeline/
├── voice_pipeline.py   # Script principal
├── requirements.txt    # Dépendances
├── samples/            # Samples vocaux pour cloning
│   └── (ton_sample.wav)
├── output/             # Fichiers générés
└── README.md
```

## ⚙️ Configuration

Édite `voice_pipeline.py` pour ajuster:

```python
WHISPER_MODEL = "base"  # tiny, base, small, medium, large
CLAUDE_MODEL = "claude-haiku-4-5"
LANGUAGE = "fr"
```

## 🐛 Troubleshooting

### "CUDA out of memory"
- Utilise un modèle Whisper plus petit (`tiny` ou `base`)
- Ajoute `--device cpu` (plus lent)

### Voix clonée de mauvaise qualité
- Utilise un sample plus long (15-30 sec)
- Assure-toi que l'audio est propre
- Évite la musique/bruit de fond dans le sample

### "No module named 'TTS'"
```bash
pip install coqui-tts
```

## 📜 License

Expérimental — Usage personnel uniquement.
XTTS v2 est sous [Coqui Public Model License](https://coqui.ai/cpml).

---

Créé par **Morwintar** 🖤
