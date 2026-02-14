# Setup Instructions - Voice Pipeline

## ⚠️ Dépendance Known Issue

XTTS v2 a une incompatibilité avec certaines versions de `transformers`.

### Solution rapide (Recommandée)

```bash
# 1. Créer un environnement virtuel propre
python3 -m venv voice_env
source voice_env/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Downgrade transformers si nécessaire
pip install transformers==4.35.0
```

### Tester après installation

```bash
# Test simple TTS avec JARVIS
python3 test_tts_only.py \
  --voice samples/morlutin_voice.wav \
  --text "Bonjour, c'est Morwintar qui parle"
```

## Voice Samples

- **morlutin_voice.wav** - Convertie de l'audio Telegram (12 sec)
- **JARVIS.mp3** - Original JARVIS (pour référence)

## Pipeline complet

Une fois les dépendances résolues:

```bash
python3 voice_pipeline.py \
  --text "Salut, comment ça va?" \
  --voice samples/morlutin_voice.wav \
  --output output/response.wav
```

## Troubleshooting

### ImportError: cannot import name 'isin_mps_friendly'

**Cause:** Incompatibilité transformers/coqui-tts

**Fix:**
```bash
pip install transformers==4.35.0
```

Ou mettre à jour coqui-tts:
```bash
pip install --upgrade coqui-tts
```

### CUDA out of memory

```bash
python3 voice_pipeline.py --device cpu ...
```

(Plus lent, mais utilise seulement le CPU)

---

Le projet est prêt une fois qu'on a résolu la dépendance! 🖤
