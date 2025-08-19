import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."

# Default settings
wav_default = model.generate(text, exaggeration=0.5, cfg_weight=0.5)
ta.save("test-default.wav", wav_default, model.sr)

# More expressive
wav_expressive = model.generate(text, exaggeration=0.8, cfg_weight=0.3)
ta.save("test-expressive.wav", wav_expressive, model.sr)

# Less expressive
wav_subtle = model.generate(text, exaggeration=0.2, cfg_weight=0.7)
ta.save("test-subtle.wav", wav_subtle, model.sr)

print("Generated audio files with different emotion settings: test-default.wav, test-expressive.wav, test-subtle.wav")