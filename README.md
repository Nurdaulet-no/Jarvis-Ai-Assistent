---

## 🤖 **Jarvis Assistant: An Asynchronous Voice Assistant in Python**

This is your own personal J.A.R.V.I.S., built with Python! 🧠🎙️ Inspired by the Marvel universe, this voice-controlled assistant listens, thinks, and talks back using cutting-edge tools for speech and AI interaction. It’s fully asynchronous and responsive, designed to make conversations smooth and natural—even running on CPU! 💻⚡

---

### 🧩 **Key Technologies**

- 🗣️ **STT (Speech-to-Text):** `faster-whisper` (small model)  
- 🧠 **Brain:** Google Gemini API 🧬  
- 🔊 **TTS (Text-to-Speech):** `Coqui TTS` (`xtts_v2`) with **voice cloning** 🧍🗣️  
- 👂 **VAD (Voice Activity Detection):** Silero VAD 🎧  
- 🔄 **Architecture:** Fully **asynchronous** using `asyncio` ⏱️  
- 🎵 **Audio Handling:** `sounddevice`, `torchaudio` (resampling), `numpy` 🎚️

---

### 🚀 **Current Status**

✅ Fully functional on **CPU** (tested on Ryzen 5)  
⚡ Built with **asynchronous processing** for better responsiveness  
🎙️ Integrated **VAD** for smart listening  
🎛️ Supports **microphone resampling** for compatibility  
🧾 Splits long AI responses to fit **TTS length limits**  
🧑‍🎨 Easily customizable **AI personality** and **response style** via system prompts.

---
