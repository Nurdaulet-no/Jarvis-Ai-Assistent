
# STT (faster-whisper)
STT_MODEL_SIZE = "small"
STT_COMPUTE_TYPE = "int8"
STT_DEVICE = "cpu"     # 'cpu' or 'cuda'
STT_BEAM_SIZE = 5
STT_DEFAULT_LANGUAGE = None

# Audio Recording
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_RECORD_DURATION = 5
AUDIO_TEMP_FILE = "temp_user_query.wav"

# AI Core (Gemini)
AI_MODEL_NAME = 'gemini-2.5-pro-exp-03-25'

# TTS (pyttsx3)
TTS_RATE = 180 # Скорость речи
TTS_VOLUME = 1.0 # Громкость (0.0 to 1.0)
TTS_VOICE_LANG_HINT = 'en'

# Другие настройки
APP_NAME = "Jarvis Assistant"