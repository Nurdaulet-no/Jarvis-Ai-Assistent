# STT (faster-whisper)
STT_MODEL_SIZE = "small"
STT_COMPUTE_TYPE = "int8"
STT_DEVICE = "cpu"     # 'cpu' or 'cuda'
STT_BEAM_SIZE = 4
STT_DEFAULT_LANGUAGE = 'ru'

# Audio Recording & VAD
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_RECORD_DURATION = 5
AUDIO_TEMP_FILE = "temp_user_query.wav"
AUDIO_CHUNK_SIZE = 512
AUDIO_VAD_THRESHOLD = 0.2
AUDIO_VAD_SILENCE_TIMEOUT_MS = 2000
AUDIO_VAD_SPEECH_PAD_MS = 500

# VAD Model (Silero VAD)
VAD_MODEL_REPO = 'snakers4/silero-vad'
VAD_MODEL_NAME = 'silero_vad'
VAD_FORCE_DOWNLOAD = True

# AI Core (Gemini)
AI_MODEL_NAME = 'gemini-2.5-pro-exp-03-25'

# TTS (Coqui TTS)
TTS_SPEAKER_WAV = "resources/demo_speaker0.mp3"
TTS_STREAM_CHUNK_SIZE = 20
TTS_STREAM_PLAYBACK_BUFFER = 2

# Другие настройки
APP_NAME = "Jarvis Assistant"