import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import os


# Импортируем настройки
from config import settings

def record_audio(filename=settings.AUDIO_TEMP_FILE,
                 duration=settings.AUDIO_RECORD_DURATION,
                 sample_rate=settings.AUDIO_SAMPLE_RATE,
                 channels=settings.AUDIO_CHANNELS):
    """Записывает аудио с микрофона указанной длительности."""
    print(f"Нажмите Enter и говорите ~{duration} секунд...")
    input() # Ждем нажатия Enter
    print("Запись...")
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
        sd.wait()
        # Конвертация и сохранение
        recording_int16 = (recording * 32767).astype(np.int16)
        wav.write(filename, sample_rate, recording_int16)
        print(f"Запись сохранена в {filename}")
        return filename # Возвращаем имя файла для дальнейшей обработки
    except Exception as e:
        print(f"Ошибка записи звука: {e}")
        return None

def play_audio(filename):
    try:
        samplerate, data = wav.read(filename)
        sd.play(data, samplerate)
        sd.wait()
        print(f"Воспроизведение {filename} завершено.")
    except Exception as e:
        print(f"Ошибка воспроизведения файла {filename}: {e}")

def remove_temp_audio(filename=settings.AUDIO_TEMP_FILE):
    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"Временный файл {filename} удален.")
        except Exception as e:
            print(f"Не удалось удалить временный файл {filename}: {e}")