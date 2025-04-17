from faster_whisper import WhisperModel
import time
from config import settings

class SpeechRecognizer:
    def __init__(self,
                 model_size=settings.STT_MODEL_SIZE,
                 device=settings.STT_DEVICE,
                 compute_type=settings.STT_COMPUTE_TYPE):
        print(f"Загрузка STT модели: {model_size} ({compute_type}) на {device}...")
        self._load_start_time = time.time()
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"STT модель загружена за {time.time() - self._load_start_time:.2f} сек.")
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить STT модель {model_size}. Ошибка: {e}")
            print("Убедитесь, что модель существует и зависимости установлены.")
            # В реальном приложении здесь лучше выбрасывать исключение
            self.model = None # Или exit()

    def transcribe(self, audio_path, language=settings.STT_DEFAULT_LANGUAGE, beam_size=settings.STT_BEAM_SIZE):
        """Преобразует аудиофайл в текст."""
        if not self.model or not audio_path:
            print("Ошибка: STT модель не инициализирована или не указан путь к аудио.")
            return None

        print("Распознавание речи...")
        try:
            start_time = time.time()
            segments, info = self.model.transcribe(audio_path, beam_size=beam_size, language=language)
            recognized_text = "".join(segment.text + " " for segment in segments).strip()
            end_time = time.time()

            detected_lang = info.language
            lang_prob = info.language_probability
            print(f"Распознано (язык: {detected_lang} {lang_prob:.2f}) за {end_time - start_time:.2f} сек: '{recognized_text}'")

            if not recognized_text:
                print("Ничего не удалось распознать.")
                return None
            return recognized_text
        except Exception as e:
            print(f"Ошибка распознавания речи: {e}")
            return None