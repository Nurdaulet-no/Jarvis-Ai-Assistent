import torch
import os
import sys
import time
import asyncio
import numpy as np
from collections import deque

from config import settings
from core import audio_handler


# --- Блок безопасности PyTorch для Coqui TTS ---
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.config import BaseAudioConfig
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseAudioConfig, BaseDatasetConfig])
    print("Необходимые классы Coqui TTS добавлены в безопасные глобальные объекты PyTorch.")
except ImportError as e:
    print(f"Предупреждение: Не удалось импортировать классы Coqui TTS для безопасной загрузки: {e}")
except AttributeError:
    print("Предупреждение: torch.serialization.add_safe_globals не найден. Возможно, используется старая версия PyTorch.")
# ---------------------------------------------



try:
    from TTS.api import TTS
except ImportError:
    print("КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать TTS из Coqui. Установите библиотеку: pip install TTS")
    raise


from TTS.tts.models.xtts import XttsArgs

torch.serialization.add_safe_globals([XttsArgs])

# --- Настройки ---
DEVICE = "cpu"
DEFAULT_SPEAKER_WAV = settings.TTS_SPEAKER_WAV # Берем из настроек
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

class SpeechSynthesizer:
    def __init__(self):
        print(f"Инициализация TTS (Coqui TTS: {MODEL_NAME}) на {DEVICE}...")
        self.tts_model = None
        self.engine = False
        self.speaker_wav_path = DEFAULT_SPEAKER_WAV

        if not os.path.exists(self.speaker_wav_path):
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл спикера не найден: {self.speaker_wav_path}")
            raise FileNotFoundError(f"Файл спикера не найден: {self.speaker_wav_path}")

        try:
            start_load = time.time()
            # Загружаем модель
            self.tts_model = TTS(MODEL_NAME).to(DEVICE)

            # --- ВАЖНО: Предварительный "прогрев" модели XTTS ---
            # XTTS может быть медленным при первом запуске после загрузки.
            # Сделаем короткий синтез для инициализации всех компонентов.
            print("TTS: Прогрев модели XTTS...")
            warmup_text = "Привет."
            try:
                # Используем tts_to_file для простоты прогрева
                warmup_path = os.path.join(OUTPUT_DIR, "warmup_tts.wav")
                self.tts_model.tts_to_file(
                    text=warmup_text,
                    speaker_wav=self.speaker_wav_path,
                    language='ru', # Или другой язык для прогрева
                    file_path=warmup_path
                )
                if os.path.exists(warmup_path):
                    os.remove(warmup_path)
                print("TTS: Модель XTTS прогрета.")
            except Exception as warmup_e:
                print(f"TTS: Предупреждение - не удалось прогреть модель: {warmup_e}")
                # Продолжаем работу, но первый синтез может быть долгим

            end_load = time.time()
            print(f"Модель Coqui TTS загружена и прогрета за {end_load - start_load:.2f} сек.")
            self.engine = True
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА при инициализации Coqui TTS: {e}")
            import traceback
            traceback.print_exc()
            self.engine = False
            raise


    async def speak(self, text, audio_chunk_queue: asyncio.Queue, language='ru', output_filename="output_tts.wav"):
        """
        Синтезирует речь в файл, затем передает ПОЛНЫЙ аудио-массив в очередь для воспроизведения.
        (Временная замена для speak_stream, пока не решен вопрос с API стриминга TTS 0.22.0)
        """
        if not self.engine or not text:
            print("TTS Ошибка: движок не готов или нет текста.")
            await audio_chunk_queue.put(None)
            return

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        lang_code = language.lower()[:2]

        print(f"TTS (не-потоковый): Синтез в файл (язык '{lang_code}'): '{text[:60]}...'")
        try:
            start_synth = time.time()
            # 1. Синтезируем весь текст в файл
            self.tts_model.tts_to_file(
                text=text,
                speaker_wav=self.speaker_wav_path,
                language=lang_code,
                file_path=output_path,
            )
            end_synth = time.time()
            print(f"TTS: Аудио сохранено в: {output_path}")
            print(f"TTS: Синтез занял: {end_synth - start_synth:.2f} сек.")

            # 2. Читаем аудио из файла в память (numpy массив)
            # Используем soundfile для чтения, т.к. wav.read может иметь проблемы с путями
            import soundfile as sf
            try:
                audio_data, samplerate = sf.read(output_path, dtype='float32')
                print(
                    f"TTS: Аудио прочитано из файла, sample rate: {samplerate}, длительность: {len(audio_data) / samplerate:.2f}с")

                # --- Проверка и Ресемплинг (если нужно) ---
                tts_native_sr = self.tts_model.synthesizer.output_sample_rate
                if samplerate != settings.AUDIO_SAMPLE_RATE:
                    print(
                        f"TTS: Обнаружено несовпадение SR файла ({samplerate}) и настроек ({settings.AUDIO_SAMPLE_RATE}). Выполняю ресемплинг...")
                    if samplerate != tts_native_sr:
                        print(
                            f"TTS Предупреждение: SR файла ({samplerate}) не совпадает с ожидаемым SR модели ({tts_native_sr}). Ресемплинг может быть неточным.")

                    try:
                        import torchaudio.transforms as T
                        # Конвертируем numpy в tensor для ресемплинга
                        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)  # [samples] -> [1, samples]
                        resampler = T.Resample(orig_freq=samplerate, new_freq=settings.AUDIO_SAMPLE_RATE).to(DEVICE)
                        resampled_audio = resampler(audio_tensor.to(DEVICE))
                        final_audio_data = resampled_audio.squeeze(0).cpu().numpy()  # [1, samples] -> [samples]
                        print(f"TTS: Ресемплинг завершен. Новый размер: {len(final_audio_data)}")
                    except Exception as resample_e:
                        print(f"TTS Ошибка ресемплинга: {resample_e}. Отправляю аудио как есть.")
                        final_audio_data = audio_data  # Отправляем без ресемплинга
                else:
                    final_audio_data = audio_data  # Ресемплинг не нужен

                # 3. Кладем весь numpy массив в очередь для воспроизведения
                await audio_chunk_queue.put(final_audio_data)

            except Exception as read_e:
                print(f"TTS Ошибка чтения/ресемплинга файла {output_path}: {read_e}")

        except ValueError as ve:
            print(f"TTS Ошибка ValueError при синтезе для языка '{lang_code}': {ve}")
        except Exception as e:
            print(f"TTS: Непредвиденная ошибка во время синтеза: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Всегда сигнализируем конец, даже если была ошибка
            print("TTS: Отправка сигнала конца потока в очередь воспроизведения.")
            await audio_chunk_queue.put(None)


