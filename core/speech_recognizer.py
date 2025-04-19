import functools
from faster_whisper import WhisperModel
import time
import numpy as np
import asyncio
import os # Добавлен для работы с путями, если понадобится

# --- Добавленные импорты ---
try:
    import soundfile as sf
except ImportError:
    print("ПРЕДУПРЕЖДЕНИЕ: Библиотека soundfile не найдена. Сохранение аудио для отладки будет невозможно.")
    print("Установите ее: pip install soundfile")
    sf = None
# -------------------------

from config import settings # Убедимся, что импорт настроек работает

class SpeechRecognizer:
    def __init__(self,
                 model_size=settings.STT_MODEL_SIZE,
                 device=settings.STT_DEVICE,
                 compute_type=settings.STT_COMPUTE_TYPE):
        print(f"Загрузка STT модели: {model_size} ({compute_type}) на {device}...")
        self._load_start_time = time.time()
        try:
            # Возможно, нужно указать num_workers=1 для CPU, чтобы избежать конфликтов потоков с asyncio
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type) # num_workers=1
            print(f"STT модель загружена за {time.time() - self._load_start_time:.2f} сек.")
            # Сохраняем целевую частоту дискретизации для использования при сохранении файла
            self.target_sr = settings.AUDIO_SAMPLE_RATE
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить STT модель {model_size}. Ошибка: {e}")
            self.model = None
            raise # Перевыбрасываем ошибку

    async def transcribe_stream(self, audio_queue: asyncio.Queue, stop_event: asyncio.Event):
        """
        Асинхронно получает аудио чанки из очереди, собирает их, сохраняет в файл
        и запускает распознавание после сигнала остановки.
        """
        if not self.model:
            print("Ошибка: STT модель не инициализирована.")
            return None

        audio_buffer = []
        print("STT: Ожидание аудио данных...")

        # --- Цикл сбора чанков (как у тебя) ---
        while True:
            try:
                # Используем wait_for с таймаутом, чтобы не блокироваться навечно
                # и иметь возможность проверить stop_event (хотя в твоей логике он сейчас не проверяется в цикле)
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)

                if chunk is None: # Сигнал конца от audio_handler
                    print("STT: Получен сигнал конца аудио (None).")
                    break
                # Проверка типа данных
                if isinstance(chunk, np.ndarray):
                    audio_buffer.append(chunk)
                else:
                     print(f"STT: ПРЕДУПРЕЖДЕНИЕ: Получен не numpy массив из очереди: {type(chunk)}. Игнорирую.")

            except asyncio.TimeoutError:
                # Если время вышло, проверяем внешний сигнал остановки
                if stop_event.is_set():
                     print("STT: Получен внешний сигнал остановки во время ожидания очереди.")
                     break # Выходим, чтобы обработать то, что есть
                continue # Иначе продолжаем ждать
            except Exception as e:
                 print(f"STT: Ошибка при получении данных из аудио очереди: {e}")
                 import traceback
                 traceback.print_exc()
                 return "[Ошибка чтения очереди STT]"
        # ---------------------------------------


        if not audio_buffer:
            print("STT: Аудио буфер пуст, распознавать нечего.")
            return None

        try:
            # --- Сборка и подготовка аудио (как у тебя + доработки) ---
            print(f"STT: Собираем {len(audio_buffer)} аудио чанков...")
            full_audio = np.concatenate(audio_buffer, axis=0)

            # --- !!! ВАЖНО: Whisper ожидает одномерный массив (моно) !!! ---
            if full_audio.ndim > 1 and full_audio.shape[1] == 1:
                print(f"STT: Преобразование аудио из формы {full_audio.shape} в 1D...")
                full_audio = full_audio.flatten()
            elif full_audio.ndim > 1:
                 print(f"STT: ПРЕДУПРЕЖДЕНИЕ: Обнаружено {full_audio.shape[1]} канала. Используется только первый канал для STT.")
                 full_audio = full_audio[:, 0].copy() # Берем первый канал

            # Конвертируем в float32, если вдруг не так
            if full_audio.dtype != np.float32:
                 print(f"STT: ПРЕДУПРЕЖДЕНИЕ: Конвертация типа аудио из {full_audio.dtype} в float32.")
                 full_audio = full_audio.astype(np.float32)
            # -------------------------------------------------------------

            # --- Расчет длительности и амплитуды ---
            audio_duration_sec = len(full_audio) / self.target_sr if self.target_sr > 0 else 0
            print(f"STT: Собран полный аудиофрагмент длиной {audio_duration_sec:.2f} сек (сэмплов: {len(full_audio)}, SR: {self.target_sr}).")
            max_abs_val = np.max(np.abs(full_audio)) if full_audio.size > 0 else 0
            print(f"STT: Максимальная абсолютная амплитуда: {max_abs_val:.4f}")
            # --------------------------------------


            # --- *** НАЧАЛО БЛОКА СОХРАНЕНИЯ АУДИО *** ---
            if sf is not None and full_audio.size > 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"debug_stt_input_{timestamp}.wav"
                print(f"STT: Попытка сохранения аудио в файл: {filename}")
                try:
                    sf.write(filename, full_audio, self.target_sr)
                    print(f"STT: Аудио УСПЕШНО сохранено в '{filename}'")
                except Exception as save_err:
                    print(f"STT: ОШИБКА сохранения аудио в '{filename}': {save_err}")
            elif sf is None:
                print("STT: Сохранение аудио невозможно (soundfile не импортирован).")
            elif full_audio.size == 0:
                print("STT: Нет аудио данных для сохранения (массив пуст).")
            # --- *** КОНЕЦ БЛОКА СОХРАНЕНИЯ АУДИО *** ---

            # Проверка на совсем короткое аудио
            min_duration_sec = 0.1
            if audio_duration_sec < min_duration_sec:
                 print(f"STT: ПРЕДУПРЕЖДЕНИЕ: Длительность аудио ({audio_duration_sec:.2f}с) слишком мала. Пропуск распознавания.")
                 return None


            # --- Запуск распознавания (как у тебя + VAD Filter) ---
            print(f"STT: Запуск распознавания...")
            loop = asyncio.get_event_loop()
            start_time = time.time()

            transcribe_func = functools.partial(
                self.model.transcribe,
                beam_size=settings.STT_BEAM_SIZE,
                language=settings.STT_DEFAULT_LANGUAGE, # Используем язык по умолчанию из настроек
                # !!! ВАЖНО: отключаем VAD в Whisper, так как он уже сделан ранее !!!
                vad_filter=False,
                # Если нужно передать другие параметры Whisper, добавляем их сюда
                # temperature=...,
                # initial_prompt=...,
            )

            segments, info = await loop.run_in_executor(
                None,
                transcribe_func,
                full_audio
            )

            recognized_text = "".join(segment.text for segment in segments).strip()
            print(f"DEBUG STT: Собранный текст ДО проверки: '{recognized_text}'") # Отладочный принт
            end_time = time.time()

            detected_lang = info.language
            lang_prob = info.language_probability
            duration = info.duration # Это длительность *распознанного* аудио по мнению Whisper
            processing_time = end_time - start_time

            print(f"STT: Распознано (язык: {detected_lang} {lang_prob:.2f}, длительность обработанного аудио: {duration:.2f}с / общая: {audio_duration_sec:.2f}с) за {processing_time:.2f} сек: '{recognized_text}'")

            if not recognized_text:
                print("STT: Ничего не удалось распознать.")
                return None
            return recognized_text
            # ------------------------------------------------------

        except Exception as e:
            print(f"STT: КРИТИЧЕСКАЯ ОШИБКА во время сборки/распознавания аудио: {e}")
            import traceback
            traceback.print_exc()
            return "[Ошибка STT]"