# core/audio_handler.py
import sounddevice as sd
import numpy as np
import asyncio
import torch
import sys
import time
import os
import wave
from collections import deque
import traceback # Добавим на всякий случай

# Импорт для ресемплинга
try:
    import torchaudio.transforms as T
except ImportError:
    print("ПРЕДУПРЕЖДЕНИЕ: torchaudio не найден. Ресемплинг звука с микрофона невозможен!")
    T = None # Устанавливаем в None, чтобы код не падал

# Импорт настроек
try:
    from config import settings
except ImportError:
    print("Критическая ошибка: Не удалось импортировать config.settings в audio_handler.py")
    # Можно сделать sys.exit(1) или поднять исключение, если без настроек никак
    class MockSettings: # Заглушка, чтобы код не падал сразу
        AUDIO_SAMPLE_RATE = 16000
        AUDIO_VAD_THRESHOLD = 0.5
        AUDIO_VAD_SILENCE_TIMEOUT_MS = 1500
        AUDIO_VAD_SPEECH_PAD_MS = 300
        VAD_MODEL_REPO = 'snakers4/silero-vad'
        VAD_MODEL_NAME = 'silero_vad'
        VAD_FORCE_DOWNLOAD = False
        AUDIO_CHANNELS = 1
        AUDIO_TEMP_FILE = "temp_debug_audio.wav" # Другое имя на всякий случай
        TTS_STREAM_PLAYBACK_BUFFER = 2 # Для play_audio_stream
    settings = MockSettings()


# --- Глобальные переменные для VAD и ресемплера ---
_vad_model = None
_vad_utils = None # пока не используется, но оставляем
_resampler = None
_actual_input_sr = None

# --- Функции инициализации и записи (из тестового скрипта) ---

def _load_vad_model_and_check_sr():
    """Загружает VAD модель, определяет частоту микрофона и настраивает ресемплер."""
    global _vad_model, _vad_utils, _resampler, _actual_input_sr
    if _vad_model is None:
        if T is None and settings.AUDIO_SAMPLE_RATE != sd.query_devices(kind='input')['default_samplerate']:
             # Если torchaudio нет И частоты не совпадают - это проблема
             print("[Setup] ОШИБКА: torchaudio не найден, а ресемплинг требуется! Инициализация прервана.")
             raise ImportError("torchaudio is required for resampling but not found.")

        print("[Setup] Загрузка VAD и проверка SR...")
        try:
            # Загружаем VAD
            print(f"[Setup] Используется репозиторий VAD: {settings.VAD_MODEL_REPO}")
            model, utils = torch.hub.load(repo_or_dir=settings.VAD_MODEL_REPO,
                                          model=settings.VAD_MODEL_NAME,
                                          force_reload=settings.VAD_FORCE_DOWNLOAD,
                                          onnx=False)
            _vad_model = model
            _vad_utils = utils
            print("[Setup] Модель Silero VAD загружена.")

            # Проверка и настройка ресемплинга
            target_sr = settings.AUDIO_SAMPLE_RATE
            try:
                print("[Setup] Запрос информации об устройстве ввода...")
                device_info = sd.query_devices(kind='input')
                _actual_input_sr = int(device_info['default_samplerate'])
                print(f"[Setup] АУДИО: Частота микрофона: {_actual_input_sr} Гц (Устройство: {device_info['name']})")

                if _actual_input_sr != target_sr:
                    if T is not None: # Проверяем еще раз на случай, если был None
                        print(f"[Setup] АУДИО: Требуется ресемплинг из {_actual_input_sr} Гц в {target_sr} Гц.")
                        _resampler = T.Resample(orig_freq=_actual_input_sr, new_freq=target_sr).to("cpu")
                        print("[Setup] АУДИО: Ресемплер для микрофона инициализирован.")
                    else:
                        # Эта ветка не должна достигаться из-за проверки в начале, но на всякий случай
                         print("[Setup] ОШИБКА: torchaudio недоступен, хотя ресемплинг нужен!")
                         _resampler = None
                         # Здесь можно было бы выбросить исключение, т.к. работа некорректна
                else:
                    print(f"[Setup] АУДИО: Ресемплинг не требуется.")
                    _resampler = None

            except Exception as e:
                print(f"[Setup] АУДИО: Не удалось определить частоту микрофона ({type(e).__name__}: {e}).")
                print(f"[Setup] АУДИО: Использую целевую частоту {target_sr} Гц как предполагаемую.")
                _actual_input_sr = target_sr
                _resampler = None

        except Exception as e:
            print(f"[Setup] КРИТИЧЕСКАЯ ОШИБКА при загрузке VAD/проверке SR: {e}")
            traceback.print_exc()
            raise
    # Возвращаем только модель и utils (если они понадобятся где-то еще)
    return _vad_model, _vad_utils


async def record_audio_stream(audio_queue: asyncio.Queue, stop_event: asyncio.Event):
    """
    Асинхронно записывает аудио с микрофона, РЕСЕМПЛИРУЕТ, использует VAD
    и кладет ресемплированные аудио-чанки в очередь STT.
    """
    # Убедимся, что VAD и ресемплер инициализированы при первом вызове
    # (хотя в main.py мы вызываем _load_vad_model_and_check_sr заранее)
    if _vad_model is None:
        try:
            _load_vad_model_and_check_sr()
        except Exception as e:
             print(f"[Record] Не удалось инициализировать VAD/ресемплер перед записью: {e}")
             await audio_queue.put(None) # Сигнал ошибки
             return

    vad_model = _vad_model
    target_sr = settings.AUDIO_SAMPLE_RATE
    vad_chunk_size = 512 # Размер чанка, ожидаемый Silero VAD @ 16kHz

    # Настройки VAD
    vad_queue_size = int(settings.AUDIO_VAD_SILENCE_TIMEOUT_MS / 1000 * target_sr / vad_chunk_size) + 1
    q = deque(maxlen=vad_queue_size)
    voiced_confidences = deque(maxlen=30)
    triggered = False
    silence_start_time = None
    speech_detected_once = False

    loop = asyncio.get_event_loop()
    callback_count = 0
    audio_buffer_for_file = bytearray() # Буфер для файла отладки (ресемплированное int16)

    # --- Аудио колбэк ---
    def audio_callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        nonlocal triggered, silence_start_time, speech_detected_once, callback_count
        callback_count += 1
        if status:
            print(f"CALLBACK #{callback_count}: Status={status}", file=sys.stderr)

        current_chunk_np = indata.copy() # shape: [frames, channels]
        # Отладочный print уровня входного сигнала (раз в N вызовов)
        if callback_count % 100 == 1: # Реже, чтобы не мешать
            max_abs_val = np.max(np.abs(current_chunk_np)) if current_chunk_np.size > 0 else 0
            print(f"Mic Input #{callback_count}: frames={frames}, max_abs={max_abs_val:.4f}")

        # --- Ресемплинг ---
        if _resampler:
            try:
                chunk_tensor = torch.from_numpy(current_chunk_np.T).float()
                resampled_tensor = _resampler(chunk_tensor)
                vad_input_tensor = resampled_tensor.flatten()
                processed_chunk_np = resampled_tensor.T.numpy() # shape: [new_frames, channels]
            except Exception as resample_err:
                print(f"CALLBACK #{callback_count}: ОШИБКА РЕСЕМПЛИНГА: {resample_err}.")
                # В случае ошибки, не отправляем данные дальше? Или отправлять оригинал?
                # Безопаснее пропустить этот чанк, чтобы не сломать VAD/STT
                return # Просто выходим из колбэка
        else:
            vad_input_tensor = torch.from_numpy(current_chunk_np.flatten().astype(np.float32))
            processed_chunk_np = current_chunk_np # shape: [frames, channels]

        # --- VAD ---
        q.append(processed_chunk_np) # Добавляем ресемплированный (или нет) чанк
        speech_prob = 0.0
        avg_confidence = 0.0
        try:
            num_samples_vad = vad_input_tensor.shape[0]
            if num_samples_vad >= vad_chunk_size:
                 # Пока берем только начало, если чанк больше vad_chunk_size
                 vad_chunk_for_model = vad_input_tensor[:vad_chunk_size]
                 speech_prob = vad_model(vad_chunk_for_model, target_sr).item()
                 voiced_confidences.append(speech_prob)
                 avg_confidence = np.mean(list(voiced_confidences)) if voiced_confidences else 0
                 # Отладочный print уверенности VAD
                 # if callback_count % 50 == 1:
                 #    print(f"VAD #{callback_count}: Prob={speech_prob:.3f}, AvgConf={avg_confidence:.3f}")
            # else: Пропускаем VAD, если чанк слишком мал

            # --- Логика VAD ---
            if avg_confidence > settings.AUDIO_VAD_THRESHOLD:
                if not triggered:
                    print(f"\n>>> Обнаружено начало речи (AvgConf={avg_confidence:.3f})")
                    triggered = True
                    # Добавляем паддинг
                    pad_duration_samples = int(settings.AUDIO_VAD_SPEECH_PAD_MS / 1000 * target_sr)
                    samples_to_pad = 0
                    padding_chunks = []
                    for past_chunk in reversed(list(q)[:-1]):
                        padding_chunks.insert(0, past_chunk.copy())
                        samples_to_pad += past_chunk.shape[0]
                        if samples_to_pad >= pad_duration_samples: break
                    # print(f"Adding {len(padding_chunks)} padding chunks.") # Отладка
                    for chunk in padding_chunks:
                         loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)
                         audio_buffer_for_file.extend((chunk * 32767).astype(np.int16).tobytes())

                # Отправляем текущий чанк
                loop.call_soon_threadsafe(audio_queue.put_nowait, processed_chunk_np.copy())
                audio_buffer_for_file.extend((processed_chunk_np * 32767).astype(np.int16).tobytes())
                silence_start_time = None
                speech_detected_once = True

            elif triggered: # Тишина после речи
                loop.call_soon_threadsafe(audio_queue.put_nowait, processed_chunk_np.copy())
                audio_buffer_for_file.extend((processed_chunk_np * 32767).astype(np.int16).tobytes())
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif (time.time() - silence_start_time) * 1000 > settings.AUDIO_VAD_SILENCE_TIMEOUT_MS:
                    print(f"\n>>> Обнаружен конец речи (тишина > {settings.AUDIO_VAD_SILENCE_TIMEOUT_MS}ms)")
                    loop.call_soon_threadsafe(stop_event.set)
                    triggered = False
                    silence_start_time = None
            # else: Тишина до речи - ничего не делаем

        except Exception as e:
            print(f"\nCALLBACK #{callback_count}: Ошибка в VAD/логике: {e}")
            traceback.print_exc()
            loop.call_soon_threadsafe(stop_event.set)

    # --- Запуск потока ---
    stream_sr = _actual_input_sr if _actual_input_sr else target_sr
    calculated_blocksize = int(round(vad_chunk_size * (stream_sr / target_sr)))
    input_blocksize = max(calculated_blocksize, vad_chunk_size)

    print(f"Начинаю слушать... (Микрофон SR={stream_sr} Гц, Blocksize={input_blocksize}, Целевой SR={target_sr} Гц)")
    stream = None # Определяем заранее
    try:
        stream = sd.InputStream(
            samplerate=stream_sr,
            channels=settings.AUDIO_CHANNELS,
            dtype='float32',
            blocksize=input_blocksize,
            callback=audio_callback
        )
        stream.start()
        print("Поток записи запущен. Говорите!")
        await stop_event.wait() # Ждем сигнала от VAD
        print("Получен сигнал остановки от VAD.")

    except sd.PortAudioError as pae:
        print(f"КРИТИЧЕСКАЯ ОШИБКА PortAudio при запуске записи: {pae}")
        try: print(sd.query_devices())
        except Exception: pass
        await audio_queue.put(None) # Сигнализируем об ошибке
        return # Выходим из функции записи
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА при запуске/работе потока записи: {e}")
        traceback.print_exc()
        await audio_queue.put(None)
        return # Выходим
    finally:
        # Гарантированно останавливаем и закрываем поток
        if stream is not None and stream.active:
            try:
                stream.stop()
                stream.close()
                print("Поток записи остановлен и закрыт.")
            except Exception as close_e:
                 print(f"Ошибка при закрытии потока записи: {close_e}")
        # Сигнализируем конец STT
        await asyncio.sleep(0.1) # Небольшая пауза перед отправкой None
        await audio_queue.put(None)
        print("Отправлен None в очередь STT.")

        # Сохраняем файл отладки
        if speech_detected_once and audio_buffer_for_file:
            try:
                with wave.open(settings.AUDIO_TEMP_FILE, 'wb') as wf:
                    wf.setnchannels(settings.AUDIO_CHANNELS)
                    wf.setsampwidth(2) # int16
                    wf.setframerate(target_sr) # Целевая частота
                    wf.writeframes(audio_buffer_for_file)
                print(f"Полная запись (отладка, {target_sr} Гц) сохранена в {settings.AUDIO_TEMP_FILE}")
            except Exception as e:
                print(f"Не удалось сохранить отладочный wav: {e}")
        elif not speech_detected_once:
            print("Речь не была обнаружена VAD (файл не сохранен).")


# --- Async Playback ---
# Вставляем УПРОЩЕННУЮ версию play_audio_stream из предыдущих шагов,
# так как мы пока используем НЕ-ПОТОКОВЫЙ TTS
async def play_audio_stream(audio_chunk_queue: asyncio.Queue):
    """
    Асинхронно воспроизводит аудио данные из очереди.
    ОПТИМИЗИРОВАНО для получения одного большого чанка от не-потокового TTS.
    """
    playback_finished_event = asyncio.Event()
    audio_data = None # Для доступа в finally

    try:
        audio_data = await audio_chunk_queue.get()

        if audio_data is None:
            print("Воспроизведение: Получен None. Нечего воспроизводить.")
            return

        if not isinstance(audio_data, np.ndarray):
             if isinstance(audio_data, torch.Tensor): audio_data = audio_data.numpy()
             elif isinstance(audio_data, bytes): audio_data = np.frombuffer(audio_data, dtype=np.float32)
             else: print(f"Воспр: Неизвестный тип данных: {type(audio_data)}"); return

        if audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32)
        if audio_data.ndim == 1: audio_data = audio_data.reshape(-1, 1)

        target_sr = settings.AUDIO_SAMPLE_RATE # Частота для воспроизведения
        print(f"Воспроизведение: Получено аудио shape={audio_data.shape}, duration={len(audio_data)/target_sr:.2f}с")

        if audio_data.size == 0:
             print("Воспроизведение: Получен пустой аудио массив. Пропуск.")
             return

        try:
            print("Воспроизведение: Запуск sd.play / sd.wait...")
            loop = asyncio.get_event_loop()
            # Запускаем блокирующие операции в executor'е
            await loop.run_in_executor(None, sd.play, audio_data, target_sr)
            await loop.run_in_executor(None, sd.wait)
            print("Воспроизведение: sd.wait() завершен.")
            playback_finished_event.set()

        except sd.PortAudioError as pae:
            print(f"Воспр: Ошибка PortAudio: {pae}")
            try: print(sd.query_devices())
            except Exception: pass
        except Exception as e:
            print(f"Воспр: Ошибка во время sd.play/wait: {e}")
            traceback.print_exc()

        # Ожидаем None из очереди
        final_signal = await audio_chunk_queue.get()
        # if final_signal is not None: print("Воспр: Предупреждение: получен неожиданный сигнал.")

    except asyncio.CancelledError:
        print("Воспроизведение: Задача отменена.")
        sd.stop() # Останавливаем воспроизведение при отмене
    except Exception as e:
        print(f"\nВоспр: Ошибка в play_audio_stream: {e}")
        traceback.print_exc()
    finally:
        print("Воспроизведение: Функция play_audio_stream завершена.")
        # Убедимся, что воспроизведение остановлено, если оно было запущено
        # sd.stop() # sd.wait() должен это сделать, но для надежности можно добавить


# --- Очистка временных файлов ---
def remove_temp_audio(filename=None):
    """Удаляет временный аудиофайл, если он указан и существует."""
    if filename is None:
         filename = settings.AUDIO_TEMP_FILE # Берем из настроек по умолчанию

    if filename and os.path.exists(filename):
        try:
            os.remove(filename)
            # print(f"Временный файл {filename} удален.")
        except Exception as e:
            print(f"Не удалось удалить временный файл {filename}: {e}")