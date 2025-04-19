# main.py
import os
import re
import asyncio
import signal
from dotenv import load_dotenv
import nltk
import traceback

# --- Загрузка зависимостей проекта ---
try:
    from config import settings
    from core import audio_handler
    from core.speech_recognizer import SpeechRecognizer
    from core.ai_core import AiCore
    from core.speech_synthesizer import SpeechSynthesizer
    import sounddevice as sd # Импортируем для PortAudioError
except ImportError as import_err:
     print(f"Ошибка импорта: {import_err}")
     print("Убедитесь, что все зависимости установлены и скрипт запускается из корневой папки проекта.")
     exit(1)
# ------------------------------------

# --- Загрузка токенизатора предложений NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' не найден. Скачиваю...")
    try:
        nltk.download('punkt', quiet=True)
        print("NLTK 'punkt' успешно скачан.")
    except Exception as nltk_e:
        print(f"Не удалось скачать NLTK 'punkt': {nltk_e}. Разбиение на предложения может не работать.")
# ---------------------------------------------

def clean_text_for_tts(text):
    if not text: return text
    # Удаляем ссылки ([текст](url)), оставляя текст
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Удаляем Markdown форматирование (**, *, `), оставляя текст
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Удаляем заголовки Markdown (#, ##, ...)
    text = re.sub(r'^\s*#+\s+', '', text, flags=re.MULTILINE)
    # Удаляем горизонтальные линии
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Удаляем маркеры списков (*, -, +) в начале строк
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    # Удаляем URL
    text = re.sub(r'https?://\S+', '', text)
    # Заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text).strip()
    return text

shutdown_event = asyncio.Event()

def handle_signal(sig, frame):
    print(f"\nПолучен сигнал {sig}. Завершаю работу...")
    shutdown_event.set()

async def main():
    print(f"Запуск {settings.APP_NAME} (Асинхронная версия)...")
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")

    recognizer = None
    ai = None
    synthesizer = None

    try:
        print("Инициализация распознавателя речи...")
        recognizer = SpeechRecognizer()
        print("Инициализация AI ядра...")
        ai = AiCore(api_key=google_api_key)
        print("Инициализация синтезатора речи...")
        synthesizer = SpeechSynthesizer()
        print("Загрузка модели VAD и проверка SR...")
        audio_handler._load_vad_model_and_check_sr()
    except ImportError as e:
         print(f"\nОшибка импорта при инициализации: {e}")
         print("Возможно, не установлена необходимая библиотека (например, torchaudio).")
         return
    except Exception as e:
        print(f"\nКритическая ошибка при инициализации одного из компонентов: {e}")
        traceback.print_exc()
        print("Работа программы невозможна.")
        return

    if not recognizer or not recognizer.model or \
       not ai or not ai.initialized or \
       not synthesizer or not synthesizer.engine or \
       audio_handler._vad_model is None:
         print("Один или несколько ключевых компонентов не инициализированы корректно. Завершение работы.")
         return

    print("-" * 20)
    print("Системы готовы к работе! Нажмите Ctrl+C для выхода.")
    print("-" * 20)

    loop_count = 0
    while not shutdown_event.is_set():
        loop_count += 1
        print(f"\n--- Цикл обработки запроса #{loop_count} ---")

        raw_audio_queue = asyncio.Queue()
        tts_output_queue = asyncio.Queue()
        stop_recording_event = asyncio.Event()
        current_tasks = set()

        try:
            record_task = asyncio.create_task(
                audio_handler.record_audio_stream(raw_audio_queue, stop_recording_event), name="RecordVADTask"
            )
            current_tasks.add(record_task)

            stt_task = asyncio.create_task(
                recognizer.transcribe_stream(raw_audio_queue, stop_recording_event), name="STTTask"
            )
            current_tasks.add(stt_task)

            print("Ожидание завершения речи и распознавания...")
            user_query = await stt_task
            await record_task
            current_tasks.remove(stt_task)
            current_tasks.remove(record_task)

            if user_query and not shutdown_event.is_set():
                print(f"Пользователь сказал: '{user_query}'")

                # --- ФОРМИРОВАНИЕ ПРОМПТА ДЛЯ AI ---
                ai_system_prompt = (
                    "Ты - ассистент, который общается как 'свой парень' или 'братан'. "
                    "Используй неформальный стиль, обращайся на 'ты', можешь использовать разговорные слова типа 'че', 'короче', 'ваще', 'типа'. "
                    "Если вопрос пользователя кажется тебе простым или очевидным, отреагируй удивленно или с подколкой, например: 'Братан, ты че, серьезно это не знаешь?', 'Да ладно, это ж элементарно!', 'Слышь, ну ты даешь...'. "
                    "Отвечай по существу, но очень коротко - одно, максимум два предложения (до 15-20 слов). "
                    "Не используй Markdown и формальности."
                )
                final_prompt = f"ИНСТРУКЦИЯ ПО ПОВЕДЕНИЮ:\n{ai_system_prompt}\n\nЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_query}"
                print(f"Сформирован промпт для AI: '{final_prompt[:200]}...'")
                # --- ----------------------------- ---

                ai_task = asyncio.create_task(
                    ai.get_response_async(final_prompt), # Передаем измененный промпт
                    name="AITask"
                )
                current_tasks.add(ai_task)

                print("Ожидание ответа от AI...")
                ai_answer = await ai_task
                current_tasks.remove(ai_task)
                print(f"AI ответил (сырой): '{ai_answer[:150]}...'")

                play_task = asyncio.create_task(
                    audio_handler.play_audio_stream(tts_output_queue), name="PlaybackTask"
                )
                current_tasks.add(play_task)

                if ai_answer and "AI модуль не работает" not in ai_answer and "проблемы с подключением" not in ai_answer and not shutdown_event.is_set():
                    ai_answer_cleaned = clean_text_for_tts(ai_answer)
                    print(f"AI ответил (очищенный): '{ai_answer_cleaned[:150]}...'")

                    if ai_answer_cleaned:
                        try:
                            # Разбиваем на предложения, даже если ответ короткий
                            sentences = nltk.sent_tokenize(ai_answer_cleaned, language='russian')
                            print(f"Разбито на {len(sentences)} предложений для TTS.")
                        except Exception as nltk_e:
                            print(f"Ошибка NLTK sent_tokenize: {nltk_e}. Синтезирую текст целиком.")
                            sentences = [ai_answer_cleaned]

                        tts_tasks_group = set()
                        for i, sentence in enumerate(sentences):
                            if not sentence.strip(): continue
                            print(f"TTS: Отправка предложения #{i+1} на синтез: '{sentence[:50]}...'")
                            tts_sentence_task = asyncio.create_task(
                                 synthesizer.speak(sentence, tts_output_queue, language='ru', output_filename=f"tts_sentence_{i}.wav"),
                                 name=f"TTSSentenceTask_{i}"
                            )
                            tts_tasks_group.add(tts_sentence_task)
                            current_tasks.add(tts_sentence_task)

                        if tts_tasks_group:
                            print(f"Ожидание завершения {len(tts_tasks_group)} задач синтеза...")
                            await asyncio.gather(*tts_tasks_group)
                            print("Все задачи синтеза завершены.")
                            current_tasks.difference_update(tts_tasks_group)
                        else:
                            print("Нет предложений для синтеза.")
                            await tts_output_queue.put(None)

                    else:
                        print("Ответ AI пуст после очистки.")
                        await tts_output_queue.put(None)

                elif ai_answer: # Сообщение об ошибке от AI
                    print(f"Получено сообщение об ошибке от AI: {ai_answer}")
                    # Синтезируем сообщение об ошибке
                    error_tts_task = asyncio.create_task(
                         synthesizer.speak(ai_answer, tts_output_queue, language='ru'), name="ErrorTTSTask"
                    )
                    current_tasks.add(error_tts_task)
                    await error_tts_task
                    current_tasks.remove(error_tts_task)
                else: # Пустой ответ AI
                     print("AI не дал ответа или работа прервана.")
                     await tts_output_queue.put(None)

                print("Ожидание завершения воспроизведения...")
                await play_task
                print("Воспроизведение завершено.")
                current_tasks.remove(play_task)

            elif not user_query: # STT не распознал
                print("Не удалось распознать речь или была тишина.")
                error_message = "Я не расслышал ваш запрос. Пожалуйста, повторите."

                play_task = asyncio.create_task(
                     audio_handler.play_audio_stream(tts_output_queue), name="ErrorPlaybackTask"
                )
                current_tasks.add(play_task)

                error_tts_task = asyncio.create_task(
                     synthesizer.speak(error_message, tts_output_queue, language='ru'), name="ErrorTTSTask"
                )
                current_tasks.add(error_tts_task)

                await error_tts_task
                current_tasks.remove(error_tts_task)
                await play_task
                current_tasks.remove(play_task)

            # Очистка временных файлов
            num_sentences_local = len(locals().get('sentences', [])) # Получаем кол-во предложений, если переменная была создана
            for i in range(num_sentences_local):
                 audio_handler.remove_temp_audio(f"outputs/tts_sentence_{i}.wav")
            audio_handler.remove_temp_audio(settings.AUDIO_TEMP_FILE)


        except asyncio.CancelledError:
            print("\nЦикл обработки прерван сигналом.")
            for task in current_tasks:
                if not task.done():
                    print(f"Отмена задачи {task.get_name()} при прерывании...")
                    task.cancel()
            await asyncio.sleep(0.1)
            break
        except sd.PortAudioError as pae:
             print(f"\nКРИТИЧЕСКАЯ ОШИБКА ЗВУКА в цикле: {pae}")
             print("Проверьте настройки устройств ввода/вывода звука.")
             shutdown_event.set()
        except Exception as e:
            print(f"\nНепредвиденная ошибка в основном цикле: {type(e).__name__}: {e}")
            traceback.print_exc()
            print("Попытка продолжить работу...")
            for task in current_tasks:
                if not task.done():
                    print(f"Отмена задачи {task.get_name()} из-за ошибки...")
                    task.cancel()
            await asyncio.sleep(0.5)

    print("\nНачинаю остановку систем...")
    audio_handler.remove_temp_audio(settings.AUDIO_TEMP_FILE)
    print(f"{settings.APP_NAME} завершил работу.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        if os.name == 'nt':
            if 'TORCH_HOME' not in os.environ:
                 cache_dir = "C:\\torch_cache"
                 if not os.path.exists(cache_dir):
                      try: os.makedirs(cache_dir)
                      except OSError as e: print(f"Не удалось создать {cache_dir}: {e}")
                 if os.path.exists(cache_dir):
                     os.environ['TORCH_HOME'] = cache_dir
                     print(f"Установлена TORCH_HOME={os.environ['TORCH_HOME']}")

        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nЗавершение работы по KeyboardInterrupt...")
    except Exception as main_run_e:
        print(f"\nФатальная ошибка при запуске/работе программы: {main_run_e}")
        traceback.print_exc()
    finally:
        print("Программа завершена.")