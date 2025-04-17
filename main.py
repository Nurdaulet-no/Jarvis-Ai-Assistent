import os
import re
from dotenv import load_dotenv

from config import settings
from core import audio_handler
from core.speech_recognizer import SpeechRecognizer
from core.ai_core import AiCore
from core.speech_synthesizer import SpeechSynthesizer


def clean_text_for_tts(text):
    if not text:
        return text
    print(f"Исходный текст от AI: '{text}'")  # Лог для отладки
    text = re.sub(r'^\s*[\*#\-]+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'https?://\S+', '', text)

    text = re.sub(r'[`\*_]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"Очищенный текст для TTS: '{text}'")
    return text


def main():
    print(f"Запуск {settings.APP_NAME}...")

    # --- 1. Загрузка конфигурации и инициализация ---
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")

    try:
        recognizer = SpeechRecognizer()
        ai = AiCore(api_key=google_api_key)
        synthesizer = SpeechSynthesizer(lang_hint=settings.TTS_VOICE_LANG_HINT) # Передаем подсказку языка
    except Exception as e:
        print(f"Критическая ошибка при инициализации: {e}")
        print("Работа программы невозможна.")
        return

    if not recognizer.model or not ai.model or not synthesizer.engine:
         print("Один или несколько ключевых компонентов не инициализированы. Завершение работы.")
         return

    print("-" * 20)
    print("Системы готовы к работе!")
    print("-" * 20)

    # --- 2. Основной цикл ---
    try:
        while True:
            audio_file = audio_handler.record_audio(
                filename=settings.AUDIO_TEMP_FILE,
                duration=settings.AUDIO_RECORD_DURATION
            )

            if not audio_file:
                synthesizer.speak("Проблема с записью звука, попробуйте еще раз.")
                continue

            user_query = recognizer.transcribe(audio_file, language=settings.STT_DEFAULT_LANGUAGE)

            ai_answer = ai.get_response(user_query)
            ai_answer_cleaned = clean_text_for_tts(ai_answer)

            synthesizer.speak(ai_answer_cleaned)

            audio_handler.remove_temp_audio(audio_file)

    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания (Ctrl+C). Завершаю работу...")
    except Exception as e:
        print(f"\nПроизошла непредвиденная ошибка: {e}")
    finally:
        print("Остановка систем...")
        if 'synthesizer' in locals() and synthesizer.engine:
            synthesizer.stop()
        audio_handler.remove_temp_audio(settings.AUDIO_TEMP_FILE)
        print(f"{settings.APP_NAME} завершил работу.")

if __name__ == "__main__":
    main()