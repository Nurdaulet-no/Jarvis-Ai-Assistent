import pyttsx3
from config import settings

class SpeechSynthesizer:
    def __init__(self, rate=settings.TTS_RATE, volume=settings.TTS_VOLUME, lang_hint=settings.TTS_VOICE_LANG_HINT):
        print("Инициализация TTS движка (pyttsx3)...")
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            self._set_voice(lang_hint)
            print("TTS движок инициализирован.")
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать TTS: {e}")
            print("Проверьте установку pyttsx3 и его зависимостей (espeak/festival на Linux).")
            self.engine = None

    def _set_voice(self, lang_hint):
        if not self.engine: return
        voices = self.engine.getProperty('voices')
        selected_voice = None
        lang_hint_upper = lang_hint.upper()

        print("Поиск подходящего TTS голоса...")
        for voice in voices:
            if lang_hint_upper in voice.id.upper() or lang_hint_upper in voice.name.upper():
                 selected_voice = voice
                 break

        if selected_voice:
            print(f"Устанавливаю TTS голос: {selected_voice.id} ({selected_voice.name})")
            self.engine.setProperty('voice', selected_voice.id)
        else:
            print(f"Голос для языка '{lang_hint}' не найден автоматически. Используется голос по умолчанию.")


    def speak(self, text):
        """Озвучивает текст."""
        if not self.engine:
            print("Ошибка: TTS движок не инициализирован.")
            return
        if not text:
            print("TTS: Нечего озвучивать (пустой текст).")
            return

        print(f"Озвучиваю: '{text[:60]}...'")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Ошибка озвучивания текста: {e}")

    def stop(self):
        """Останавливает воспроизведение (если необходимо)."""
        if self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                print(f"Ошибка при остановке TTS: {e}")