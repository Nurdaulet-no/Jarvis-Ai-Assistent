import google.generativeai as genai
import time
from config import settings

class AiCore:
    def __init__(self, api_key, model_name=settings.AI_MODEL_NAME):
        print(f"Инициализация AI Core (Модель: {model_name})...")
        if not api_key:
            print("КРИТИЧЕСКАЯ ОШИБКА: API ключ Google не найден.")
            self.model = None
            return
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.model.generate_content("Test connection")
            print("AI Core инициализирован.")
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать Gemini: {e}")
            print("Проверьте API ключ и интернет-соединение.")
            self.model = None

    def get_response(self, prompt):
        """Получает ответ от AI на заданный текст."""
        if not self.model:
            print("Ошибка: AI модель не инициализирована.")
            return "AI модуль не работает."
        if not prompt:
            return "Я не расслышал ваш запрос. Пожалуйста, повторите."

        print("Отправка запроса к AI...")
        try:
            start_time = time.time()
            # TODO: Добавить обработку истории диалога, если нужно
            response = self.model.generate_content(prompt)
            ai_response = response.text
            end_time = time.time()
            print(f"AI ответил за {end_time - start_time:.2f} сек.")
            return ai_response
        except Exception as e:
            print(f"Ошибка при запросе к AI: {e}")
            return "У меня возникли проблемы с подключением к моему цифровому разуму. Попробуйте запрос позже."