import google.generativeai as genai
import time
import asyncio
from config import settings

class AiCore:
    def __init__(self, api_key, model_name=settings.AI_MODEL_NAME):
        print(f"Инициализация AI Core (Модель: {model_name})...")
        if not api_key:
            print("КРИТИЧЕСКАЯ ОШИБКА: API ключ Google не найден.")
            self.model = None
            self.initialized = False
            return
        try:
            genai.configure(api_key=api_key)
            # Используем асинхронную версию для проверки соединения, если есть
            # Если нет, придется делать первую проверку синхронно
            self.model = genai.GenerativeModel(model_name)
            # Синхронная проверка соединения при инициализации
            try:
                 print("AI Core: Проверка соединения с Gemini...")
                 self.model.generate_content("Test connection", request_options={'timeout': 10}) # Таймаут 10 сек
                 print("AI Core: Соединение с Gemini успешно установлено.")
                 self.initialized = True
            except Exception as conn_e:
                 print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к Gemini при инициализации: {conn_e}")
                 self.model = None
                 self.initialized = False
                 # Не перевыбрасываем, позволяем приложению запуститься, но AI не будет работать

        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать Gemini библиотеку: {e}")
            self.model = None
            self.initialized = False

    async def get_response_async(self, prompt):
        """Асинхронно получает ответ от AI на заданный текст."""
        if not self.initialized or not self.model:
            print("Ошибка: AI модель не инициализирована или не удалось подключиться.")
            return "AI модуль не работает. Проверьте соединение или API ключ."
        if not prompt:
            # Это сообщение будет синтезировано TTS
            return "Я не расслышал ваш запрос. Пожалуйста, повторите."

        print("AI Core: Отправка асинхронного запроса к AI...")
        try:
            start_time = time.time()
            # Используем асинхронный метод generate_content_async
            response = await self.model.generate_content_async(prompt)
            # Добавляем таймаут к запросу, если нужно (например, 60 секунд)
            # response = await self.model.generate_content_async(prompt, request_options={'timeout': 60})

            # Обработка потенциальных проблем с ответом (блокировка и т.д.)
            if not response.parts:
                 # Проверяем prompt_feedback на наличие BLOCKING
                 if response.prompt_feedback and response.prompt_feedback.block_reason:
                     reason = response.prompt_feedback.block_reason.name
                     print(f"AI Core: Запрос заблокирован Gemini по причине: {reason}")
                     return f"Мой внутренний фильтр заблокировал ваш запрос ({reason}). Попробуйте переформулировать."
                 else:
                     print("AI Core: Получен пустой ответ от Gemini без явной причины блокировки.")
                     return "Я не смог сгенерировать ответ на ваш запрос."

            ai_response = response.text # response.text должен работать и для async
            end_time = time.time()
            print(f"AI Core: AI ответил за {end_time - start_time:.2f} сек.")
            return ai_response
        except Exception as e:
            print(f"AI Core: Ошибка при асинхронном запросе к AI: {e}")
            # Детализация ошибки Gemini, если возможно
            if hasattr(e, 'message'):
                print(f"   Сообщение ошибки Gemini: {e.message}")
            import traceback
            traceback.print_exc()
            return "У меня возникли проблемы с подключением к моему цифровому разуму. Попробуйте запрос позже."

    # Синхронный метод можно оставить для тестов или удалить
    # def get_response(self, prompt): ...