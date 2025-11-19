FROM python:3.9-slim

WORKDIR /app

# Установка только необходимых системных зависимостей
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements и установка Python зависимостей
COPY WEB/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY WEB/ .

# Создание директории для временных файлов
RUN mkdir -p /app/temp

# Порт для FastAPI
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "web_app.main:app", "--host", "0.0.0.0", "--port", "8000"]