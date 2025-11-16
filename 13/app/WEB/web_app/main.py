from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sys
import httpx
import asyncio
import json

# Добавляем путь к родительской директории для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataManage.DataLoadMain import data_main

app = FastAPI(title="CSV Uploader and ML Runner")

# Настройки ML сервиса
ML_SERVICE_URL = "http://localhost:8001"

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

# Глобальная переменная для хранения статуса данных
data_loaded = False


@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "data_loaded": data_loaded
    })


@app.post("/upload/")
async def upload_csv(
        request: Request,
        file: UploadFile = File(...),
        num_points: int = Form(300)
):
    global data_loaded

    try:
        # Сохраняем загруженный файл временно
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Обрабатываем данные
        success = data_main(file_path, num_collocation_points=num_points)

        # Удаляем временный файл
        os.remove(file_path)

        if success:
            data_loaded = True
            message = "Файл успешно обработан и данные загружены в ClickHouse! Теперь можно запустить ML модель."
            message_type = "success"
        else:
            data_loaded = False
            message = "Ошибка при обработке файла"
            message_type = "danger"

    except Exception as e:
        data_loaded = False
        message = f"Ошибка: {str(e)}"
        message_type = "danger"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": message,
        "message_type": message_type,
        "data_loaded": data_loaded
    })


@app.post("/run_ml/")
async def run_ml_model(request: Request):
    global data_loaded

    if not data_loaded:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Сначала загрузите данные!",
            "message_type": "warning",
            "data_loaded": False
        })

    ml_results = None

    try:
        # Отправляем запрос к ML сервису
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/run_ml_model",
                timeout=300.0  # Увеличиваем таймаут до 5 минут для обучения модели
            )

        if response.status_code == 200:
            ml_results = response.json()
            message = "ML модель успешно обучена!"
            message_type = "success"
        else:
            message = f"Ошибка ML сервиса: {response.text}"
            message_type = "danger"

    except httpx.ConnectError:
        message = "ML сервис недоступен. Убедитесь, что он запущен на порту 8001."
        message_type = "danger"
    except httpx.ReadTimeout:
        message = "Таймаут при выполнении ML модели. Обучение заняло слишком много времени."
        message_type = "warning"
    except Exception as e:
        message = f"Ошибка при запуске ML модели: {str(e)}"
        message_type = "danger"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": message,
        "message_type": message_type,
        "data_loaded": data_loaded,
        "ml_results": ml_results
    })


@app.get("/ml_status")
async def get_ml_status():
    """Получение статуса ML модели из ML сервиса"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ML_SERVICE_URL}/ml_status")
            return response.json()
    except:
        return {"error": "ML service unavailable"}


@app.get("/health")
async def health_check():
    """Проверка статуса веб-сервиса"""
    return {"status": "healthy", "service": "Web", "data_loaded": data_loaded}