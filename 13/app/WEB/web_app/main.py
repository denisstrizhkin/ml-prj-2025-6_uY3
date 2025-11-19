from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sys
import httpx
#import asyncio
#import json
#import shutil
import glob
from datetime import datetime
from minio import Minio
import io
#import uuid

app = FastAPI(title="CSV Uploader and ML Runner")

# Настройки ML сервиса - для Kubernetes используем имя сервиса
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8001")

# Инициализация MinIO клиента
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio-service:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "admin123"),
    secure=False
)

# Добавляем путь к родительской директории для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataManage.DataLoadMain import data_main

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
temp_dir = os.path.join(current_dir, "temp")

templates = Jinja2Templates(directory=templates_dir)

# Глобальные переменные
data_loaded = False
last_uploaded_file_path = None
last_model_id = None  # Будет хранить ID последней обученной модели

#app.mount("/static", StaticFiles(directory="./static"), name="static")


def clear_temp_directory():
    try:
        files = glob.glob(os.path.join(temp_dir, "*"))
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
        print(f"Очищена директория temp: удалено {len(files)} файлов")
    except Exception as e:
        print(f"Ошибка при очистке директории temp: {e}")


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    # Очищаем директорию temp перед сохранением нового файла
    clear_temp_directory()

    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "data_loaded": data_loaded,
        "last_model_id": last_model_id
    })


@app.post("/upload/")
async def upload_csv(
        request: Request,
        file: UploadFile = File(...),
        num_points: int = Form(300)
):
    global data_loaded, last_uploaded_file_path

    try:
        # Читаем содержимое файла
        content = await file.read()

        # Сохраняем файл в директорию temp
        file_path = save_uploaded_file(content,
                                       f"uploaded_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{file.filename}")
        last_uploaded_file_path = file_path

        # Обрабатываем данные
        success = data_main(file_path, num_collocation_points=num_points)

        if success:
            data_loaded = True
            message = f"Файл успешно обработан и данные загружены в ClickHouse! Загружено {num_points} точек коллокации. Теперь можно запустить ML модель."
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
        "data_loaded": data_loaded,
        "last_model_id": last_model_id
    })


@app.post("/update_collocation/")
async def update_collocation_points(
        request: Request,
        num_points: int = Form(300)
):
    global data_loaded, last_uploaded_file_path

    if not data_loaded:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Сначала загрузите данные!",
            "message_type": "warning",
            "data_loaded": False,
            "last_model_id": last_model_id
        })

    if not last_uploaded_file_path or not os.path.exists(last_uploaded_file_path):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Файл данных не найден. Пожалуйста, загрузите файл снова.",
            "message_type": "warning",
            "data_loaded": data_loaded,
            "last_model_id": last_model_id
        })

    try:
        # Обновляем точки коллокации используя сохраненный файл
        success = data_main(last_uploaded_file_path, num_collocation_points=num_points)

        if success:
            message = f"Точки коллокации успешно обновлены! Новое количество: {num_points}"
            message_type = "success"
        else:
            message = "Ошибка при обновлении точек коллокации"
            message_type = "danger"

    except Exception as e:
        message = f"Ошибка при обновлении точек коллокации: {str(e)}"
        message_type = "danger"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": message,
        "message_type": message_type,
        "data_loaded": data_loaded,
        "last_model_id": last_model_id
    })


@app.post("/run_ml/")
async def run_ml_model(
        request: Request,
        num_layers: int = Form(4),
        num_perceptrons: int = Form(50),
        num_epoch: int = Form(10000),
        optimizer: str = Form("Adam"),
        loss_weights_config: str = Form("")
):
    global data_loaded, last_model_id

    if not data_loaded:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Сначала загрузите данные!",
            "message_type": "warning",
            "data_loaded": False,
            "last_model_id": last_model_id
        })

    ml_results = None

    try:
        # Подготавливаем данные для ML сервиса
        ml_params = {
            "num_layers": num_layers,
            "num_perceptrons": num_perceptrons,
            "num_epoch": num_epoch,
            "optimizer": optimizer,
            "loss_weights_config": loss_weights_config
        }

        print(f"Отправка запроса к ML сервису: {ML_SERVICE_URL}")
        print(f"Параметры: {ml_params}")

        # Отправляем запрос к ML сервису с параметрами
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/run_ml_model",
                json=ml_params,
                timeout=300.0
            )

        if response.status_code == 200:
            ml_results = response.json()
            # Сохраняем ID модели для скачивания
            last_model_id = ml_results.get("model_id")

            message = f"ML модель успешно обучена! Параметры: {num_layers} слоев, {num_perceptrons} нейронов, {num_epoch} эпох, оптимизатор: {optimizer}"
            message_type = "success"
        else:
            message = f"Ошибка ML сервиса: {response.text}"
            message_type = "danger"

    except httpx.ConnectError:
        message = f"ML сервис недоступен по адресу {ML_SERVICE_URL}. Убедитесь, что он запущен."
        message_type = "danger"
    except httpx.ReadTimeout:
        message = "Таймаут при выполнения ML модели. Обучение заняло слишком много времени."
        message_type = "warning"
    except Exception as e:
        message = f"Ошибка при запуске ML модели: {str(e)}"
        message_type = "danger"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": message,
        "message_type": message_type,
        "data_loaded": data_loaded,
        "ml_results": ml_results,
        "last_model_id": last_model_id
    })


@app.post("/predict/")
async def make_prediction(
        request: Request,
        num_x_points: int = Form(100),
        prediction_time: float = Form(0.5)
):
    global data_loaded

    if not data_loaded:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Сначала загрузите данные и обучите ML модель!",
            "message_type": "warning",
            "data_loaded": False,
            "last_model_id": last_model_id
        })

    prediction_results = None

    try:
        # Подготавливаем данные для ML сервиса
        prediction_params = {
            "num_x_points": num_x_points,
            "prediction_time": prediction_time
        }

        # Отправляем запрос к ML сервису
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/predict",
                json=prediction_params,
                timeout=60.0
            )

        if response.status_code == 200:
            prediction_results = response.json()
            message = f"Предсказание успешно построено! {num_x_points} точек при t={prediction_time}"
            message_type = "success"
        else:
            message = f"Ошибка ML сервиса при предсказании: {response.text}"
            message_type = "danger"

    except httpx.ConnectError:
        message = "ML сервис недоступен. Убедитесь, что он запущен на порту 8001."
        message_type = "danger"
    except Exception as e:
        message = f"Ошибка при построении предсказания: {str(e)}"
        message_type = "danger"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": message,
        "message_type": message_type,
        "data_loaded": data_loaded,
        "ml_results": None,
        "prediction_results": prediction_results,
        "last_model_id": last_model_id
    })


@app.get("/ml_status")
async def get_ml_status():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ML_SERVICE_URL}/ml_status")
            return response.json()
    except:
        return {"error": "ML service unavailable"}


@app.get("/download_model")
async def download_model():
    """Скачивание последней обученной модели из MinIO"""
    global last_model_id

    if not last_model_id:
        return {"status": "error", "message": "Модель еще не обучена"}

    try:
        # Получаем модель из MinIO
        response = minio_client.get_object("models", f"{last_model_id}.keras")
        model_data = response.read()

        # Возвращаем файл как поток
        return StreamingResponse(
            io.BytesIO(model_data),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=pinn_model.keras",
                "Content-Type": "application/octet-stream"
            }
        )

    except Exception as e:
        return {"status": "error", "message": f"Ошибка при скачивании модели: {str(e)}"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Web", "data_loaded": data_loaded}


@app.on_event("startup")
async def startup_event():
    print("Запуск приложения...")

    # Создаем бакеты если их нет
    try:
        for bucket in ["models", "plots", "temp-files"]:
            if not minio_client.bucket_exists(bucket):
                minio_client.make_bucket(bucket)
                print(f"Создан бакет MinIO: {bucket}")
    except Exception as e:
        print(f"Ошибка при создании бакетов MinIO: {e}")

    clear_temp_directory()
    print(f"ML_SERVICE_URL: {ML_SERVICE_URL}")


@app.on_event("shutdown")
async def shutdown_event():
    print("Завершение работы приложения...")
    clear_temp_directory()