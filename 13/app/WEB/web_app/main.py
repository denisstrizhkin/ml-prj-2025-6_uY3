from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sys

# Добавляем путь к родительской директории для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataManage.data_prep import DataManager
from DataManage.Client_ClickHouse import client

app = FastAPI(title="CSV Uploader")

# Монтируем статические файлы (для Bootstrap CSS)
#app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)


@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_csv(
        request: Request,
        file: UploadFile = File(...),
        num_points: int = Form(300)
):
    try:
        # Сохраняем загруженный файл временно
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Обрабатываем данные
        data_manager = DataManager(client, database='db')
        success = data_manager.process_csv_data(file_path, num_collocation_points=num_points)

        # Удаляем временный файл
        os.remove(file_path)

        if success:
            message = "Файл успешно обработан и данные загружены в ClickHouse!"
            message_type = "success"
        else:
            message = "Ошибка при обработке файла"
            message_type = "danger"

    except Exception as e:
        message = f"Ошибка: {str(e)}"
        message_type = "danger"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": message,
        "message_type": message_type
    })
