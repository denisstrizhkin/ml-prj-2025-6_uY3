from fastapi import FastAPI, HTTPException
from model_train import train_pinn_model
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os
from prediction_manager import PredictionManager
app = FastAPI(title="ML Service")



ml_status = {
    "is_running": False,
    "last_result": None,
    "error": None
}

# Добавьте эту переменную для хранения ID последней модели
last_model_id = None

class MLParams(BaseModel):
    num_layers: int = 4
    num_perceptrons: int = 50
    num_epoch: int = 10000
    optimizer: str = "Adam"
    loss_weights_config: str = ""  # Добавлен параметр конфигурации весов ошибок

class PredictionParams(BaseModel):
    num_x_points: int = 100
    prediction_time: float = 0.5

@app.post("/run_ml_model")
async def run_ml_model(params: MLParams):
    global ml_status, last_model_id

    if ml_status["is_running"]:
        raise HTTPException(status_code=400, detail="ML модель уже выполняется")

    ml_status["is_running"] = True
    ml_status["error"] = None

    try:
        print(f"Запуск ML модели с параметрами: {params.num_layers} слоев, {params.num_perceptrons} нейронов, {params.num_epoch} эпох, оптимизатор: {params.optimizer}")
        print(f"Конфигурация весов ошибок: {params.loss_weights_config}")

        # Запускаем обучение модели с переданными параметрами
        results = train_pinn_model(
            num_layers=params.num_layers,
            num_perceptrons=params.num_perceptrons,
            num_epoch=params.num_epoch,
            optimizer=params.optimizer,
            loss_weights_config=params.loss_weights_config
        )

        if results["status"] == "error":
            ml_status["error"] = results["message"]
            ml_status["is_running"] = False
            raise HTTPException(status_code=500, detail=results["message"])
        else:
            ml_status["last_result"] = results
            ml_status["is_running"] = False
            last_model_id = results.get("model_id")  # Сохраняем ID модели
            return results

    except Exception as e:
        error_msg = f"Неожиданная ошибка: {str(e)}"
        ml_status["error"] = error_msg
        ml_status["is_running"] = False
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/download_model")
async def download_model():
    try:
        model_path = "pinn_model.keras"
        if os.path.exists(model_path):
            return FileResponse(
                model_path,
                media_type='application/octet-stream',
                filename="pinn_model.keras"
            )
        else:
            raise HTTPException(status_code=404, detail="Модель не найдена. Сначала обучите модель.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при скачивании модели: {str(e)}")


@app.get("/ml_status")
async def get_ml_status():
    return ml_status


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ML",
        "ml_running": ml_status["is_running"]
    }


@app.post("/predict")
async def make_prediction(params: PredictionParams):
    try:
        # Получаем ID последней обученной модели из глобальной переменной
        # (нужно добавить эту переменную в main.py ML сервиса)
        global last_model_id

        if not last_model_id:
            return {"status": "error", "message": "Модель еще не обучена"}

        # Используем класс PredictionManager с указанием model_id
        prediction_manager = PredictionManager(model_id=last_model_id)
        results = prediction_manager.make_prediction(
            num_x_points=params.num_x_points,
            prediction_time=params.prediction_time
        )

        return results

    except Exception as e:
        error_msg = f"Ошибка при предсказании: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    except Exception as e:
        error_msg = f"Ошибка при предсказании: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)