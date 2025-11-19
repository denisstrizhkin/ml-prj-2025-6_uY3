from Logger import Logger
from DataLoadFromDB import PINNDataLoader
from Client_ClickHouse import client
from PINN import PINN
from PINN import init_model_params
from PINN import save_model_to_minio
import traceback
import time
import os
from minio import Minio
import uuid


def get_minio_client():
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio-service:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "admin123"),
        secure=False
    )


def train_pinn_model(num_layers, num_perceptrons, num_epoch, optimizer, loss_weights_config=""):
    print(f"Полученная конфигурация весов: '{loss_weights_config}'")
    try:
        start_time = time.time()

        # Инициализация загрузчика данных
        data_loader = PINNDataLoader(client)

        # Загрузка данных из ClickHouse
        X_u_train, u_train = data_loader.load_training_data()
        X_f_train = data_loader.load_collocation_points()

        if X_u_train is None or u_train is None or X_f_train is None:
            error_msg = "Ошибка при загрузке данных из ClickHouse"
            return {"status": "error", "message": error_msg}

        # Вычисление границ области
        lb, ub = data_loader.get_domain_bounds(X_u_train, X_f_train)

        print(f"Границы области: lb={lb}, ub={ub}")
        print(f"Размерности: X_u_train {X_u_train.shape}, u_train {u_train.shape}, X_f_train {X_f_train.shape}")

        # Параметры обучения с учетом выбранного оптимизатора
        lr_schedule, tf_optimizer, layers, tf_epochs = init_model_params(
            num_layers, num_perceptrons, num_epoch, optimizer
        )
        logger = Logger(frequency=200)

        # Создание и обучение модели с передачей конфигурации весов
        pinn = PINN(layers, tf_optimizer, logger, X_f_train, lb, ub)
        training_results = pinn.fit(X_u_train, u_train, tf_epochs, loss_weights_config)

        end_time = time.time()
        training_duration = end_time - start_time

        # Сохраняем модель в MinIO с уникальным именем
        model_id = f"pinn_model_{int(time.time())}"
        model_saved = save_model_to_minio(pinn, model_id)

        if not model_saved:
            return {"status": "error", "message": "Ошибка при сохранении модели в MinIO"}

        # Получаем график и статистику обучения
        training_plot = logger.get_training_plot()

        # Упрощенные результаты - только основные параметры
        results = {
            "status": "success",
            "message": "ML модель успешно обучена",
            "training_model_time": round(training_duration, 2),
            "training_epochs": tf_epochs,
            "model_layers": len(layers),
            "num_perceptrons": num_perceptrons,
            "optimizer": optimizer,
            "best_loss": float(training_results["best_loss"]),
            "training_plot": training_plot,
            "loss_weights_used": loss_weights_config if loss_weights_config else "равные веса [1.0, 1.0]",
            "model_id": model_id  # Возвращаем ID модели для скачивания
        }

        return results

    except Exception as e:
        error_msg = f"Ошибка при выполнении ML модели: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"status": "error", "message": error_msg}