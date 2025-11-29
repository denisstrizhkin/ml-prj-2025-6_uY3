import numpy as np
import tensorflow as tf
import os
from DataLoadFromDB import PINNDataLoader
from Client_ClickHouse import client
from visualization import PredictionVisualizer
import tempfile
from Minio_Client import minio_client

#def get_minio_client():
#    return Minio(
#        os.getenv("MINIO_ENDPOINT", "minio-service:9000"),
#        access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
#        secret_key=os.getenv("MINIO_SECRET_KEY", "admin123"),
#        secure=False
#    )

class PredictionManager:
    def __init__(self, model_id=None, minion_client=minio_client):
        self.model = None
        self.data_loader = PINNDataLoader(client)
        self.visualizer = PredictionVisualizer()
        self.model_id = model_id
        self.minion_client = minion_client

    def load_model(self, model_id=None):
        if model_id:
            self.model_id = model_id

        if not self.model_id:
            raise ValueError("Model ID not specified")

        try:
            #minio_client = get_minio_client()

            # Получаем модель из MinIO
            response = minio_client.get_object("models", f"{self.model_id}.keras")
            model_data = response.read()
            response.close()
            response.release_conn()

            # Создаем временный файл для загрузки
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Записываем данные во временный файл
            with open(tmp_path, 'wb') as f:
                f.write(model_data)

            # Загружаем модель из временного файла
            self.model = tf.keras.models.load_model(tmp_path)

            # Удаляем временный файл
            os.unlink(tmp_path)

            print(f"Модель загружена из MinIO с ID: {self.model_id}")
            return True

        except Exception as e:
            print(f"Ошибка при загрузке модели из MinIO: {e}")
            # Убедимся, что временный файл удален даже при ошибке
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass
            return False

    def get_domain_bounds(self, original_data):

        x_min = float(original_data['x'].min())
        x_max = float(original_data['x'].max())
        t_min = float(original_data['t'].min())
        t_max = float(original_data['t'].max())

        print(f"Границы оригинальных данных: x=[{x_min}, {x_max}], t=[{t_min}, {t_max}]")
        return x_min, x_max, t_min, t_max

    def get_boundary_points(self, original_data, target_time, t_min):

        if original_data.empty:
            return original_data

        # Определяем, является ли время начальным условием
        is_initial_condition = bool(abs(float(target_time) - float(t_min)) < 1e-6)

        if is_initial_condition:
            # Для начального условия берем все точки с этим временем
            time_mask = abs(original_data['t'] - target_time) < 1e-6
            boundary_points = original_data[time_mask]

            # Если точек слишком много, выбираем подмножество для лучшей визуализации
            max_points = 20
            if len(boundary_points) > max_points:
                boundary_points = boundary_points.sample(n=max_points, random_state=42)

        else:
            # Для граничных условий берем только точки с минимальным и максимальным x
            # Сначала находим точки, близкие к целевому времени
            time_tolerance = 0.01 * (float(original_data['t'].max()) - float(original_data['t'].min()))
            time_mask = (original_data['t'] >= target_time - time_tolerance) & \
                        (original_data['t'] <= target_time + time_tolerance)

            time_points = original_data[time_mask]

            if time_points.empty:
                # Если нет точек для точного времени, используем ближайшее доступное
                unique_times = original_data['t'].unique()
                closest_time = unique_times[np.argmin(np.abs(unique_times - target_time))]
                time_points = original_data[abs(original_data['t'] - closest_time) < 1e-6]

            if not time_points.empty:
                # Берем точки с минимальным и максимальным x
                min_x = float(time_points['x'].min())
                max_x = float(time_points['x'].max())

                boundary_points = time_points[
                    (abs(time_points['x'] - min_x) < 1e-6) |
                    (abs(time_points['x'] - max_x) < 1e-6)
                    ]
            else:
                boundary_points = time_points

        return boundary_points

    def prepare_prediction_data(self, x_min, x_max, t_min, t_max, num_x_points, prediction_time):

        # Создаем сетку для предсказания в оригинальных координатах
        x_points = np.linspace(x_min, x_max, num_x_points).reshape(-1, 1)

        # Нормализуем данные
        x_norm = (x_points - x_min) / (x_max - x_min)
        t_norm = np.full_like(x_points, prediction_time)

        # Создаем входные данные для модели
        X_pred = np.hstack([x_norm, t_norm])

        return x_points, X_pred

    def predict_boundary_points(self, boundary_points, x_min, x_max, t_min, t_max, prediction_time):

        if boundary_points.empty:
            return None

        # Нормализуем граничные точки
        boundary_x_norm = (boundary_points['x'].values - x_min) / (x_max - x_min)
        boundary_t_norm = np.full_like(boundary_x_norm, prediction_time)

        # Создаем входные данные для граничных точек
        X_boundary_pred = np.column_stack([boundary_x_norm, boundary_t_norm])

        # Получаем предсказания для граничных точек
        boundary_predictions = self.model.predict(X_boundary_pred, verbose=0)

        return boundary_predictions

    def make_prediction(self, num_x_points=100, prediction_time=0.5):

        try:
            # Загружаем модель если еще не загружена
            if self.model is None:
                self.load_model()

            # Загружаем данные
            original_data = self.data_loader.load_original_data()
            if original_data is None:
                return {"status": "error", "message": "Не удалось загрузить оригинальные данные"}

            # Получаем границы области
            x_min, x_max, t_min, t_max = self.get_domain_bounds(original_data)

            # Преобразуем нормализованное время в оригинальное
            t_original = float(prediction_time * (t_max - t_min) + t_min)
            print(f"Запрошенное время: нормализованное={prediction_time}, оригинальное={t_original}")


            # Подготавливаем данные для основного предсказания
            x_points, X_pred = self.prepare_prediction_data(
                x_min, x_max, t_min, t_max, num_x_points, prediction_time
            )



            # Выполняем основное предсказание
            predictions = self.model.predict(X_pred, verbose=0)

            # Получаем граничные точки
            boundary_points = self.get_boundary_points(original_data, t_original, t_min)
            print(f"Найдено {len(boundary_points)} граничных точек")



            # Вычисляем предсказания для граничных точек
            boundary_predictions = None
            if not boundary_points.empty:
                boundary_predictions = self.predict_boundary_points(
                    boundary_points, x_min, x_max, t_min, t_max, prediction_time
                )



            # Создаем график с помощью визуализатора - передаем prediction_time и num_x_points
            plot_data = self.visualizer.create_prediction_plot(
                x_points=x_points,
                predictions=predictions,
                prediction_time=prediction_time,
                num_x_points=num_x_points,
                boundary_x=boundary_points['x'].values if not boundary_points.empty else None,
                boundary_true=boundary_points['value'].values if not boundary_points.empty else None,
                boundary_pred=boundary_predictions.flatten() if boundary_predictions is not None else None
            )


            if plot_data is None:
                return {"status": "error", "message": "Ошибка при создании графика"}

            # Возвращаем результаты
            results = {
                "status": "success",
                "num_x_points": int(num_x_points),
                "prediction_time": float(prediction_time),
                "prediction_plot": plot_data
            }

            return results



        except Exception as e:
            error_msg = f"Ошибка при выполнении предсказания: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": error_msg}