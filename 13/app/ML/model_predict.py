import numpy as np
import tensorflow as tf
import os
from DataLoadFromDB import PINNDataLoader
from Client_ClickHouse import client
from visualization import PredictionVisualizer


class PredictionManager:
    def __init__(self, model_id=None):
        self.model_id = model_id
        #self.model_path = model_path
        self.model = None
        self.data_loader = PINNDataLoader(client)
        self.visualizer = PredictionVisualizer()

    def load_model(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        print("Модель успешно загружена")
        return True

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

            # Создаем график с помощью визуализатора
            plot_data = self.visualizer.create_prediction_plot(
                x_points=x_points,
                predictions=predictions,
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