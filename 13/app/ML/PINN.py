import tensorflow as tf
import math
import io
import os
from minio import Minio
import tempfile

def get_minio_client():
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio-service:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "admin123"),
        secure=False
    )


def save_model_to_minio(model, model_id):
    """Сохраняет модель в MinIO"""
    try:
        # Создаем временный файл с правильным расширением
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # Сохраняем модель во временный файл
        model.u_model.save(tmp_path)

        # Читаем содержимое файла
        with open(tmp_path, 'rb') as f:
            model_data = f.read()

        # Создаем буфер из данных
        model_buffer = io.BytesIO(model_data)
        model_buffer.seek(0)

        # Получаем клиент MinIO
        minio_client = get_minio_client()

        # Создаем бакет если не существует
        bucket_name = "models"
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Сохраняем модель в MinIO
        minio_client.put_object(
            bucket_name,
            f"{model_id}.keras",
            model_buffer,
            length=len(model_data)
        )

        # Удаляем временный файл
        os.unlink(tmp_path)

        print(f"Модель сохранена в MinIO с ID: {model_id}")
        return True

    except Exception as e:
        print(f"Ошибка при сохранении модели в MinIO: {e}")
        # Убедимся, что временный файл удален даже при ошибке
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except:
            pass
        return False


def load_model_from_minio(model_id):
    """Загружает модель из MinIO"""
    try:
        minio_client = get_minio_client()

        # Получаем модель из MinIO
        response = minio_client.get_object("models", f"{model_id}.keras")
        model_data = response.read()

        # Создаем временный файл для загрузки
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
            tmp_file.write(model_data)
            tmp_file.flush()

            # Загружаем модель из временного файла
            model = tf.keras.models.load_model(tmp_file.name)

            # Удаляем временный файл
            os.unlink(tmp_file.name)

        print(f"Модель загружена из MinIO с ID: {model_id}")
        return model

    except Exception as e:
        print(f"Ошибка при загрузке модели из MinIO: {e}")
        # Убедимся, что временный файл удален даже при ошибке
        try:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)
        except:
            pass
        return None



def init_model_params(num_layers, num_perceptrons, num_epoch, optimizer):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=2000,
        decay_rate=0.95
    )

    # Выбор оптимизатора
    tf_optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    # Архитектура сети
    layers = [2]
    for i in range(num_layers):
        layers.append(num_perceptrons)
    layers.append(1)

    tf_epochs = num_epoch

    return lr_schedule, tf_optimizer, layers, tf_epochs


def parse_loss_weights_config(config_str):

    if not config_str:
        return None

    schedule = []
    try:
        entries = config_str.split(';')
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            epoch_part, weights_part = entry.split(':')
            epoch = int(epoch_part.strip())
            weights_str = weights_part.split(',')
            data_weight = float(weights_str[0].strip())
            pde_weight = float(weights_str[1].strip())
            schedule.append((epoch, [data_weight, pde_weight]))

        # Сортируем по эпохе
        schedule.sort(key=lambda x: x[0])
        return schedule
    except Exception as e:
        print(f"Ошибка парсинга конфигурации весов: {e}. Используются веса по умолчанию.")
        return None







def get_loss_weights_for_epoch(epoch, loss_weights_schedule):

    if not loss_weights_schedule:
        # Значения по умолчанию
        return [1.0, 1.0]

    # Находим последний подходящий интервал
    current_weights = [1.0, 1.0]  # значения по умолчанию
    for threshold_epoch, weights in loss_weights_schedule:
        if epoch >= threshold_epoch:
            current_weights = weights
        else:
            break

    return current_weights







class PINN(object):
    def __init__(self, layers, optimizer, logger, X_f, lb, ub):
        self.layers = layers
        self.u_model = tf.keras.Sequential()
        self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        self.best_loss = math.inf

        # нормируем входной слой
        #self.u_model.add(tf.keras.layers.Lambda(
        #    lambda X: 2.0 * (X - lb) / (ub - lb) - 1.0))

        #self.u_model.add(NormalizationLayer(lb, ub))

        self.u_model.add(tf.keras.layers.Rescaling(scale=2.0, offset=-1.0))

        # инициализируем слои НН
        for width in layers[1:-1]:
            self.u_model.add(tf.keras.layers.Dense(
                width,
                activation=tf.nn.tanh,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                bias_initializer='zeros'
            ))

        # Последний будет ф-ией relu ( шоб ниже нуля не упало )
        self.u_model.add(tf.keras.layers.Dense(
            layers[-1],
            activation=None,  # Linear activation for output
            kernel_initializer=tf.keras.initializers.GlorotNormal()
        ))

        self.optimizer = optimizer
        self.logger = logger
        self.dtype = tf.float32

        # Фигачим в тензора
        self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=self.dtype)
        self.t_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=self.dtype)

        self.X_f = tf.convert_to_tensor(X_f, dtype=self.dtype)

    @tf.function
    def loss_fn(self, X_u, u, loss_weights=None):
        if loss_weights is None:
            loss_weights = [1.0, 1.0]  # пока пусть ошибка по (Г.У. + Н.У.) и ур-ю одинаковой будет

        u_pred = self.u_model(X_u)
        data_loss = tf.reduce_mean(tf.square(u - u_pred))

        pde_loss = tf.reduce_mean(tf.square(self.f_model()))

        total_loss = loss_weights[0] * data_loss + loss_weights[1] * pde_loss
        return total_loss, data_loss, pde_loss

    def __wrap_training_variables(self):
        return self.u_model.trainable_variables

    # надо по гиперпараметрам пройтись тренировкой ( обратный град спуск )
    @tf.function
    def train_step(self, X_u, u, loss_weights=None):

        with tf.GradientTape() as tape:
            total_loss, data_loss, pde_loss = self.loss_fn(X_u, u, loss_weights)

        grads = tape.gradient(total_loss, self.__wrap_training_variables())
        self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))

        return total_loss, data_loss, pde_loss

    @tf.function
    def f_model(self):

        # берем производные для ур-я Б-Л
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_f)
            tape.watch(self.t_f)

            X_f = tf.stack([self.x_f[:, 0], self.t_f[:, 0]], axis=1)
            u = self.u_model(X_f)

            Swc = 0.0
            Sor = 0.0
            M = 2.0

            numerator = tf.square(u)
            denominator = numerator + tf.square(1.0 - u) / M
            frac_org = numerator / (denominator + 1e-8)  # Add small epsilon for stability

            # крит точка
            Sf = tf.sqrt((1.0 / M) / (1.0 / M + 1.0))
            numerator_Sf = tf.square(Sf - Swc)
            denominator_Sf = numerator_Sf + tf.square(1.0 - Sf - Sor) / M
            frac_Sf = numerator_Sf / (denominator_Sf + 1e-8)

            linear_part = (frac_Sf / Sf) * u
            f_u = tf.where(u < Sf, linear_part, frac_org)

        u_t = tape.gradient(u, self.t_f)
        u_x = tape.gradient(u, self.x_f)
        f_u_x = tape.gradient(f_u, self.x_f)

        del tape

        # полученное зн-е ур-я после предикта : u_t + f(u)_x = 0
        return u_t + f_u_x

    def fit(self, X_u, u, tf_epochs, loss_weights_config=None):

        self.logger.log_train_start(self)

        X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
        u = tf.convert_to_tensor(u, dtype=self.dtype)

        # Парсим конфигурацию весов ошибок
        loss_weights_schedule = parse_loss_weights_config(loss_weights_config)
        if loss_weights_schedule:
            print(f"Используется пользовательское расписание весов: {loss_weights_schedule}")
        else:
            print("Используется расписание весов по умолчанию")

        # адамом решено
        self.logger.log_train_opt("Adam")

        best_loss = math.inf

        for epoch in range(tf_epochs):

            # Определяем веса для текущей эпохи
            current_weights = get_loss_weights_for_epoch(epoch, loss_weights_schedule)

            total_loss, data_loss, pde_loss = self.train_step(X_u, u, current_weights)

            current_loss = total_loss.numpy()
            if current_loss < best_loss:
                best_loss = current_loss

            if epoch % self.logger.frequency == 0:
                custom_info = f"data_loss: {data_loss:.2e}, pde_loss: {pde_loss:.2e}, weights: {current_weights}"
                self.logger.log_train_epoch(epoch, total_loss, custom_info)

        return {"best_loss": best_loss}

    def predict(self, X):
        return self.u_model(X)

def save_model(model, save_dir="pinn_model.keras"):

    model.u_model.save(save_dir)
    print(f"Модель сохранена в {save_dir}")
