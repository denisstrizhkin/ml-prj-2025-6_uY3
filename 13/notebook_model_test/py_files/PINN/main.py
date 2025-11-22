from Logger import Logger  # Импортируем обычный логгер
from DataLoadFromDB import PINNDataLoader
from Client_ClickHouse import client
from PINN import PINN
from PINN import init_model_params

def main():
    # Инициализация загрузчика данных
    data_loader = PINNDataLoader(client)

    # Загрузка данных из ClickHouse
    X_u_train, u_train = data_loader.load_training_data()
    X_f_train = data_loader.load_collocation_points()

    if X_u_train is None or u_train is None or X_f_train is None:
        print("Ошибка при загрузке данных из ClickHouse")
        return

    # Вычисление границ области
    lb, ub = data_loader.get_domain_bounds(X_u_train, X_f_train)

    print(f"Границы области: lb={lb}, ub={ub}")
    print(f"Размерности: X_u_train {X_u_train.shape}, u_train {u_train.shape}, X_f_train {X_f_train.shape}")

    # Параметры обучения
    lr_schedule, tf_optimizer, layers, tf_epochs = init_model_params()
    logger = Logger(frequency=200)  # Используем обычный логгер

    # Создание и обучение модели
    pinn = PINN(layers, tf_optimizer, logger, X_f_train, lb, ub)
    pinn.fit(X_u_train, u_train, tf_epochs)

if __name__ == "__main__":
    main()