import time
from datetime import datetime
from visualization import TrainingVisualizer


class Logger(object):
    def __init__(self, frequency=100, database='db', table_name="TableEpochLog"):
        self.start_time = time.time()
        self.frequency = frequency
        self.visualizer = TrainingVisualizer()




    def __get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")

    def log_train_start(self, model):
        print("\nTraining started")
        print("================")

    def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
        if epoch % self.frequency == 0:
            print(
                f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e} {custom}")

            # Парсим custom строку чтобы извлечь data_loss и pde_loss
            data_loss = 0.0
            pde_loss = 0.0
            if "data_loss" in custom and "pde_loss" in custom:
                parts = custom.split(",")
                for part in parts:
                    if "data_loss" in part:
                        data_loss = float(part.split(":")[1].strip().split()[0])
                    elif "pde_loss" in part:
                        pde_loss = float(part.split(":")[1].strip().split()[0])

            # Добавляем данные в визуализатор
            self.visualizer.add_epoch_data(epoch, float(loss), data_loss, pde_loss)



    def log_train_opt(self, name):
        print(f"—— Starting {name} optimization ——")

    def log_train_end(self, epoch, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} {custom}")

    def get_training_plot(self) -> str:
        return self.visualizer.create_training_plot()

    def get_training_stats(self) -> dict:
        return self.visualizer.get_training_stats()