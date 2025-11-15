import time
from datetime import datetime
from Client_ClickHouse import client
import queries

class Logger(object):
    def __init__(self, frequency=100, database='db', table_name="TableEpochLog", client=client):
        self.start_time = time.time()
        self.frequency = frequency
        self.client = client
        self.database = database
        self.table_name = table_name

        # Создаем таблицу при инициализации
        self._create_table()

    def _create_table(self):
        try:
            create_table_query = queries.create_table(self.database, self.table_name)
            self.client.execute(create_table_query)
            print(f"Table {self.database}.{self.table_name} created successfully")
        except Exception as e:
            print(f"Warning: Could not create table in ClickHouse: {e}")

    def __get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")

    def log_train_start(self, model):
        print("\nTraining started")
        print("================")

    def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
        if epoch % self.frequency == 0:
            print(
                f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e} {custom}")

            # Логируем в ClickHouse
            try:
                # Парсим custom строку чтобы извлечь data_loss и pde_loss
                data_loss = 0.0
                pde_loss = 0.0
                if "data_loss" in custom and "pde_loss" in custom:
                    parts = custom.split(",")
                    for part in parts:
                        if "data_loss" in part:
                            data_loss = float(part.split(":")[1].strip())
                        elif "pde_loss" in part:
                            pde_loss = float(part.split(":")[1].strip())

                insert_query = queries.insert_query(self.database, self.table_name, epoch, time.time(), self.start_time,
                                                    loss, data_loss, pde_loss)
                self.client.execute(insert_query)
            except Exception as e:
                print(f"Warning: Could not log to ClickHouse: {e}")

    def log_train_opt(self, name):
        print(f"—— Starting {name} optimization ——")

    def log_train_end(self, epoch, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} {custom}")