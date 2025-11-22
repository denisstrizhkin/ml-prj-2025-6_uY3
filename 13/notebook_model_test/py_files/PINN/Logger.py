import time
from datetime import datetime
from Client_ClickHouse import client


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
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.table_name} (
                epoch UInt32,
                elapsed_seconds Float64,
                total_loss Float64,
                data_loss Float64,
                pde_loss Float64,
                weights_data_loss Float64,
                weights_pde_loss Float64,
                timestamp DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (epoch)
            """
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

                insert_query = f"""
                INSERT INTO {self.database}.{self.table_name} 
                (epoch, elapsed_seconds, total_loss, data_loss, pde_loss, weights_data_loss, weights_pde_loss)
                VALUES ({epoch}, {time.time() - self.start_time}, {float(loss)}, {data_loss}, {pde_loss}, 1.0, 1.0)
                """
                self.client.execute(insert_query)
            except Exception as e:
                print(f"Warning: Could not log to ClickHouse: {e}")

    def log_train_opt(self, name):
        print(f"—— Starting {name} optimization ——")

    def log_train_end(self, epoch, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} {custom}")