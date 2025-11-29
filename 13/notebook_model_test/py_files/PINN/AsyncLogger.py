import threading
from queue import Queue
from Client_ClickHouse import client
import time
from datetime import datetime


class AsyncLogger(object):
    def __init__(self, frequency=100, database='db', table_name="TableEpochLog", client=client):
        # Инициализируем базовые атрибуты
        self.start_time = time.time()
        self.frequency = frequency
        self.database = database
        self.table_name = table_name
        self.client = client
        self.error_fn = None

        # Асинхронные компоненты
        self.log_queue = Queue()
        self.worker_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.worker_thread.start()

        # Создаем таблицу если не существует
        self._setup_database()

    def _setup_database(self):
        """Создает базу данных и таблицу если они не существуют"""
        try:
            # Создаем базу данных если не существует
            self.client.execute(f'CREATE DATABASE IF NOT EXISTS {self.database}')

            # Создаем таблицу для логов с правильной структурой
            create_table_query = f'''
            CREATE TABLE IF NOT EXISTS {self.database}.{self.table_name} (
                event_time DateTime DEFAULT now(),
                event_type String,
                epoch UInt32,
                loss Float64,
                elapsed_time Float64,
                custom_info String,
                model_info String,
                optimizer_name String
            ) ENGINE = MergeTree()
            ORDER BY (event_time, event_type)
            '''
            self.client.execute(create_table_query)
            print(f"Table {self.database}.{self.table_name} created successfully")

        except Exception as e:
            print(f"Warning: Could not setup ClickHouse database: {e}")

    def __get_elapsed(self):
        return time.time() - self.start_time

    def _log_worker(self):
        while True:
            log_data = self.log_queue.get()
            if log_data is None:
                break
            try:
                self.client.execute(log_data['query'], log_data['params'])
            except Exception as e:
                print(f"Warning: Could not insert log into ClickHouse: {e}")
            self.log_queue.task_done()

    def _insert_log(self, event_type="", epoch=0, loss=0.0, elapsed_time=0.0, custom_info="", model_info="",
                    optimizer_name=""):
        try:
            query = f'''
            INSERT INTO {self.database}.{self.table_name} 
            (event_type, epoch, loss, elapsed_time, custom_info, model_info, optimizer_name)
            VALUES (%(event_type)s, %(epoch)s, %(loss)s, %(elapsed_time)s, %(custom_info)s, %(model_info)s, %(optimizer_name)s)
            '''

            self.log_queue.put({
                'query': query,
                'params': {
                    'event_type': event_type,
                    'epoch': epoch,
                    'loss': loss,
                    'elapsed_time': elapsed_time,
                    'custom_info': custom_info,
                    'model_info': model_info,
                    'optimizer_name': optimizer_name
                }
            })
        except Exception as e:
            print(f"Warning: Could not queue log for ClickHouse: {e}")

    # Добавляем недостающие методы
    def log_train_start(self, model):
        """Логирование начала обучения"""
        print("\nTraining started")
        print("================")
        model_info = f"Model with {len(model.u_model.layers)} layers"
        self._insert_log(
            event_type="train_start",
            model_info=model_info,
            elapsed_time=self.__get_elapsed()
        )

    def log_train_opt(self, optimizer_name):
        """Логирование используемого оптимизатора"""
        print(f"—— Starting {optimizer_name} optimization ——")
        self._insert_log(
            event_type="train_opt",
            optimizer_name=optimizer_name,
            elapsed_time=self.__get_elapsed()
        )

    def log_train_epoch(self, epoch, loss, custom_info):
        """Логирование эпохи обучения"""
        if epoch % self.frequency == 0:
            print(f"tf_epoch = {epoch:6d}  elapsed = {self.__get_elapsed():.2f}s  loss = {loss:.4e} {custom_info}")

            # Преобразуем тензор в float
            loss_value = float(loss.numpy()) if hasattr(loss, 'numpy') else float(loss)

            self._insert_log(
                event_type="train_epoch",
                epoch=epoch,
                loss=loss_value,
                custom_info=custom_info,
                elapsed_time=self.__get_elapsed()
            )

    def log_train_end(self, epoch, custom=""):
        """Логирование завершения обучения"""
        print("==================")
        print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed():.2f}s {custom}")
        self._insert_log(
            event_type="train_end",
            epoch=epoch,
            elapsed_time=self.__get_elapsed(),
            custom_info=custom
        )

    def close(self):
        """Завершение работы логгера"""
        self.log_queue.put(None)
        self.worker_thread.join()