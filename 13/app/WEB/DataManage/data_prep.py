import numpy as np
import pandas as pd
from pyDOE import lhs
from . import queries


class DataProcessor:

    @staticmethod
    def normalize_data(data, columns):
        try:
            normalized_data = data.copy()
            for col in columns:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val - min_val == 0:
                    normalized_data[col] = 0
                else:
                    normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
            return normalized_data
        except Exception as e:
            print(f"Ошибка при нормализации данных: {e}")
            return data


    @staticmethod
    def clean_csv_data(filepath):

        try:
            # Пробуем прочитать CSV с разными параметрами
            try:

                df = pd.read_csv(filepath)
                print("CSV прочитан с заголовком")
            except:

                df = pd.read_csv(filepath, header=None, dtype=float)
                print("CSV прочитан без заголовка")


                num_columns = len(df.columns)
                if num_columns >= 3:
                    df.columns = ['x', 't', 'value'][:num_columns]
                else:

                    for i in range(num_columns, 3):
                        df[f'col_{i}'] = 0.0

            # Проверяем и преобразуем данные
            print(f"Столбцы в данных: {list(df.columns)}")
            print(f"Первые несколько строк:\n{df.head()}")

            # Удаляем строки с нечисловыми данными
            for col in df.columns:
                # Пробуем преобразовать в числовой тип, ошибки превращаем в NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Удаляем строки с NaN значениями
            df_cleaned = df.dropna()

            if len(df_cleaned) == 0:
                print("Ошибка: после очистки не осталось данных")
                return None

            print(f"После очистки осталось {len(df_cleaned)} строк из {len(df)}")

            # Переименовываем столбцы в стандартные, если нужно
            if 'x' not in df_cleaned.columns or 't' not in df_cleaned.columns or 'value' not in df_cleaned.columns:
                if len(df_cleaned.columns) >= 3:
                    df_cleaned.columns = ['x', 't', 'value'][:len(df_cleaned.columns)]
                else:
                    print("Ошибка: недостаточно столбцов в данных")
                    return None

            # Преобразуем в float
            df_cleaned = df_cleaned.astype(float)

            return df_cleaned

        except Exception as e:
            print(f"Ошибка при чтении CSV файла: {e}")
            return None


class ClickHouseManager:

    def __init__(self, client, database='db'):
        self.client = client
        self.database = database

    def check_connection(self):

        try:
            self.client.execute('SELECT 1')
            return True
        except Exception as e:
            print(f"Ошибка подключения к ClickHouse: {e}")
            return False

    def clear_table(self, table_name):

        try:
            truncate_query = queries.truncate_table(self.database, table_name)
            self.client.execute(truncate_query)
            print(f"Таблица {table_name} успешно очищена")
            return True
        except Exception as e:
            print(f"Ошибка при очистке таблицы {table_name}: {e}")
            return False

    def prepare_data_for_insert(self, data, data_type):

        try:
            if data_type == "original_data":
                data_to_insert = []
                for _, row in data.iterrows():
                    # Проверяем и преобразуем значения, гарантируем что это float
                    x_val = float(row['x']) if pd.notna(row['x']) else 0.0
                    t_val = float(row['t']) if pd.notna(row['t']) else 0.0
                    value_val = float(row['value']) if pd.notna(row['value']) else 0.0
                    data_to_insert.append((x_val, t_val, value_val))
                return data_to_insert

            elif data_type == "train_points":
                data_to_insert = []
                for _, row in data.iterrows():
                    x_val = float(row['x']) if pd.notna(row['x']) else 0.0
                    t_val = float(row['t']) if pd.notna(row['t']) else 0.0
                    value_val = float(row['value']) if pd.notna(row['value']) else 0.0
                    data_to_insert.append((x_val, t_val, value_val))
                return data_to_insert

            elif data_type == "collocate_points":
                data_to_insert = []
                for row in data:
                    # Проверяем что row является итерируемым объектом
                    if hasattr(row, '__iter__') and not isinstance(row, (str, bytes)):
                        x_val = float(row[0]) if len(row) > 0 and pd.notna(row[0]) else 0.0
                        t_val = float(row[1]) if len(row) > 1 and pd.notna(row[1]) else 0.0
                    else:
                        x_val = 0.0
                        t_val = 0.0
                    data_to_insert.append((x_val, t_val))
                return data_to_insert

            else:
                raise ValueError(f"Неизвестный тип данных: {data_type}")

        except Exception as e:
            print(f"Ошибка при подготовке данных: {e}")
            print(f"Тип данных: {type(data)}")
            print(f"Пример данных: {data[:5] if hasattr(data, '__getitem__') else data}")
            return None

    def save_to_clickhouse(self, data, table_name, data_type, clear_existing=True):

        try:
            # Проверяем подключение
            if not self.check_connection():
                print("Ошибка: нет подключения к ClickHouse")
                return False

            print("Успешное подключение к ClickHouse")

            # Создаем БД и таблицу
            self.client.execute(queries.create_db(self.database))
            print(f"База данных {self.database} создана или уже существует")

            create_table_query = queries.create_table(self.database, table_name, data_type)
            self.client.execute(create_table_query)
            print(f"Таблица {table_name} создана или уже существует")

            # Очищаем таблицу если требуется
            if clear_existing:
                self.clear_table(table_name)

            # Подготавливаем данные для вставки
            data_to_insert = self.prepare_data_for_insert(data, data_type)
            if data_to_insert is None:
                print("Ошибка: не удалось подготовить данные для вставки")
                return False

            # Проверяем, что есть данные для вставки
            if len(data_to_insert) == 0:
                print("Предупреждение: нет данных для вставки")
                return False

            # Дополнительная проверка данных
            print(f"Подготовлено {len(data_to_insert)} записей для вставки")
            print(f"Первые 3 записи: {data_to_insert[:3]}")

            # Выполняем вставку
            insert_query = queries.insert_query(self.database, table_name, data_type)
            print(f"Выполняем запрос: {insert_query}")

            self.client.execute(insert_query, data_to_insert)

            print(f"Успешно сохранено {len(data_to_insert)} записей в таблицу {table_name}")
            return True

        except Exception as e:
            print(f"Ошибка при работе с ClickHouse: {e}")
            import traceback
            traceback.print_exc()
            return False


class DataGenerator:

    @staticmethod
    def generate_collocation_points(data, num_points=300):
        try:


            x = data['x']
            t = data['t']

            # Создаем сетку
            X, T = np.meshgrid(x, t)
            X_setka = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

            # Генерируем точки коллокации
            lb = X_setka.min(axis=0)
            ub = X_setka.max(axis=0)
            X_collocate = lb + (ub - lb) * lhs(2, num_points)

            return X_collocate

        except Exception as e:
            print(f"Ошибка при генерации точек коллокации: {e}")
            return None


class DataManager:

    def __init__(self, clickhouse_client, database='db'):
        self.processor = DataProcessor()
        self.generator = DataGenerator()
        self.db_manager = ClickHouseManager(clickhouse_client, database)

    def process_csv_data(self, filepath, num_collocation_points=300):
        try:
            # Читаем и очищаем CSV файл
            df = self.processor.clean_csv_data(filepath)
            if df is None:
                print("Ошибка: не удалось прочитать CSV файл")
                return False

            print(f"Успешно загружено {len(df)} строк данных")

            # Сохраняем оригинальные данные (очищаем существующие)
            if not self.db_manager.save_to_clickhouse(df, "OriginalData", "original_data"):
                return False

            # Нормализуем данные
            df_normalized = self.processor.normalize_data(df, ["x", "t"])

            # Сохраняем нормализованные тренировочные данные (очищаем существующие)
            if not self.db_manager.save_to_clickhouse(df_normalized, "TableTrainData", "train_points",
                                                      clear_existing=True):
                return False

            # Генерируем и сохраняем точки коллокации (очищаем существующие)
            collocation_data = self.generator.generate_collocation_points(df_normalized, num_collocation_points)
            if collocation_data is not None:
                success = self.db_manager.save_to_clickhouse(
                    collocation_data,
                    "CollocatePointsTable",
                    "collocate_points",
                    clear_existing=True
                )
                if success:
                    print("Все таблицы успешно созданы и заполнены")
                return success

            return False

        except Exception as e:
            print(f"Ошибка при обработке CSV файла: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_collocation_points(self, filepath, num_collocation_points=300):

        try:
            # Читаем существующие данные для получения границ
            df = self.processor.clean_csv_data(filepath)
            if df is None:
                print("Ошибка: не удалось прочитать CSV файл")
                return False

            df_normalized = self.processor.normalize_data(df, ["x", "t"])

            # Генерируем новые точки коллокации
            collocation_data = self.generator.generate_collocation_points(df_normalized, num_collocation_points)
            if collocation_data is not None:
                return self.db_manager.save_to_clickhouse(
                    collocation_data,
                    "CollocatePointsTable",
                    "collocate_points",
                    clear_existing=True
                )

            return False

        except Exception as e:
            print(f"Ошибка при обновлении точек коллокации: {e}")
            return False