import numpy as np
import pandas as pd
from clickhouse_driver import Client

def get_values_from_csv(filepath):
    try:
        df_test = pd.read_csv(filepath)
        return df_test
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return None

def save_to_clickhouse(data, host='localhost', port=9000, 
                      user='user', password='123', 
                      database='db', table_name="TableTrainData"):

    try:
        # Подключаемся к ClickHouse
        client = Client(
            host=host,
            port=port,
            user=user,
            password=password
        )
        
        print("Успешное подключение к ClickHouse")
    
        client.execute(f'CREATE DATABASE IF NOT EXISTS {database}')
        print(f"База данных {database} создана или уже существует")
        
        # Создаем таблицу
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {database}.{table_name}
        (
            x Float64,
            t Float64,
            value Float64,
            point_type Enum8('initial' = 1, 'left_boundary' = 2, 'right_boundary' = 3)
        )
        ENGINE = MergeTree()
        ORDER BY (x, t)
        """
        client.execute(create_table_query)
        print(f"Таблица {table_name} создана или уже существует")
        
        # Подготавливаем данные в правильном формате
        data_to_insert = []
        for _, row in data.iterrows():
            data_to_insert.append((
                float(row['x']),
                float(row['t']),
                float(row['value']),
                str(row['point_type'])
            ))
        
        # Вставляем данные
        insert_query = f"INSERT INTO {database}.{table_name} (x, t, value, point_type) VALUES"
        client.execute(insert_query, data_to_insert)
        
        # Проверяем количество записей
        count_result = client.execute(f"SELECT count() FROM {database}.{table_name}")
        print(f"Успешно сохранено {count_result[0][0]} записей в ClickHouse")
        
        return client
        
    except Exception as e:
        print(f"Ошибка при работе с ClickHouse: {e}")
        return None
