import numpy as np
import pandas as pd
from pyDOE import lhs
import queries
from Client_ClickHouse import client


def get_values_from_csv(filepath):
    try:
        df = pd.read_csv(filepath)

        save_to_clickhouse(df, client,
                           database='db',
                           table_name="TableTrainData",
                           type="train_points")

        generate_data(df, N_r=300)

        return True


    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return None




def generate_data(data, N_r = 300):

    x = data['x']
    t = data['t']
    Value = data['value']

    X, T = np.meshgrid(x, t)

    try:
        X_setka = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

        # крайние значения
        lb = X_setka.min(axis=0)
        ub = X_setka.max(axis=0)

        X_collocate = lb + (ub - lb) * lhs(2, N_r)

        save_to_clickhouse(X_collocate, client,
                           database='db',
                           table_name="CollocatePointsTable",
                           type="collocate_points")

        return True


    except Exception as e:
        print(f"Ошибка при работе с ClickHouse: {e}")
        return None







def save_to_clickhouse(data, client,
                       database='db', table_name="TableTrainData", type = "train_points"):
    try:


        print("Успешное подключение к ClickHouse")

        client.execute(queries.create_db(database))
        print(f"База данных {database} создана или уже существует")

        create_table_query = queries.create_table(database, table_name, type)
        client.execute(create_table_query)
        print(f"Таблица {table_name} создана или уже существует")

        if type == "train_points":
            data_to_insert = []

            for _, row in data.iterrows():
                data_to_insert.append((
                    float(row['x']),
                    float(row['t']),
                    float(row['value']),
                    str(row['point_type'])
                ))


        elif type == "collocate_points":
            data_to_insert = [(float(row[0]), float(row[1])) for row in data]

        insert_query = queries.insert_query(database, table_name, type)
        client.execute(insert_query, data_to_insert)

        print(f"Успешно сохранено в ClickHouse")

        return client

    except Exception as e:
        print(f"Ошибка при работе с ClickHouse: {e}")
        return None
