from Client_ClickHouse import client
import numpy as np
import pandas as pd


class PINNDataLoader:
    def __init__(self, client, database='db'):
        self.client = client
        self.database = database

    def load_training_data(self):
        """Загрузка тренировочных данных (граничные и начальные условия)"""
        try:
            query = f"SELECT x, t, value FROM {self.database}.TableTrainData"
            result = self.client.execute(query)
            df = pd.DataFrame(result, columns=['x', 't', 'value'])

            X_u_train = df[['x', 't']].values.astype(np.float32)
            u_train = df[['value']].values.astype(np.float32)

            print(f"Загружено {len(X_u_train)} тренировочных точек")
            return X_u_train, u_train

        except Exception as e:
            print(f"Ошибка при загрузке тренировочных данных: {e}")
            return None, None

    def load_collocation_points(self):
        """Загрузка коллокационных точек"""
        try:
            query = f"SELECT x, t FROM {self.database}.CollocatePointsTable"
            result = self.client.execute(query)
            X_f_train = np.array(result, dtype=np.float32)

            print(f"Загружено {len(X_f_train)} коллокационных точек")
            return X_f_train

        except Exception as e:
            print(f"Ошибка при загрузке коллокационных точек: {e}")
            return None

    def get_domain_bounds(self, X_u_train, X_f_train):
        all_points = np.vstack([X_u_train, X_f_train])
        lb = all_points.min(axis=0)
        ub = all_points.max(axis=0)
        return lb, ub