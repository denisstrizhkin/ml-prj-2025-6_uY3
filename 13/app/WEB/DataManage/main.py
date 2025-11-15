import data_prep
from Client_ClickHouse import client
filepath = "test1.csv"
if __name__ == "__main__":
    data_manager = data_prep.DataManager(client, database='db')
    success = data_manager.process_csv_data(filepath, num_collocation_points=300)

    if success:
        print("Все операции выполнены успешно!")
    else:
        print("Произошла ошибка при обработке данных!")
