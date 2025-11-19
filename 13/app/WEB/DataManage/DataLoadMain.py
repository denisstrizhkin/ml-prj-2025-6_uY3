from . import data_prep
from . Client_ClickHouse import client
#filepath = "test1.csv"

def data_main(filepath, num_collocation_points):
    data_manager = data_prep.DataManager(client, database='db')
    success = data_manager.process_csv_data(filepath, num_collocation_points)

    if success:
        print("Все операции выполнены успешно!")
    else:
        print("Произошла ошибка при обработке данных!")

    return success