def create_db(database):
    return f'CREATE DATABASE IF NOT EXISTS {database}'

def create_table(database, table_name, type):
    if type == "train_points" or type == "original_data":
        return f"""
        CREATE TABLE IF NOT EXISTS {database}.{table_name}
        (
            x Float64,
            t Float64,
            value Float64
        )
        ENGINE = MergeTree()
        ORDER BY (x, t)
        """
    elif type == "collocate_points":
        return f"""
                CREATE TABLE IF NOT EXISTS {database}.{table_name}
                (
                    x Float64,
                    t Float64
                )
                ENGINE = MergeTree()
                ORDER BY (x, t)
                """

def drop_table(database, table_name):
    return f"DROP TABLE IF EXISTS {database}.{table_name}"

def truncate_table(database, table_name):
    return f"TRUNCATE TABLE IF EXISTS {database}.{table_name}"

def insert_query(database, table_name, data_type):
    if data_type == "original_data":
        return f"INSERT INTO {database}.{table_name} (x, t, value) VALUES"
    elif data_type == "train_points":
        return f"INSERT INTO {database}.{table_name} (x, t, value) VALUES"
    elif data_type == "collocate_points":
        return f"INSERT INTO {database}.{table_name} (x, t) VALUES"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")