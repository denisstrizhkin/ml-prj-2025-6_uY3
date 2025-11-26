def create_db(database):
    return f'CREATE DATABASE IF NOT EXISTS {database}'

def create_table(database, table_name, type):

    if type == "train_points":
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


def insert_query(database, table_name, type):
    if type == "train_points":
        return f"INSERT INTO {database}.{table_name} (x, t, value) VALUES"
    elif type == "collocate_points":
        return f"INSERT INTO {database}.{table_name} (x, t) VALUES"