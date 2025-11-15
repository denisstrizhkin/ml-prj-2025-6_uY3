def create_db(database):
    return f'CREATE DATABASE IF NOT EXISTS {database}'

def create_table(database, table_name):
    return f"""
                CREATE TABLE IF NOT EXISTS {database}.{table_name} (
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



def insert_query(database, table_name, epoch, time, start_time, loss, data_loss, pde_loss):
    return f"""
                    INSERT INTO {database}.{table_name} 
                    (epoch, elapsed_seconds, total_loss, data_loss, pde_loss, weights_data_loss, weights_pde_loss)
                    VALUES ({epoch}, {time - start_time}, {float(loss)}, {data_loss}, {pde_loss}, 1.0, 1.0)
                    """