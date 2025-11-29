from notebook_model_test.py_files.data_prep import data_prep

if __name__ == '__main__':
    filepath = 'data_prep/buckley_pde_data.csv'
    data = data_prep.get_values_from_csv(filepath)
    data_prep.save_to_clickhouse(data)