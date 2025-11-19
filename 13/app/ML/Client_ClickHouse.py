from clickhouse_driver import Client
import os

# Используем переменные окружения для подключения
host = os.getenv('CLICKHOUSE_HOST', 'clickhouse')
port = int(os.getenv('CLICKHOUSE_PORT', 9000))
user = os.getenv('CLICKHOUSE_USER', 'user')
password = os.getenv('CLICKHOUSE_PASSWORD', '123')

client = Client(
    host=host,
    port=port,
    user=user,
    password=password
)