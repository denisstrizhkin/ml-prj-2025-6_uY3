from clickhouse_driver import Client

host='localhost'
port=9000
user='user'
password='123'


client = Client(
            host=host,
            port=port,
            user=user,
            password=password
        )
