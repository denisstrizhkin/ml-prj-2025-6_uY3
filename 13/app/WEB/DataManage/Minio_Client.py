from minio import Minio
import os


minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio-service:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "admin123"),
    secure=False
)