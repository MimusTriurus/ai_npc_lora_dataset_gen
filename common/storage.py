from minio import Minio
from pathlib import Path
import io
import json

class MinioStorage:
    def __init__(self, endpoint, access_key, secret_key, bucket, secure=False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket = bucket

        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def upload_file(self, key: str, path: str):
        self.client.fput_object(self.bucket, key, path)

    def download_file(self, key: str, path: str):
        self.client.fget_object(self.bucket, key, path)

    def upload_bytes(self, key: str, data: bytes):
        self.client.put_object(
            self.bucket,
            key,
            io.BytesIO(data),
            length=len(data),
        )

    def upload_json(self, key: str, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self.upload_bytes(key, data)

    def list(self, prefix=""):
        return [
            obj.object_name
            for obj in self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
        ]

    def exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except:
            return False
