import boto3
import re
import os

from typing import Optional, Generator


class S3Connector:
    def __init__(
        self,
        bucket_name: str,
        access_key: str,
        secret_key: str,
        endpoint_url: str,
        region: str = "ru-central-1",
    ):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            region_name=region,
        )

    @staticmethod
    def filter_files_by_reg_exp(files: list[str], reg_exp: str) -> list[str]:
        return [file for file in files if re.fullmatch(reg_exp, os.path.basename(file))]

    def list_files(
        self, files: list[str], reg_exp: Optional[str] = None, recursive: bool = False
    ) -> list[str]:
        all_files = []

        for file in files:
            continuation_token = None
            while True:
                # Передаем ContinuationToken ТОЛЬКО если он не None
                params = {"Bucket": self.bucket_name, "Prefix": file}
                if continuation_token:
                    params["ContinuationToken"] = continuation_token

                response = self.s3_client.list_objects_v2(**params)

                if "Contents" in response:
                    all_files.extend([item["Key"] for item in response["Contents"]])

                if response.get("IsTruncated"):  # Если ещё есть файлы
                    continuation_token = response["NextContinuationToken"]
                else:
                    break  # Всё получили

        if reg_exp:
            all_files = self.filter_files_by_reg_exp(all_files, reg_exp)

        return all_files

    def get_files(
        self, files: list[str], reg_exp: Optional[str] = None, recursive: bool = False
    ) -> Generator[tuple[str, bytes], None, None]:
        files = self.list_files(files, reg_exp, recursive)
        for file in files:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file)
            content = response["Body"].read()
            yield file, content
