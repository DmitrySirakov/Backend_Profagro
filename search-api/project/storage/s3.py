import boto3
import re
import os

from typing import Optional, Generator
from project.settings import _settings


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
        return [file for file in files if re.match(reg_exp, os.path.basename(file))]

    def list_files(
        self, files: list[str], reg_exp: Optional[str] = None, recursive: bool = False
    ) -> list[str]:
        if recursive:
            new_files = []
            for file in files:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name, Prefix=file
                )
                subfiles = [item["Key"] for item in response.get("Contents", [])]
                new_files.extend(subfiles)
            files = new_files

        if reg_exp is not None:
            files = self.filter_files_by_reg_exp(files, reg_exp)

        return files

    def get_files(
        self, files: list[str], reg_exp: Optional[str] = None, recursive: bool = False
    ) -> Generator[tuple[str, bytes], None, None]:
        files = self.list_files(files, reg_exp, recursive)
        for file in files:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file)
            content = response["Body"].read()
            yield file, content

    def list_folders(self, prefix: str = "", delimiter: str = "/") -> list[str]:
        """
        Возвращает список всех папок в бакете или под указанным префиксом.

        :param prefix: Префикс для поиска папок (по умолчанию пустая строка, что означает корневой уровень).
        :param delimiter: Разделитель для определения папок (по умолчанию "/").
        :return: Список папок.
        """
        paginator = self.s3_client.get_paginator("list_objects_v2")
        result = set()
        for page in paginator.paginate(
            Bucket=self.bucket_name, Prefix=prefix, Delimiter=delimiter
        ):
            for common_prefix in page.get("CommonPrefixes", []):
                result.add(common_prefix["Prefix"])

        return sorted(result)


storage = S3Connector(
    bucket_name=_settings.indexer_s3_bucket,
    access_key=_settings.indexer_s3_access_key,
    secret_key=_settings.indexer_s3_secret_key,
    endpoint_url=_settings.indexer_s3_endpoint,
)
