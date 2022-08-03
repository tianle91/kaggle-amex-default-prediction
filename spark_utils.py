from typing import Optional

import psutil
from pyspark.sql import SparkSession


def get_spark_session(num_cores: Optional[int] = None, memory: Optional[int] = None) -> SparkSession:
    num_cores = '*' if num_cores is None else num_cores
    memory = psutil.virtual_memory().total if memory is None else memory

    return (
        SparkSession
        .builder
        .master(f'local[{num_cores}]')
        .config('spark.driver.memory', str(memory))
        .config('spark.driver.maxResultSize', '0')
        # enables arrow-related features if using JDK 11
        # https://spark.apache.org/docs/latest/api/python/getting_started/install.html
        .config('Dio.netty.tryReflectionSetAccessible', 'true')
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        .getOrCreate()
    )


class SparkSessionContext:
    def __init__(self):
        pass

    def __enter__(self) -> SparkSession:
        self.spark = get_spark_session()
        return self.spark

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.spark.stop()
