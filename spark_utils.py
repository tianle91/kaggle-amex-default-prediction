from pyspark.sql import SparkSession


def get_spark_session(num_cores: int = 8, gigs_ram: int = 16) -> SparkSession:
    return (
        SparkSession
        .builder
        .master(f'local[{num_cores}]')
        .config('spark.driver.memory', f'{gigs_ram}g')
        .config('spark.driver.maxResultSize', f'{gigs_ram}g')
        # enables arrow-related features if using JDK 11
        # https://spark.apache.org/docs/latest/api/python/getting_started/install.html
        .config('Dio.netty.tryReflectionSetAccessible', 'true')
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        # compression
        .config('spark.shuffle.compress', 'true')
        .config('spark.shuffle.spill.compress', 'true')
        .config('spark.broadcast.compress', 'true')
        .config('spark.checkpoint.compress', 'true')
        .config('spark.rdd.compress', 'true')
        .getOrCreate()
    )
