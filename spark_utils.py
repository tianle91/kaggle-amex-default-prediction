from pyspark.sql import SparkSession


def get_spark_session(num_cores: int = 8, gigs_ram: int = 16) -> SparkSession:
    return (
        SparkSession
        .builder
        .master(f'local[{num_cores}]')
        .config('spark.driver.memory', f'{gigs_ram}g')
        .getOrCreate()
    )
