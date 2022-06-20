import pyspark.sql.functions as F
from pyspark.sql.window import Window

from spark_utils import get_spark_session

S_2_max = 'S_2_max'

if __name__ == '__main__':

    spark = get_spark_session()
    window = (
        Window
        .partitionBy('customer_ID')
        .orderBy(F.col('S_2'))
        .rangeBetween(
            Window.unboundedPreceding,
            Window.unboundedFollowing,
        )
    )

    # Run format_data.py first if you haven't done so yet.
    for p in [
        'data/amex-default-prediction/test_data',
        'data/amex-default-prediction/train_data',
    ]:
        print(f'Filtering for latest statement for {p}')
        df = spark.read.parquet(p)
        num_parts = df.rdd.getNumPartitions()

        df_latest = (
            df
            .withColumn(S_2_max, F.max('S_2').over(window))
            .withColumn('num_statements', F.count('*').over(window))
            .filter(F.col(S_2_max) == F.col('S_2'))
            .drop(S_2_max)
        )
        out_p = (
            p
            .replace('data/', 'data_transformed/')
            .replace('_data', '_data_latest')
        )
        (
            df_latest
            .repartition(num_parts)
            .write
            .mode('overwrite')
            .parquet(out_p)
        )
        print(f'Wrote to {out_p}')

        df_latest = spark.read.parquet(out_p)
        assert df_latest.count() == df_latest.select('customer_ID').distinct().count()

    # validations
    test_data = spark.read.parquet('data_transformed/amex-default-prediction/test_data_latest')
    train_data = spark.read.parquet('data_transformed/amex-default-prediction/train_data_latest')
    train_labels = spark.read.parquet('data/amex-default-prediction/train_labels')
    sample_submission = spark.read.parquet('data/amex-default-prediction/sample_submission')
    
    assert train_data.count() == train_data.join(train_labels, on='customer_ID', how='inner').count()
    assert test_data.count() == test_data.join(sample_submission, on='customer_ID', how='inner').count()

    spark.stop()
