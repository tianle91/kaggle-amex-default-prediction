import pyspark.sql.functions as F
from pyspark.sql.window import Window

from spark_utils import get_spark_session

RANKED_S_2 = '__S_2_rank_by_latest__'

if __name__ == '__main__':

    spark = get_spark_session()
    window_latest_date_by_id = (
        Window
        .partitionBy('customer_ID')
        .orderBy(F.col('S_2').desc())
    )

    # Run format_data.py first if you haven't done so yet.
    for p in [
        'data/amex-default-prediction/test_data',
        'data/amex-default-prediction/train_data',
    ]:
        print(f'Filtering for latest statement for {p}')
        df = spark.read.parquet(p)
        num_parts = df.rdd.getNumPartitions()
        df = (
            df
            .withColumn(RANKED_S_2, F.rank().over(window_latest_date_by_id))
            .filter(F.col(RANKED_S_2) == 1)
            .drop(RANKED_S_2)
        )
        out_p = (
            p
            .replace('data/', 'data_transformed/')
            .replace('_data', '_data_latest')
        )
        (
            df
            .repartition(num_parts)
            .write
            .mode('overwrite')
            .parquet(out_p)
        )
        print(f'Wrote to {out_p}')

    # validations
    test_data = spark.read.parquet(
        'data_transformed/amex-default-prediction/test_data_latest')
    train_data = spark.read.parquet(
        'data_transformed/amex-default-prediction/train_data_latest')
    train_labels = spark.read.parquet(
        'data/amex-default-prediction/train_labels')
    sample_submission = spark.read.parquet(
        'data/amex-default-prediction/sample_submission')

    assert train_data.count() == train_data.select('customer_ID').distinct().count()
    assert train_labels.count() == train_labels.select(
        'customer_ID').distinct().count()
    assert train_data.count() == train_data.join(
        train_labels, on='customer_ID', how='inner').count()

    assert test_data.count() == test_data.select('customer_ID').distinct().count()
    assert sample_submission.count() == sample_submission.select(
        'customer_ID').distinct().count()
    assert test_data.count() == test_data.join(
        sample_submission, on='customer_ID', how='inner').count()

    spark.stop()
