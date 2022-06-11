import pyspark.sql.functions as F
from pyspark.sql.window import Window
from spark_utils import get_spark_session

# Run format_data.py first if you haven't done so yet.
spark = get_spark_session()
window_latest_date_by_id = Window.partitionBy(
    'customer_ID').orderBy(F.col('S_2').desc())
RANKED_S_2 = '__S_2_rank_by_latest__'

if __name__ == '__main__':
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
