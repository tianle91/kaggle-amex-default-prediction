import pyspark.sql.functions as F
from pyspark.sql.window import Window
from spark_utils import get_spark_session

# Run format_data.py first if you haven't done so yet.
spark = get_spark_session()
window_latest_date_by_id = Window.partitionBy('customer_ID').orderBy(F.col('S_2').desc())

if __name__ == '__main__':
    for p in [
        'data/amex-default-prediction/test_data',
        'data/amex-default-prediction/train_data',
    ]:
        df = spark.read.parquet(p)
        df = df.withColumn('S_2_rank_by_latest',
                           F.rank().over(window_latest_date_by_id))
        out_p = p.replace('data/', 'data_transformed/')
        df.write.mode('overwrite').parquet(out_p)
        print(f'Wrote: {p}\n->{out_p}')
