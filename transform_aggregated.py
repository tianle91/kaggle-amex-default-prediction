import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType, StringType
from pyspark.sql.window import Window
from scipy import stats

from spark_utils import get_spark_session


def get_mode(list_of_values) -> str:
    if len(list_of_values) > 0:
        return str(stats.mode(list_of_values)[0][0])


def get_aggregated(df: DataFrame) -> DataFrame:

    window_latest_date_by_id = (
        Window
        .partitionBy('customer_ID')
        .orderBy(F.col('S_2'))
        .rowsBetween(
            Window.unboundedPreceding,
            Window.unboundedFollowing,
        )
    )

    generic_selected = []
    for c in df.columns:
        if c in ['customer_ID', 'S_2']:
            continue
        generic_selected += [
            # keep this for aggregation later
            c,
            # these are ordered by statement date
            F.first(c).over(window_latest_date_by_id).alias(f'{c}_first'),
            F.last(c).over(window_latest_date_by_id).alias(f'{c}_last'),
        ]

    generic_aggregated = []
    for c in df.columns:
        if c in ['customer_ID', 'S_2']:
            continue
        generic_aggregated += [
            # don't do anything to these
            F.first(f'{c}_first').alias(f'{c}_first'),
            F.first(f'{c}_last').alias(f'{c}_last'),
        ]
        # aggregation varies depending on data type
        if isinstance(df.schema[c].dataType, StringType):
            generic_aggregated.append(F.udf(get_mode, 'string')(
                F.collect_list(c)).alias(f'{c}_mode'))
        elif isinstance(df.schema[c].dataType, FloatType):
            generic_aggregated.append(F.mean(c).alias(f'{c}_mean'))
        else:
            raise ValueError(
                f'Unexpected {c} due to {df.schema[c].dataType} not being string or float')

    return (
        df.select(
            'customer_ID',
            F.first('S_2').over(window_latest_date_by_id).alias('S_2_first'),
            F.last('S_2').over(window_latest_date_by_id).alias('S_2_last'),
            *generic_selected
        )
        .groupBy('customer_ID', 'S_2_first', 'S_2_last')
        .agg(
            F.count('*').alias('num_statements'),
            *generic_aggregated
        )
    )


if __name__ == '__main__':

    spark = get_spark_session()

    # Run format_data.py first if you haven't done so yet.
    for p in [
        'data/amex-default-prediction/test_data',
        'data/amex-default-prediction/train_data',
    ]:
        print(f'Aggregating for {p}')
        df = spark.read.parquet(p)
        num_parts = df.rdd.getNumPartitions()
        out_p = (
            p
            .replace('data/', 'data_transformed/')
            .replace('_data', '_data_aggregated')
        )
        (
            get_aggregated(df)
            .repartition(num_parts)
            .write
            .mode('overwrite')
            .parquet(out_p)
        )
        print(f'Wrote to {out_p}')

        df_aggregated = spark.read.parquet(out_p)
        assert df_aggregated.count() == df_aggregated.select('customer_ID').distinct().count()


    # validations
    test_data = spark.read.parquet('data_transformed/amex-default-prediction/test_data_aggregated')
    train_data = spark.read.parquet('data_transformed/amex-default-prediction/train_data_aggregated')
    train_labels = spark.read.parquet('data/amex-default-prediction/train_labels')
    sample_submission = spark.read.parquet('data/amex-default-prediction/sample_submission')

    assert train_data.count() == train_data.join(train_labels, on='customer_ID', how='inner').count()
    assert test_data.count() == test_data.join(sample_submission, on='customer_ID', how='inner').count()

    spark.stop()
