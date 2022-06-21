import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType, StringType
from pyspark.sql.window import Window
from scipy import stats

from spark_utils import get_spark_session


def get_mode(list_of_values) -> str:
    if len(list_of_values) > 0:
        return str(stats.mode(list_of_values)[0][0])


def get_window_features(df: DataFrame) -> DataFrame:
    window = Window.partitionBy('customer_ID').orderBy(F.col('S_2'))
    generic_selected = []
    for c in df.columns:
        if c in ['customer_ID', 'S_2']:
            continue
        generic_selected += [
            # keep this for aggregation later
            c,
            F.lag(c, offset=-1, default=None).over(window).alias(f'{c}_previous'),
        ]
    df_windowed_features = (
        df
        .select(
            'customer_ID',
            'S_2',
            F.datediff(
                'S_2', F.lag('S_2', offset=-1).over(window)
            ).alias('S_2_days_since_previous'),
            *generic_selected
        )
        .join(
            df.groupBy('customer_ID').agg(F.max('S_2').alias('S_2_last')),
            on='customer_ID',
            how='inner',
        )
        .filter(F.col('S_2') == F.col('S_2_last'))
    )
    df_windowed_features = df_windowed_features.select(
        *df_windowed_features.columns,
        *[
            (F.col(f'{c}_previous') != F.col(c)).alias(f'{c}_changed')
            for c in df.columns if c not in ['customer_ID', 'S_2']
        ]
    )
    return df_windowed_features


def get_summary_features(df: DataFrame) -> DataFrame:
    generic_aggregated = []
    for c in df.columns:
        if c in ['customer_ID', 'S_2']:
            continue
        generic_aggregated += [
            F.size(F.collect_set(c)).alias(f'{c}_num_unique'),
        ]
        # aggregation varies depending on data type
        if isinstance(df.schema[c].dataType, StringType):
            generic_aggregated += [
                F.udf(get_mode, 'string')(F.collect_list(c)).alias(f'{c}_mode')
            ]
        elif isinstance(df.schema[c].dataType, FloatType):
            generic_aggregated += [
                F.mean(c).alias(f'{c}_min'),
                F.mean(c).alias(f'{c}_max'),
            ]
        else:
            raise ValueError(
                f'Unexpected {c} due to {df.schema[c].dataType} not being string or float')
    df_aggregated_features = (
        df
        .groupBy('customer_ID')
        .agg(
            F.count('*').alias('num_statements'),
            *generic_aggregated,
        )
    )
    df_aggregated_features = (
        df_aggregated_features
        .select(
            *df_aggregated_features.columns,
            *[
                (F.col('num_statements') - F.col(f'{c}_num_unique')).alias(f'{c}_num_duplicates')
                for c in df.columns if c not in ['customer_ID', 'S_2']
            ]
        )
    )
    return df_aggregated_features


def get_transformed(df: DataFrame) -> DataFrame:
    df_windowed_features = get_window_features(df)
    df_aggregated_features = get_summary_features(df)
    return df_windowed_features.join(
        df_aggregated_features,
        on='customer_ID',
        how='inner',
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
            get_transformed(df)
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
