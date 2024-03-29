import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from format_data import CATEGORICAL_VARIABLES, FEATURE_VARIABLES
from spark_utils import get_spark_session

WINDOW_FEATURE_PREVIOUS_VARIABLES = [
    f'{c}_previous'
    for c in FEATURE_VARIABLES
]
WINDOW_FEATURE_CHANGED_VARIABLES = [
    f'{c}_changed'
    for c in FEATURE_VARIABLES
]
WINDOW_FEATURE_VARIABLES = [
    'S_2_days_since_previous',
    *FEATURE_VARIABLES,
    *WINDOW_FEATURE_PREVIOUS_VARIABLES,
    *WINDOW_FEATURE_CHANGED_VARIABLES,
]
WINDOW_FEATURE_CATEGORICAL_VARIABLES = [
    *CATEGORICAL_VARIABLES,
    *[f'{c}_previous' for c in CATEGORICAL_VARIABLES],
]


def get_window_features(df: DataFrame) -> DataFrame:
    window = Window.partitionBy('customer_ID').orderBy(F.col('S_2'))
    select_cols = [
        F.datediff(
            'S_2',
            F.lag('S_2', offset=1).over(window)
        ).alias('S_2_days_since_previous'),
        *FEATURE_VARIABLES,
        *[
            F.lag(c, offset=1).over(window).alias(f'{c}_previous')
            for c in FEATURE_VARIABLES
        ]
    ]
    __S_2_LAST_COL__ = '__S_2_LAST_COL__'
    out = (
        df
        .select('customer_ID', 'S_2', *select_cols)
        .join(
            (
                df
                .groupBy('customer_ID')
                .agg(
                    F.max('S_2').alias(__S_2_LAST_COL__)
                )
            ),
            on='customer_ID',
            how='inner',
        )
        .filter(F.col('S_2') == F.col(__S_2_LAST_COL__))
        .drop(__S_2_LAST_COL__)
    )
    out = out.select(
        *out.columns,
        *[
            F.when(
                F.col(f'{c}_previous') == F.col(c),
                F.lit(0.)
            ).otherwise(
                F.lit(1.)
            ).alias(f'{c}_changed')
            for c in FEATURE_VARIABLES
        ],
    )
    return out


SUMMARY_FEATURE_NUM_UNIQUE_VARIABLES = [
    f'{c}_num_unique'
    for c in FEATURE_VARIABLES
]
SUMMARY_FEATURE_CATEGORICAL_VARIABLES = [
    f'{c}_mode'
    for c in CATEGORICAL_VARIABLES
]
SUMMARY_FEATURE_MIN_VARIABLES = [
    f'{c}_min'
    for c in FEATURE_VARIABLES if c not in CATEGORICAL_VARIABLES
]
SUMMARY_FEATURE_MAX_VARIABLES = [
    f'{c}_max'
    for c in FEATURE_VARIABLES if c not in CATEGORICAL_VARIABLES
]
SUMMARY_FEATURE_MEAN_VARIABLES = [
    f'{c}_mean'
    for c in FEATURE_VARIABLES if c not in CATEGORICAL_VARIABLES
]
SUMMARY_FEATURE_VARIABLES = [
    'num_statements',
    *SUMMARY_FEATURE_NUM_UNIQUE_VARIABLES,
    *SUMMARY_FEATURE_CATEGORICAL_VARIABLES,
    *SUMMARY_FEATURE_MIN_VARIABLES,
    *SUMMARY_FEATURE_MAX_VARIABLES,
    *SUMMARY_FEATURE_MEAN_VARIABLES,
]


def get_mode(list_of_things):
    if len(list_of_things) == 0:
        return None
    mode_values = pd.Series(list_of_things).mode()
    if len(mode_values) == 0:
        return None
    else:
        return mode_values.iloc[0]


def get_summary_features(df: DataFrame) -> DataFrame:
    agg_cols = [F.count('*').alias('num_statements')]
    mode_udf = F.udf(get_mode, 'string')
    for c in FEATURE_VARIABLES:
        agg_cols.append(F.size(F.collect_set(c)).alias(f'{c}_num_unique'))
        if c in CATEGORICAL_VARIABLES:
            agg_cols.append(mode_udf(F.collect_list(c)).alias(f'{c}_mode'))
        else:
            agg_cols += [
                F.min(c).alias(f'{c}_min'),
                F.max(c).alias(f'{c}_max'),
                F.mean(c).alias(f'{c}_mean'),
            ]
    return (
        df
        .groupBy('customer_ID')
        .agg(*agg_cols)
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
            get_window_features(df).join(
                get_summary_features(df),
                on='customer_ID',
                how='inner',
            )
            .repartition(num_parts)
            .write
            .mode('overwrite')
            .parquet(out_p)
        )
        print(f'Wrote to {out_p}')

        df_aggregated = spark.read.parquet(out_p)
        assert df_aggregated.count() == df_aggregated.select(
            'customer_ID').distinct().count()

    # validations
    test_data = spark.read.parquet(
        'data_transformed/amex-default-prediction/test_data_aggregated')
    train_data = spark.read.parquet(
        'data_transformed/amex-default-prediction/train_data_aggregated')
    train_labels = spark.read.parquet(
        'data/amex-default-prediction/train_labels')
    sample_submission = spark.read.parquet(
        'data/amex-default-prediction/sample_submission')

    assert train_data.count() == train_data.join(
        train_labels, on='customer_ID', how='inner').count()
    assert test_data.count() == test_data.join(
        sample_submission, on='customer_ID', how='inner').count()

    spark.stop()

    # # nohup python transform_aggregated.py >log.out 2>&1 &
    # [2] 4983
