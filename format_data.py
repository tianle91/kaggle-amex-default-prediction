import os
from glob import glob

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

CATEGORICAL_VARIABLES = ['B_30', 'B_38', 'D_114', 'D_116',
                         'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
ID_VARIABLES = ['customer_ID']
DATE_VARIABLES = {'S_2': 'yyyy-MM-dd'}
TARGET_VARIABLE = 'target'
PREDICTION_VARIABLE = 'prediction'


def get_casted_variable(s: str):
    if s in CATEGORICAL_VARIABLES or s in ID_VARIABLES:
        return F.col(s).cast('string').alias(s)
    elif s in DATE_VARIABLES:
        return F.to_date(s, format=DATE_VARIABLES[s]).alias(s)
    # everything else we cast to float
    elif s in [TARGET_VARIABLE, PREDICTION_VARIABLE]:
        pass
    elif s.startswith('D_'):
        # D_* = Delinquency variables
        pass
    elif s.startswith('S_'):
        # S_* = Spend variables
        pass
    elif s.startswith('P_'):
        # P_* = Payment variables
        pass
    elif s.startswith('B_'):
        # B_* = Balance variables
        pass
    elif s.startswith('R_'):
        # R_* = Risk variables
        pass
    else:
        raise ValueError(f'Unexpected column: {s}')
    return F.col(s).cast('float').alias(s)


if __name__ == '__main__':

    path = 'data/amex-default-prediction'
    csvs = glob(os.path.join(path, '*.csv'))
    print(f'Will be processing these .csv files:\n{csvs}')

    spark = (
        SparkSession
        .builder
        .master('local[8]')
        .config('spark.driver.memory', '16g')
        .getOrCreate()
    )

    num_rows_per_partition = 1000000
    for csv in csvs:
        df = spark.read.options(header=True).csv(csv)
        num_rows = df.count()
        num_partitions = max(1, num_rows // num_rows_per_partition)
        print(
            f'{csv} has {num_rows} rows '
            f'and will be in {num_partitions} partitions '
            f'with {num_rows_per_partition} rows per partition.'
        )
        df = df.select(*[get_casted_variable(c) for c in df.columns])
        output_path = csv.replace('.csv', '')
        (
            df
            .repartition(num_partitions)
            .write
            .mode('overwrite')
            .parquet(output_path)
        )
        print(f'Wrote to {output_path}')
