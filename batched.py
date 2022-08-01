from math import ceil
from typing import List

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

__BATCH_INDEX_COLUMN__ = '__BATCH_INDEX_COLUMN__'
RANDOM_SEED = 20220801


class Batched:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def fit(self, df: DataFrame):
        self.num_batches = ceil(df.count() / self.batch_size)
        return self

    def transform(self, df: DataFrame):
        df = df.withColumn(
            __BATCH_INDEX_COLUMN__,
            (F.rand(seed=RANDOM_SEED) * self.num_batches).cast('int')
        )
        return [
            (
                df
                .filter(F.col(__BATCH_INDEX_COLUMN__) == i)
                .drop(__BATCH_INDEX_COLUMN__)
                # need to persist for consistent behaviour with rand
                .persist()
            )
            for i in range(self.num_batches)
        ]

    def fit_transform(self, df: DataFrame):
        return self.fit(df=df).transform(df=df)
