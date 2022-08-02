from math import ceil
from typing import List

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from lightgbm import LGBMClassifier, early_stopping
from pyspark import StorageLevel
from pyspark.sql import DataFrame
from sklearn.model_selection import train_test_split

from getxy import GetXY

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
                # persist to disk only because we're trying to avoid OOM error with batching
                .persist(storageLevel=StorageLevel.DISK_ONLY)
            )
            for i in range(self.num_batches)
        ]

    def fit_transform(self, df: DataFrame):
        return self.fit(df=df).transform(df=df)


class BatchedLGBMClassifier:
    def __init__(self, lgb_params: dict, getxy: GetXY):
        self.lgb_params = lgb_params
        self.getxy = getxy
        self.model = None

    def fit(self, dfs: List[DataFrame]):
        for i, df in enumerate(dfs):
            print(f'Fitting {i}/{len(dfs)} with {df.count()} rows')
            X, y = self.getxy.transform(df=df)
            X_fit, X_valid, y_fit, y_valid = train_test_split(X, y, test_size=.1)
            fit_params = {
                'X': X_fit,
                'y': y_fit,
                'eval_set': [(X_valid, y_valid)],
                'eval_names': ['valid10pct'],
                'categorical_feature': self.getxy.columns_encoded,
                'callbacks': [early_stopping(stopping_rounds=10, first_metric_only=False)]
            }
            if self.model is not None:
                fit_params.update({'init_model': self.model})
            self.model = LGBMClassifier(**self.lgb_params).fit(**fit_params)
        return self

    def predict(self, dfs: List[DataFrame], id_variables: List[str], prediction_variable: str):
        pred_outputs = []
        for i, df in enumerate(dfs):
            print(f'Predicting {i}/{len(dfs)} with {df.count()} rows')
            X, y = self.getxy.transform(df=df)
            pred_outputs.append((
                df.select(id_variables).toPandas(),
                self.model.predict_proba(X=X)
            ))
        ids, preds = zip(*pred_outputs)
        pred_df = pd.concat(ids, axis=0)
        pred_df[prediction_variable] = np.concatenate(preds, axis=0)[:, 1]
        return pred_df



__IS_TEST_COLUMN__ = '__IS_TEST_COLUMN__'


def train_test_split_df(df, test_size: float = .25, seed=RANDOM_SEED):
    df = df.withColumn(
        __IS_TEST_COLUMN__,
        F.rand(seed=seed) < F.lit(test_size)
    )
    df_train = df.filter(~F.col(__IS_TEST_COLUMN__)).drop(__IS_TEST_COLUMN__)
    df_test = df.filter(F.col(__IS_TEST_COLUMN__)).drop(__IS_TEST_COLUMN__)
    # need to persist for consistent behaviour with rand
    # persist to disk only because we're trying to avoid OOM error with batching
    df_train = df_train.persist(storageLevel=StorageLevel.DISK_ONLY)
    df_test = df_test.persist(storageLevel=StorageLevel.DISK_ONLY)
    return df_train, df_test
