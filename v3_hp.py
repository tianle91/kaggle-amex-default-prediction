import hyperopt
import mlflow
import numpy as np
from hyperopt import Trials
from hyperopt.pyll.base import scope
from sklearn.model_selection import train_test_split

from encoder import CategoricalToIntegerEncoders
from format_data import (CATEGORICAL_VARIABLES, DATE_VARIABLES, ID_VARIABLES,
                         TARGET_VARIABLE)
from hp import build_train_objective, find_best_run
from spark_utils import get_spark_session

spark = get_spark_session()

# run transform_latest.py if this don't exist
test_data = spark.read.parquet(
    'data_transformed/amex-default-prediction/test_data_latest')
train_data = spark.read.parquet(
    'data_transformed/amex-default-prediction/train_data_latest')
# run format_data.py if these don't exist
train_labels = spark.read.parquet('data/amex-default-prediction/train_labels')
sample_submission = spark.read.parquet(
    'data/amex-default-prediction/sample_submission')


encs = CategoricalToIntegerEncoders(
    columns=CATEGORICAL_VARIABLES).fit(train_data)

train_pdf = train_data.join(train_labels, on='customer_ID', how='inner')
train_pdf = encs.transform(spark=spark, df=train_pdf).toPandas()

test_pdf = encs.transform(spark=spark, df=test_data).toPandas()


non_feature_columns = [
    TARGET_VARIABLE,
    *ID_VARIABLES,
    *DATE_VARIABLES.keys(),
]
feature_columns = [
    c for c in train_pdf.columns
    if c not in non_feature_columns
]
print(f'Feature columns ({len(feature_columns)}):\n' +
      ', '.join(feature_columns))


X_fit = train_pdf[feature_columns].reset_index(drop=True)
X_test = test_pdf[feature_columns].reset_index(drop=True)
y_fit = np.array(train_pdf[TARGET_VARIABLE])
print(
    f'X_fit.shape: {X_fit.shape} '
    f'X_test.shape: {X_test.shape} '
    f'y_fit.shape: {y_fit.shape} '
    f'y_fit uniques: {np.unique(y_fit, return_counts=True)} '
)

X_train, X_test, y_train, y_test = train_test_split(X_fit, y_fit)
print(
    f'X_train.shape: {X_train.shape} '
    f'X_test.shape: {X_test.shape} '
    f'y_train.shape: {y_train.shape} '
    f'y_test.shape: {y_test.shape} '
)


MAX_EVALS = 5

space = {
    'class_weight': {
        0.: 1.,
        1.: hyperopt.hp.uniform('class_weight', 0., 10.)
    },
    # 'subsample': hyperopt.hp.uniform('subsample', 0.05, 1.0),
    # The parameters below are cast to int using the scope.int() wrapper
    # 'num_iterations': scope.int(hyperopt.hp.quniform('num_iterations', 10, 200, 1)),
    # 'num_leaves': scope.int(hyperopt.hp.quniform('num_leaves', 20, 50, 1))
}


with mlflow.start_run(nested=False) as run:
    train_objective = build_train_objective(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        categorical_feature=encs.columns_encoded,
    )
    hyperopt.fmin(
        fn=train_objective,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=MAX_EVALS,
        trials=Trials()
    )
    find_best_run(run)
    print(
        f'run_id: {run.info.run_id} '
        f'experiment_id: {run.info.experiment_id} '
    )

spark.stop()
