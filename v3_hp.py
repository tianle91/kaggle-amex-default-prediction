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

train_data = spark.read.parquet(
    'data_transformed/amex-default-prediction/train_data_latest')
train_labels = spark.read.parquet('data/amex-default-prediction/train_labels')


non_feature_columns = [
    TARGET_VARIABLE,
    *ID_VARIABLES,
    *DATE_VARIABLES.keys(),
]
feature_columns = [
    c for c in train_data.columns
    if c not in non_feature_columns
]
categorical_feature_columns = CATEGORICAL_VARIABLES
numerical_feature_columns = [
    c for c in feature_columns if c not in categorical_feature_columns]
print(f'''
Feature columns ({len(feature_columns)}):
{', '.join(sorted(feature_columns))}

Categorical feature columns ({len(categorical_feature_columns)}):
{', '.join(sorted(categorical_feature_columns))}

Numerical feature columns ({len(numerical_feature_columns)}):
{', '.join(sorted(numerical_feature_columns))}
''')


encs = CategoricalToIntegerEncoders(
    columns=categorical_feature_columns).fit(train_data)
transformed_feature_columns = numerical_feature_columns + encs.columns_encoded


train_pdf = train_data.join(train_labels, on='customer_ID', how='inner')
train_pdf = encs.transform(spark=spark, df=train_pdf).toPandas()
train_pdf_bytes = train_pdf.memory_usage(deep=True).sum()
print(f'train_pdf.memory_usage in megabytes: {train_pdf_bytes / 1048576: .2f}')

X_fit = train_pdf[transformed_feature_columns].reset_index(drop=True)
y_fit = np.array(train_pdf[TARGET_VARIABLE])
print(
    f'X_fit.shape: {X_fit.shape} '
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


MAX_EVALS = 100

space = {
    'scale_pos_weight': hyperopt.hp.uniform('class_weight', 0., 10.),
    # lower learning rate, more iterations and more leaves
    'learning_rate': hyperopt.hp.uniform('learning_rate', 0., .1),
    'num_iterations': scope.int(hyperopt.hp.quniform('num_iterations', 100, 5000, 1)),
    'num_leaves': scope.int(hyperopt.hp.quniform('num_leaves', 31, 100, 1))
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
        'Main run info (no details here) '
        f'run_id: {run.info.run_id} '
        f'experiment_id: {run.info.experiment_id} '
    )

spark.stop()
