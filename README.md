# kaggle-amex-default-prediction
https://www.kaggle.com/competitions/amex-default-prediction/


# Useful
`mlflow ui` runs the mlflow ui at [127.0.0.1:5000](http://127.0.0.1:5000)


# Submissions
These are taken from [kaggle submissions](https://www.kaggle.com/competitions/amex-default-prediction/submissions).
| index | notebook                                     | score | train score | valid score | mlflow run |
| ----- | -------------------------------------------- | ----- | ----------- | ----------- | ---------- |
| 3     | [use_aggregated.ipynb](use_aggregated.ipynb) | 0.778 | 0.79        | 0.775       | [link](http://127.0.0.1:5000/#/experiments/2/runs/1e0a4409d0f64b01a242d38c75df61cd) |
| 2     | [use_latest.ipynb](use_latest.ipynb)         | 0.776 | 0.786       | 0.774       | [link](http://127.0.0.1:5000/#/experiments/1/runs/65418e5e512a433fa7e669bbbeb18880) |
| 1     | sample_submission.csv                        | 0.019 | -           | -           | - |

## Learnings from 3
There were roughly 3 times the original number of features from `transform_aggregated.py` due to
aggregation but did not yield better results compared to only using the latest values.
Similar trends can be observed from the feature importance in [1e0a4409d0f64b01a242d38c75df61cd](http://127.0.0.1:5000/#/experiments/2/runs/1e0a4409d0f64b01a242d38c75df61cd),
where the last observed versions of the same variables dominate, with a few exceptions.


# Data
To download and unzip all files to `data`:
```
kaggle competitions download -c amex-default-prediction -p data
unzip data/amex-default-prediction.zip -d data/amex-default-prediction
python format_data.py
```
It should end up looking like this:
```
data
|-- amex-default-prediction.zip
|-- amex-default-prediction
|   |-- sample_submission.csv
|   |-- sample_submission
|   |   |-- *.parquet
|   |-- test_data.csv
|   |-- test_data
|   |   |-- *.parquet
|   |-- train_data.csv
|   |-- train_data
|   |   |-- *.parquet
|   |-- train_labels.csv
|   |-- train_labels
|   |   |-- *.parquet
```


# Questions
Note that this is the description of how the targets are calculated [src](https://www.kaggle.com/competitions/amex-default-prediction/data):
> The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest
statement date it is considered a default event.
> The dataset contains aggregated profile features for each customer at each statement date. 

## How much history do test ids have?
About the same as train ids. Test ids have no overlap with train ids.
Both train and test periods are about 13 months and the test dates follow immediately after train dates.
However, **the dates in test not uniformly distributed over the 13 months** (as is the case for train dates) and instead oversamples the October to May period.
See [how_much_history.ipynb](notebooks/how_much_history.ipynb).

## Label sequence for each id?
Time between statements has long tails.
The bottom 10% are less than 17 days while top 5% are 392 days.
Number of statements are 13 other than the bottom 10%, which only have 9 statements.
See [label_sequence.ipynb](notebooks/label_sequence.ipynb).

## Aggregate predictions for each statement?

## Predict time to default instead of binary outcome?

## Add date related features?
TODO:
- fourier terms.
- holiday terms.
- sentiments of news articles between 30 days prior to first statement date and last statement date.
