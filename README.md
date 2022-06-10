# kaggle-amex-default-prediction
https://www.kaggle.com/competitions/amex-default-prediction/


# data
To download and unzip all files to `data`:
```
kaggle competitions download -c amex-default-prediction -p data
unzip data/amex-default-prediction.zip -d data
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


# questions
Note that this is the description of how the targets are calculated [src](https://www.kaggle.com/competitions/amex-default-prediction/data):
> The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest
statement date it is considered a default event.
> The dataset contains aggregated profile features for each customer at each statement date. 

## how much history do test ids have?
About the same as train ids. Test ids have no overlap with train ids.
See [how_much_history.ipynb](notebooks/how_much_history.ipynb).

## label sequence for each id?
Time between statements has long tails.
The bottom 10% are less than 17 days while top 5% are 392 days.
Number of statements are 13 other than the bottom 10%, which only have 9 statements.
See [label_sequence.ipynb](notebooks/label_sequence.ipynb).

## performance using last statement or first statement?

## need to add aggregations across previous statements?

## aggregate predictions for each statement?

## predict time to default instead of binary outcome?
