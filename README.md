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

## how much history do test ids have?
About the same as train ids. Test ids have no overlap with train ids.
See [how_much_history.ipynb](notebooks/how_much_history.ipynb).
