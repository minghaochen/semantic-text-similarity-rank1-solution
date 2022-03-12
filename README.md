# Data

## Files

- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

## Columns

- `id`: The unique ID for each data point.
- `s1`: First sentence.
- `s2`: Second sentence.
- `lang`: Long name of the language (ie French, English, etc.)
- `lang_code`: A two-letter language code (fr, en, etc.)
- `sim1`: similarity score 1.
- `sim2`: similarity score 2.
- `sim3`: similarity score 3.
- `category`: the inference category. [target variable]



# Usage

Just training xlm-roberta-large will get you a place in top-5! Combine with xlm-roberta-base and you are golden.

```
python training.py --fold 0 --model xlm-roberta-large --lr 3e-5 --epochs 6 --max_len 128 --batch_size 32 --valid_batch_size 16 --output xlm-roberta-large-xnli
```

```
python inference.py --model xlm-roberta-large --max_len 128 --batch_size 64 --model_path model_0.bin --output_name xlmroblarge
```

