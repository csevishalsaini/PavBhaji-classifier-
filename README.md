# Pav Bhaji Metadata Classifier

This project predicts whether an Instagram post's attached image contains Pav Bhaji (`1`) or not (`0`) using **only the post metadata**. It is a text-classification project: no image pixels are read and no CNN is used.

## What the project does

`main.py` links each labelled image to its Instagram JSON record through the filename in `display_url`. It then learns from the caption, hashtags/tags, engagement counts, post format, and image dimensions.

The model uses a class-balanced logistic-regression classifier with TF-IDF text features. It creates a reproducible 75% training / 25% held-out test split, evaluates the model, and saves predictions.

## Requirements

- Python 3.9 or later
- The supplied `dataset/` directory with `pavbhaji.json` and `images/0`, `images/1`

## How to run

Open Terminal in this directory and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
mkdir -p .tmp
TMPDIR="$PWD/.tmp" python3 -m pip install -r requirements.txt
python3 main.py --dataset dataset --output outputs
```

`TMPDIR` is helpful on macOS when `pip` cannot write to its default temporary directory. If pip installs normally, it can be omitted.

## Output files

After running the script, the `outputs/` directory contains:

```text
outputs/
├── metrics.json
├── heldout_predictions.csv
└── important_features.csv
```

- `metrics.json`: held-out accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
- `heldout_predictions.csv`: each test image filename, its true label, predicted label, and Pav Bhaji probability.
- `important_features.csv`: fitted coefficients that show which text and metadata features most influenced a prediction.

## Features used

- Caption text and hashtags: word TF-IDF 1–2 grams
- Character TF-IDF 3–5 grams for spelling/hashtag variants
- Caption length and tag count
- Log-transformed likes and comments
- Video flag and aspect ratio

Image pixels, image URLs, post IDs, timestamps, and locations are excluded from model features.

## Results

The supplied dataset provides 452 labelled posts: 183 Pav Bhaji and 269 non-Pav Bhaji. On the held-out test set, the model achieved 0.6372 accuracy, 0.6435 Pav Bhaji F1, and 0.6463 ROC-AUC.

For the complete data analysis, feature rationale, confusion matrix, limitations, and recommendations, see `Vishal_Saini_Data_Analysis_Report.pdf`.
