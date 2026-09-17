#!/usr/bin/env python3
"""Train a metadata-only Pav Bhaji classifier.

The images are used solely to obtain the ground-truth labels from their
directory names.  Prediction features come only from pavbhaji.json.

Usage:
    python3 main.py
    python3 main.py --dataset dataset --output outputs
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def caption_from(post):
    """Return the first caption, handling missing Instagram fields safely."""
    edges = post.get("edge_media_to_caption", {}).get("edges", [])
    if edges:
        return edges[0].get("node", {}).get("text", "") or ""
    return ""


def normalise_text(text):
    """Preserve food words/hashtags while making spelling variants comparable."""
    text = str(text).lower()
    text = re.sub(r"pav\s*[-_]?\s*bhaji", "pavbhaji", text)
    text = re.sub(r"#", " hashtag_", text)
    text = re.sub(r"@", " mention_", text)
    return re.sub(r"\s+", " ", text).strip()


def build_dataframe(dataset_dir):
    """Join JSON records to image-directory labels through display URL filename."""
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "pavbhaji.json").open(encoding="utf-8") as handle:
        posts = json.load(handle)

    label_by_filename = {}
    for label in ("0", "1"):
        for image in (dataset_dir / "images" / label).glob("*"):
            if image.is_file():
                label_by_filename[image.name] = int(label)

    rows = []
    for post in posts:
        filename = Path(urlparse(post.get("display_url", "")).path).name
        if filename not in label_by_filename:
            continue  # Unlabelled JSON posts are not suitable for supervised fit.
        caption = caption_from(post)
        tags = post.get("tags") or []
        dimensions = post.get("dimensions") or {}
        likes = (post.get("edge_liked_by") or {}).get("count", 0)
        comments = (post.get("edge_media_to_comment") or {}).get("count", 0)
        height, width = dimensions.get("height", 0), dimensions.get("width", 0)

        # Prefixing fields retains their origin while allowing one text model.
        text = "caption " + normalise_text(caption)
        text += " tags " + " ".join("tag_" + normalise_text(tag) for tag in tags)
        rows.append({
            "post_id": post.get("id", ""), "image_filename": filename,
            "label": label_by_filename[filename], "text": text,
            "caption_length": len(caption), "tag_count": len(tags),
            "has_pavbhaji_variant": int(bool(re.search(r"pav\s*[-_]?\s*bhaji", caption + " " + " ".join(tags), re.I))),
            "likes_log": np.log1p(float(likes or 0)),
            "comments_log": np.log1p(float(comments or 0)),
            "is_video": int(bool(post.get("is_video", False))),
            "aspect_ratio": float(width) / height if height else 0.0,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No JSON posts could be matched to image filenames.")
    return frame, len(posts)


def fit_and_evaluate(frame, output_dir):
    """Use a held-out stratified split and save reproducible predictions."""
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, random_state=RANDOM_STATE, stratify=frame["label"]
    )
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]

    # Word terms capture ingredients/dish names; char terms tolerate spelling and hashtags.
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                           sublinear_tf=True, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                           max_features=15000, sublinear_tf=True)
    numeric_columns = ["caption_length", "tag_count", "has_pavbhaji_variant",
                       "likes_log", "comments_log", "is_video", "aspect_ratio"]
    scaler = StandardScaler()
    x_train = hstack([
        word.fit_transform(train.text), char.fit_transform(train.text),
        scaler.fit_transform(train[numeric_columns]),
    ]).tocsr()
    x_test = hstack([
        word.transform(test.text), char.transform(test.text),
        scaler.transform(test[numeric_columns]),
    ]).tocsr()

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                               random_state=RANDOM_STATE)
    model.fit(x_train, train.label)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test.label, prediction, average="binary", zero_division=0
    )
    metrics = {
        "split": "stratified 75% train / 25% held-out test",
        "random_state": RANDOM_STATE,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "accuracy": round(float(accuracy_score(test.label, prediction)), 4),
        "pavbhaji_precision": round(float(precision), 4),
        "pavbhaji_recall": round(float(recall), 4),
        "pavbhaji_f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(test.label, probability)), 4),
        "confusion_matrix_rows_actual_0_1": confusion_matrix(test.label, prediction).tolist(),
        "classification_report": classification_report(test.label, prediction,
                                                         target_names=["not_pavbhaji", "pavbhaji"],
                                                         zero_division=0),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    predictions = test[["post_id", "image_filename", "label"]].copy()
    predictions["pavbhaji_probability"] = probability
    predictions["prediction"] = prediction
    predictions.to_csv(output_dir / "heldout_predictions.csv", index=False)

    # Coefficients make the text model auditable. Numeric features follow both vocabularies.
    names = np.r_[word.get_feature_names_out(), char.get_feature_names_out(), numeric_columns]
    coefficients = pd.DataFrame({"feature": names, "coefficient": model.coef_[0]})
    pd.concat([
        coefficients.nlargest(25, "coefficient").assign(direction="supports_pavbhaji"),
        coefficients.nsmallest(25, "coefficient").assign(direction="supports_not_pavbhaji"),
    ]).to_csv(output_dir / "important_features.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset", help="directory containing pavbhaji.json and images/")
    parser.add_argument("--output", default="outputs", help="directory for metrics and predictions")
    args = parser.parse_args()
    frame, total_posts = build_dataframe(args.dataset)
    metrics = fit_and_evaluate(frame, Path(args.output))
    print(f"Matched labelled posts: {len(frame)} / {total_posts}")
    print("Class counts:", dict(Counter(frame.label)))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote results to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
