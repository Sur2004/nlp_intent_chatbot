from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "faq_intents.csv"
RESPONSES_PATH = ROOT / "data" / "faq_responses.json"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["intent"] = df["intent"].fillna("").astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df

def make_models():
    def features():
        return FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=1,
                sublinear_tf=True, strip_accents="unicode"
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                sublinear_tf=True, max_features=100000
            ))
        ])
    return {
        "Multinomial Naive Bayes": Pipeline([
            ("features", features()),
            ("classifier", MultinomialNB(alpha=0.2))
        ]),
        "Logistic Regression": Pipeline([
            ("features", features()),
            ("classifier", LogisticRegression(
                max_iter=3000, C=3.0, class_weight="balanced",
                solver="lbfgs", random_state=RANDOM_STATE
            ))
        ])
    }

def main():
    df = load_data()
    print("Dataset shape:", df.shape)
    print("\nIntent distribution:\n", df["intent"].value_counts().sort_index())
    print("\nMissing values:\n", df.isna().sum())
    print("Duplicate utterances after cleaning:", int(df["text"].duplicated().sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], test_size=0.25,
        random_state=RANDOM_STATE, stratify=df["intent"]
    )

    trained = {}
    comparison = []
    labels = sorted(df["intent"].unique())
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        trained[name] = model
        comparison.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "weighted_f1": f1_score(y_test, pred, average="weighted")
        })
        print("\n" + "=" * 70 + "\n" + name)
        print(classification_report(y_test, pred, labels=labels, zero_division=0))
        cm = confusion_matrix(y_test, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(13, 10))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=90, colorbar=False)
        plt.tight_layout()
        fig.savefig(REPORT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150)
        plt.close(fig)

    results = pd.DataFrame(comparison).sort_values("macro_f1", ascending=False)
    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:\n", results.to_string(index=False))
    best_name = str(results.iloc[0]["model"])
    best_model = trained[best_name]
    joblib.dump(best_model, MODEL_DIR / "smart_faq_pipeline.joblib")
    (MODEL_DIR / "faq_responses.json").write_text(RESPONSES_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    metadata = {
        "selected_model": best_name,
        "random_state": RANDOM_STATE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "labels": labels,
        "dataset_rows": len(df),
        "note": "Metrics are from a held-out split of a small custom dataset. Validate with unseen real user queries before deployment."
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Save the held-out error analysis for review.
    best_pred = best_model.predict(X_test)
    pd.DataFrame({"text": X_test.values, "actual_intent": y_test.values, "predicted_intent": best_pred}).query(
        "actual_intent != predicted_intent"
    ).to_csv(REPORT_DIR / "misclassified_examples.csv", index=False)
    print(f"\nSelected model: {best_name}")
    print("Artifacts saved in models/ and evaluation outputs in reports/")

if __name__ == "__main__":
    main()
