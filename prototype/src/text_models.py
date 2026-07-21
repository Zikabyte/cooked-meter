from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def build_text_classifier(max_features=5000):
    """Build a TF-IDF + classifier pipeline for Indonesian review text.

    Used twice with the same architecture but different target columns:
    once for Sentiment, once for Emotion (Model A), and again for the
    linguistic reliability flag (Model B).
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def fit_and_evaluate(pipeline, train_df, test_df, text_col, target_col):
    """Fit `pipeline` on train_df, then print a classification_report on
    test_df so you can sanity-check accuracy before trusting the model.

    Returns the fitted pipeline.
    """
    pipeline.fit(train_df[text_col], train_df[target_col])
    preds = pipeline.predict(test_df[text_col])
    print(classification_report(test_df[target_col], preds))
    return pipeline
