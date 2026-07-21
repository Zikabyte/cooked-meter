from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def split_for_leakage_avoidance(df, stratify_col="Sentiment", chunk1_size=0.5):
    """Split Dataset 2 into two disjoint chunks so Model A never sees the
    same rows that later get used to build Model C's training table.

    chunk1 -> used to fit + evaluate Model A (the sentiment/emotion classifier)
    chunk2 -> held out from Model A entirely; Model A's predictions on this
              chunk become the honest, out-of-sample predicted_sentiment /
              predicted_emotion features for Model C
    """
    chunk1, chunk2 = train_test_split(df, train_size=chunk1_size, stratify=df[stratify_col], random_state=RANDOM_STATE)
    return (chunk1, chunk2)


def train_test_split_simple(df, target_col, test_size=0.2):
    """Generic reusable train/test split, e.g. to sanity-check Model A's
    own accuracy inside chunk1, or to split chunk2 into train/test for
    Model C later.

    Returns (train_df, test_df).
    """
    res = train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=RANDOM_STATE)
    return res
