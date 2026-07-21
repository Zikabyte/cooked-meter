"""Priority target definition.

Encodes two rule-based decisions from the EDA notebook (context/DA_TimCooked.ipynb):
1. Which Dataset 3 linguistic categories make a sentiment prediction unreliable
   (Section 10's conclusion, plus judgment calls confirmed with the team for
   categories the EDA didn't explicitly analyze: comparative -> unreliable;
   ambiguous/contextual/emotional_exaggeration -> reliable by default).
2. The Priority_Score formula proposed in the EDA's Final Insight section:
   Priority_Score = emotion_weight(Emotion) x 1[Sentiment == Negative].
"""

EMOTION_WEIGHTS = {
    "Anger": 3,
    "Fear": 3,
    "Sadness": 2,
    "Happy": 0,
    "Love": 0,
}

UNRELIABLE_CATEGORIES = {
    "aspect_based",
    "negation",
    "mixed_sentiment",
    "question_conditional",
    "sarcasm",
    "comparative",
}


def category_to_reliability(category):
    """Map a Dataset 3 linguistic category to 'reliable' or 'unreliable'.

    Unlabeled reviews (category = 'tidak_berkategori') default to reliable,
    since they're the "plain" pool with no known linguistic risk.
    """
    return "unreliable" if category in UNRELIABLE_CATEGORIES else "reliable"


def priority_score(sentiment, emotion):
    """EDA's proposed formula: emotion_weight x 1[Sentiment == Negative]."""
    weight = EMOTION_WEIGHTS.get(emotion, 0)
    return weight if sentiment == "Negative" else 0


def priority_label(sentiment, emotion, reliability_flag):
    """Bucket a priority score + reliability flag into High/Medium/Low.

    Starter thresholds (not empirically validated against real CS data,
    same caveat as the EDA's own formula):
    - score 3 & reliable   -> High
    - score 3 & unreliable, or score 2 & reliable -> Medium
    - everything else      -> Low
    """
    score = priority_score(sentiment, emotion)
    reliable = reliability_flag == "reliable"

    if score == 3 and reliable:
        return "High"
    if score == 3 or (score == 2 and reliable):
        return "Medium"
    return "Low"
