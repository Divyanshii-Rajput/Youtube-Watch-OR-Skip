# backend/generate_dataset.py
import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
np.random.seed(42)

def generate_youtube_data(n=2000):
    log_views = np.random.uniform(6, 18, n)
    likes = np.random.randint(50, 50000, n)
    comment_count = np.random.randint(5, 5000, n)
    like_ratio = np.clip(np.random.normal(0.05, 0.03, n), 0.001, 0.15)
    sentiment = np.random.uniform(-1, 1, n)

    # Probabilistic label generation (balanced + noise)
    prob_watch = (
        0.35 * (likes / (likes.max() + 1)) +
        0.25 * np.clip((sentiment + 1) / 2, 0, 1) +
        0.2 * np.log1p(comment_count) / np.log1p(comment_count).max() +
        0.1 * (like_ratio / like_ratio.max()) +
        np.random.normal(0, 0.08, n)
    )

    label = (prob_watch > 0.45).astype(int)
    df = pd.DataFrame({
        "log_views": log_views,
        "likes": likes,
        "comment_count": comment_count,
        "like_ratio": like_ratio,
        "sentiment": sentiment,
        "label": label
    })
    return df

df = generate_youtube_data(2000)
df.to_csv("data/youtube_dataset.csv", index=False)
print("✅ Dataset saved:", df.shape)
print(df['label'].value_counts(normalize=True))
