import numpy as np
import re
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib, os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow frontend access (React running on port 3000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_clean.joblib")

model = joblib.load(MODEL_PATH)

# Get YouTube API key
API_KEY = os.getenv("YT_API_KEY")

@app.get("/")
def root():
    return {"message": "Backend is running ✅"}

@app.post("/predict/")
def predict(data: dict):
    print("📩 Received data:", data)
    try:
        url = data.get("url")
        print("🎥 Video URL:", url)

        # --- Extract YouTube video ID ---
        match = re.search(r"(?:v=|be/)([A-Za-z0-9_-]{11})", url)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        video_id = match.group(1)
        print("🆔 Extracted video_id:", video_id)

        # --- Fetch video statistics ---
        api_url = f"https://www.googleapis.com/youtube/v3/videos?part=statistics,snippet&id={video_id}&key={API_KEY}"
        response = requests.get(api_url)
        data = response.json()

        if "items" not in data or len(data["items"]) == 0:
            raise HTTPException(status_code=404, detail="Video not found")

        stats = data["items"][0]["statistics"]
        snippet = data["items"][0]["snippet"]

        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))

        # --- Derived features ---
        log_views = np.log1p(views)
        like_ratio = likes / (views + 1)

        analyzer = SentimentIntensityAnalyzer()
        text = snippet.get("title", "") + " " + snippet.get("description", "")
        sentiment = analyzer.polarity_scores(text)["compound"]

        print(f"📊 Features: log_views={log_views}, likes={likes}, comment_count={comments}, like_ratio={like_ratio}, sentiment={sentiment}")

        # --- Model expects 5 features ---
        features = [[log_views, likes, comments, like_ratio, sentiment]]
        prediction = model.predict(features)[0]
        print("🤖 Model prediction:", prediction)

        decision = "Watch 👍" if prediction == 1 else "Skip 👎"

        return {
            "decision": decision,
            "features": {
                "log_views": log_views,
                "likes": likes,
                "comments": comments,
                "like_ratio": like_ratio,
                "sentiment": sentiment
            }
        }

    except Exception as e:
        print("❌ ERROR in /predict/:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
