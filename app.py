import gradio as gr
import pandas as pd
from googleapiclient.discovery import build
import re
from transformers import pipeline

# ⛔ Replace with your actual API key
YOUTUBE_API_KEY = "AIzaSyDIGiBwsTD2cA4NWAvJjAyBE39QZr6BnOY"

# Setup: Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)",
        r"youtube\.com\/shorts\/([^&\n?#]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Function to fetch comments using YouTube API
def fetch_comments(video_url, max_results=10):
    video_id = extract_video_id(video_url)
    if not video_id:
        return pd.DataFrame({"error": ["Invalid YouTube URL"]})
    
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    comments = []
    try:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        return pd.DataFrame({"Comment": comments})
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

# Main analysis function
def analyze_video(video_url, max_comments=10):
    df = fetch_comments(video_url, max_comments)
    if "error" in df.columns:
        return df.to_string(index=False)
    
    results = []
    for comment in df["Comment"]:
        result = sentiment_pipeline(comment[:512])[0]
        results.append({
            "Comment": comment,
            "Sentiment": result["label"],
            "Score": round(result["score"], 3)
        })
    result_df = pd.DataFrame(results)
    return result_df

# Gradio UI
with gr.Blocks(title="YouTube Comment Sentiment Analyzer") as demo:
    gr.Markdown("# 🚀 YouTube Comment Sentiment Analyzer")

    video_url = gr.Textbox(label="📸 YouTube Video URL", placeholder="Paste the video link here")
    max_comments = gr.Slider(1, 100, value=10, step=1, label="Number of Comments")

    btn = gr.Button("Analyze")

    output = gr.Dataframe(label="🎊Sentiment Analysis Result🎊", interactive=False)

    btn.click(fn=analyze_video, inputs=[video_url, max_comments], outputs=output)

demo.launch()
