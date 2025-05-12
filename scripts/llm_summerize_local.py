import json
import logging
import requests
import pandas as pd
from textblob import TextBlob
from datetime import date
import os
from dotenv import load_dotenv

# Custom function to handle date serialization
def custom_json_serializer(obj):
    if isinstance(obj, date):  # Check if the object is a datetime.date
        return obj.isoformat()  # Return the ISO formatted string
    raise TypeError(f"Type {type(obj)} not serializable")

# LLM for detailed session analysis
def chat_with_model(WEBUI_TOKEN, summary_df, sessionwise_data):
    url = 'http://10.155.206.212:16000/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {WEBUI_TOKEN}',
        'Content-Type': 'application/json'
    }

    # 1. Extract structure and sample data from summary_df
    summary_columns = summary_df.columns.tolist()
    summary_sample = summary_df.head(3).to_dict(orient='records')
    comments_sample = "\n".join(summary_df["comments"].dropna().tolist()[:30])

    # 2. Extract metadata and samples from sessionwise_data (session-level)
    session_keys = [str(idx) for idx in range(len(sessionwise_data))]  # Use list indices for simplicity

    session_preview = {}
    for k in session_keys[:3]:
        # Access each session dictionary correctly by index
        session_preview[k] = {
            sk: str(sv)[:80] if not isinstance(sv, date) else sv.isoformat() for sk, sv in sessionwise_data[int(k)].items() if isinstance(sv, (str, int, float, date))
        }

    # 3. Analyze feedback trends and session-level patterns
    feedback_trends = analyze_feedback_trends(summary_df)
    session_patterns = analyze_session_patterns(sessionwise_data)

    # 4. Formulate a prompt for LLM based on the data
    prompt = f"""
You are analyzing animal training performance feedback and session data. You are an expert neuroscientist studying the behavior of mice performing a random-dot motion task.

--- Summary Feedback Data ---
Columns: {summary_columns}
Sample Rows (first 3 entries): {json.dumps(summary_sample, indent=2, default=custom_json_serializer)}

--- Sample Comments ---
{comments_sample}

--- Sessionwise Data Overview ---
Session IDs (sample): {session_keys[:5]}
Sample Session Contents:
{json.dumps(session_preview, indent=2, default=custom_json_serializer)}

--- Analysis Instructions ---
1. Summarize the feedback trends, including any performance issues or concerns raised.
2. Identify session-level behavioral patterns that could impact training or performance.
3. Discuss the mouse's overall performance based on metrics such as accuracy, sensory noise, reaction times, and biases.
4. Based on the analysis, provide recommendations for improving training quality and efficiency, addressing any issues found, and suggesting targeted improvements.
    """.strip()

    # 5. Send request to the LLM model
    data = {
        "model": "llama2:latest",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"❌ Request failed: {e}")
        return None


# Function to analyze feedback trends from the summary_df
def analyze_feedback_trends(summary_df):
    # Sentiment analysis of comments
    sentiment_counts = summary_df['comments'].apply(analyze_sentiment).value_counts().to_dict()
    avg_accuracy = summary_df["session_accuracy"].mean()
    avg_reward = summary_df["total_reward"].mean()

    feedback_trends = f"""
    Sentiment Distribution: {sentiment_counts}
    Average Session Accuracy: {avg_accuracy:.2f}%
    Average Total Reward: {avg_reward:.2f}
    """
    return feedback_trends


# Function to analyze session-level performance from sessionwise_data
def analyze_session_patterns(sessionwise_data):
    session_patterns = []
    for session in sessionwise_data:
        session_id = session.get('session_uuid', 'Unknown')
        session_accuracy = session.get('session_accuracy', None)
        sensory_noise = session.get('sensory_noise', None)
        reaction_time_mean = session.get('reaction_time_mean', None)

        session_pattern = f"""
        Session ID: {session_id}
        Accuracy: {session_accuracy if session_accuracy is not None else 'N/A'}%
        Sensory Noise: {sensory_noise if sensory_noise is not None else 'N/A'}
        Reaction Time Mean: {reaction_time_mean if reaction_time_mean is not None else 'N/A'} seconds
        """

        session_patterns.append(session_pattern)

    return "\n".join(session_patterns)


# Helper function to analyze sentiment of feedback comments
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'


# Perform the overall comprehensive analysis combining feedback and session data
def comprehensive_analysis(summary_df, sessionwise_data, WEBUI_TOKEN):
    structured_analysis = analyze_feedback_trends(summary_df)
    session_patterns = analyze_session_patterns(sessionwise_data)

    model_response = chat_with_model(WEBUI_TOKEN, summary_df, sessionwise_data)
    if model_response is None:
        model_summary = "⚠️ Model failed to return a summary."
    else:
        model_summary = model_response["choices"][0]["message"]['content']

    full_report = (
        f"--- Feedback Trends ---\n{structured_analysis}\n"
        # f"--- Session-Level Patterns ---\n{session_patterns}\n"
        f"--- Summary ---\n{model_summary}"
    )

    return full_report


# Summarize feedback session using both summary and session-level data
def summarize_session(summary_df, sessionwise_data):
    if summary_df["comments"].dropna().empty:
        return "No comments available for summarization."

    response = comprehensive_analysis(summary_df, sessionwise_data, WEBUI_TOKEN)
    if response is None:
        return "Error: Unable to connect to the model."
    else:
        return response#["choices"][0]["message"]['content']


# Load environment variables from .env file
load_dotenv()

# Retrieve the WebUI token from the environment variable
WEBUI_TOKEN = os.getenv("WEBUI_TOKEN")
