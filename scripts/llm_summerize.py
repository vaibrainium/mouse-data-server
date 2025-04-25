
import requests
import logging
from time import sleep
from requests.exceptions import RequestException

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def summarize_comments_with_ollama(comments):
    prompt = f"Summarize the following user session feedback:\n\n{comments}"
    ollama_url = "http://10.155.206.212:11434/api/generate"

    retries = 3  # Set the number of retries
    for attempt in range(retries):
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=5  # Timeout to avoid hanging
            )
            response.raise_for_status()  # Raise error for non-2xx status codes
            return response.json().get("response", "No summary available.")

        except RequestException as e:
            logging.error(f"Attempt {attempt + 1}: Error connecting to Ollama server: {e}")
            if attempt < retries - 1:
                logging.info("Retrying...")
                sleep(2)  # Wait before retrying
            else:
                logging.error("Summarization service failed after retries.")
                return "Summarization service is unavailable."

def summarize_session(filtered_df):
    comments = "\n".join(filtered_df["comments"].dropna().tolist())
    if not comments.strip():
        return "No comments to summarize."
    return summarize_comments_with_ollama(comments)

