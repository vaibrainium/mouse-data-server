import openai
import logging
from time import sleep
from requests.exceptions import RequestException
import os
from dotenv import load_dotenv
import pandas as pd

# from openai import OpenAI
# client = OpenAI()

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Retrieve the API key from .env
if openai.api_key is None:
    raise ValueError("API key is missing. Please check your .env file.")

def summarize_comments_with_chatgpt(comments):
    prompt = f"Summarize the following user session feedback:\n\n{comments}"

    retries = 3  # Set the number of retries
    for attempt in range(retries):
        try:
            # Send request to OpenAI's chat completion API (new API)
            response = openai.chat.completions.create(  # Updated method
                model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" depending on the model you prefer
                messages=[{
                    "role": "system", "content": "You are a helpful assistant."
                }, {
                    "role": "user", "content": prompt
                }],
                max_tokens=200,  # Adjust max_tokens as needed
                temperature=0.7,  # Adjust temperature for creativity vs accuracy
                n=1,  # Number of responses to return
            )

            # Extract the response
            return response['choices'][0]['message']['content'].strip()  # Accessing 'message' content

        except RequestException as e:
            logging.error(f"Attempt {attempt + 1}: Error connecting to OpenAI API: {e}")
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
    return summarize_comments_with_chatgpt(comments)








if __name__ == "__main__":
	# Example DataFrame with user session feedback (comments)
	data = {
		"comments": [
			"The service was good, but it could be faster.",
			"Great experience! Will definitely come back.",
			"Had some issues with the app. Not very intuitive."
		]
	}

	# Create DataFrame
	filtered_df = pd.DataFrame(data)

	# Test summarization
	summary = summarize_session(filtered_df)
	print("Summary:", summary)
