from google import genai
import os
from dotenv import load_dotenv
load_dotenv()   
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
client = genai.Client(api_key=api_key)