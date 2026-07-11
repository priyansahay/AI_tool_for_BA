import streamlit as st
import pandas as pd
from pathlib import Path
from src.business_chatbot import ask_question
from src.gemini_client import client
import os
from dotenv import load_dotenv
load_dotenv() 

st.set_page_config(
    page_title="AI Business Intellegence Copilot",
    page_icon = "📊",
    layout="wide"
)
# --- PATHS --- 
CLEAN_DATA_PATH = os.getenv("clean_data_path")
SEGMENTS_PATH = os.getenv("segmented_data_path")
ANOMALY_PATH = os.getenv("anomaly_results_path")
CHURN_PATH = os.getenv("churn_prediction_path")
REPORT_DIR = "reports"