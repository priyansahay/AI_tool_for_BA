from src.data_cleaning import cleaning_summary, clean_data, save_clean_data
from src.data_ingestion import load_data, print_dataset_info
from src.custom_exception import log_exception
import logging
import pandas as pd

raw_data_path = r"data/raw/Online Retail.xlsx"
clean_data_path = r"data/processed/Clean File.csv"

def main():
    try:
        # DATA INGESTION
        logging.info("Pipeline Started\n")
        df = load_data(raw_data_path)
        print_dataset_info(df)

        # CLEANING DATA
        logging.info("Data Cleaning\n")
        cleaned_df = clean_data(df)
        save_clean_data(cleaned_df,clean_data_path)
        original_df = pd.read_excel(raw_data_path)
        cleaning_summary(original_df,cleaned_df)

        logging.info("Cleaned Data set successfilly")
        print(f"Final Shape: {cleaned_df.shape}")

    except Exception as e:
        log_exception(e)
        raise

if __name__ == "__main__":
    main()