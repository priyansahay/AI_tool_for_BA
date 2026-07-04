import pandas as pd
import logging

logger = logging.getLogger(__name__)

def check_missing_values(df):
    return df.isnull().sum()

def remove_duplicates(df):
    return df.isnull().sum()

def remove_duplicates(df):
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(
        f"Removed {initial_rows - len(df)} duplicates rows."
    )
    return df

def convert_date_column(df):
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

def remove_missing_customer_id(df):
    initialrows = len(df)
    df=df.dropna(subset = ["CustomerID"]) 
    logger.info(f"{initialrows - len(df)} rows with missing Customer IDs")
    return df

def remove_missing_description(df):
    initialrows = len(df)
    df=df.dropna(subset = ["Description"]) 
    logger.info(f"Deleted {initialrows - len(df)} rows with missing Description")
    return df

def removed_cancel_orders(df):
    initial_rows = len(df)
    df=df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    logger.info(f"Deleted {initial_rows - len(df)} rows with CANCELLED Invoice")
    return df

def remove_negative_quant(df):
    initial_rows = len(df)
    df=df[df["Quantity"]>0]
    logger.info(f"Deleted {initial_rows - len(df)} rows with negative Quantity")
    return df

def removed_invalid_price(df):
    initialrows = len(df)
    df= df[df["UnitPrice"]>0]
    logger.info(f"Deleted {initialrows - len(df)} rows with Invalid price records.")
    return df

def standardize_ID(df):
    df["CustomerID"] = (df["CustomerID"].astype(int).astype(str))
    return df

def revenue_column(df):
    df["Revenue"]=(df["Quantity"]*df["Unitprice"])
    return df

def create_date(df):
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Quarter"] = df["InvoiceDate"].dt.quarter
    df["Day"] = df["InvoiceDate"].dt.day
    df["Weekday"] = df["InvoiceDate"].dt.day_name()
    return df

def save_clean_data(df, outputpath):
    df.to_csv(outputpath, index = False)
    logger.info(f"Cleaned dataset saved to {outputpath}")

def cleaning_summary(original_df, cleaned_df):
    return {
        "original_rows" : len(original_df),
        "cleaned_df" : len(cleaned_df),
        "rows_removed" : len(original_df) - len (cleaned_df)
    }

def clean_data(df):
    logger.info("Started Cleaning Data...\n")
    df = remove_duplicates(df)
    df = convert_date_column(df)
    df = remove_missing_customer_id(df)
    df = remove_missing_description(df)
    df = removed_cancel_orders(df)
    df = remove_negative_quant(df)
    df = removed_invalid_price(df)
    df = standardize_ID(df)
    df = revenue_column(df)
    df = create_date(df)
    logger.info("Data Cleaning completed...\n")


