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



