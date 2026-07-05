import pandas as pd
from pathlib import Path
from logging import Logger
from src.custom_exception import log_exception


try:
    def create_rfm_feature(df):
        # CREATING RECENCY, FREQUENCY AND MONETORY FEATURES
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        rfm = (
            df.groupby("CustomerID")
            .agg(
                Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
                Frequency=("InvoiceNo", "nunique"),
                Monetary=("Revenue", "sum"),
            )
            .reset_index()
        )
        return rfm

    def create_customer_features(df):
        # CREATING CUSTOMER FEATURES TOTAL REVENUE, ORDERS, ITEMS, PRODUCTS, VALUE
        customer_features = (
            df.groupby("CustomerID")
            .agg(
                Total_Revenue = ("Revenue","sum"),
                Average_Order_Value = ("Revenue", "mean"),
                Total_Orders = ("InvoiceNo", "nunique"),
                Total_Products = ("Description" , "nunique"),
                Total_Items = ("Quantity", "sum"),
                Active_days=("InvoiceDate", lambda x: (x.max()-x.min()).days) 
            ).reset_index()
        )
        return customer_features

    def create_time_features(df):
        df=df.copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Month_Name"] = (df["InvoiceDate"].dt.month_name())
        df["Week_Number"] = (df["InvoiceDate"].dt.isocalendar().week)
        df["is_Weekend"] = df["InvoiceDate"].dt.dayofweek.isin([5, 6]).astype(int)
        return df

    def sales_aggregation_features(df):
        sales_features = (
            df.groupby("InvoiceDate").agg(
                Daily_Revenue = ("Revenue","sum"),
                Daily_Orders = ("InvoiceNo","nunique"),
                Daily_Customers = ("CustomerID", "nunique"),
            ).reset_index()
        )
        return sales_features

    def create_churn_label(customer_features, threshold_days = 90):
        customer_features = (customer_features.copy())
        customer_features["Churn"] = (customer_features["Active_days"]<threshold_days).astype(int)
        return customer_features

    def save_features_dataset(df,output_path):
        Path(output_path).parent.mkdir(parents = True, exist_ok= True)
        df.to_csv(output_path, index = False)
        return df
    
    # MASTER PIPELINE

    def feature_engineering_pipeline(df):
        enriched_df = (create_time_features(df))
        rfm_df = (create_rfm_feature(df))
        customer_df = (create_customer_features(df))
        churn_df = (create_churn_label(customer_df))
        sales_df = (sales_aggregation_features(df))
        return {
            "transaction_df":enriched_df,
            "rfm_df":rfm_df,
            "customer_df": customer_df,
            "churn_df": churn_df,
            "sales_df": sales_df
        }
    
except Exception as e:
    log_exception(e)
    raise
