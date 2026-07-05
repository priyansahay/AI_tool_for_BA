import pandas as pd


def create_rfm_feature(df):
    # CREATING RECENCY, FREQUENCY AND MONETORY FEATURES
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
            Total_Orders = ("invoiceNo", "nunique"),
            Total_Products = ("Description" , "nunique"),
            Total_Items = ("Quantity", "sum"),
            Active_days=("InvoiceDate", lambda x: (x.max()-x.min()).days) 
        ).reset_index()
    )
    return customer_features