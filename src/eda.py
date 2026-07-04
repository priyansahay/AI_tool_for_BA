from src.custom_exception import log_exception
import pandas as pd

try:
    def dataset_overview(df):
        return {"rows": df.shape[0],
                "column": df.shape[1],
                "missing Values": df.isnull().sum()
                }
    
    def categorical_summary(df):
        categoricalcol = ["Country","Weekday"]
        summary = {}
        for i in categoricalcol:
            summary[i] = (
                df[i].value_counts().head(10)
            )
        return summary
    

    def numerical_summary(df):
        numericalcols = ["Quantity", "UnitPrice", "Revenue"]
        return df[numericalcols].describe()
    
    def revenue_analysis(df):
        return{
            "Total Revenue" : df["Revenue"].sum(),
            "Average Revenue" : df["Revenue"].mean(),
            "Maximum Revenue" : df["Revenue"].max()
        }

    def country_analysis(df):
        return (
            df.groupby("Country")["Revenue"].sum().sort_values(ascending = False)
            )
    
    def customer_analysis(df):
        return(
            df.groupby("CustomerID")["Revenue"].sum().sort_values(ascending = False).head(10)
        )
    
    def product_analysis(df):
        return(
            df.groupby("Description")["Revenue"].sum().sort_values(ascending = False).head(10)
        )
    def monthly_sales_analysis(df):
        return(
            df.groupby(["Year","Month"])["Revenue"].sum().sort_values(ascending = False).head(10)
        )
    
    def weekday_analysis(df):
        return (
            df.groupby("Weekday")["Revenue"].sum()
        )
    
    def coorelation_analysis(df):
        cols = ["Quantity", "UnitPrice","Revenue" ]
        return df[cols].corr()
    
    
except Exception as e:
    log_exception(e)
    raise
    