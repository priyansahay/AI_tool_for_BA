from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.custom_exception import log_exception

def prepare_anomoly_data(sales_df):
    df = sales_df.copy()
    features = ["Daily_Revenue", "Daily_Orders", "Daily_Customers"]
    anomoly_df = df[features]
    return anomoly_df

def scale_features(anomoly_df):
    scaler = StandardScaler()
    scaled_features = (scaler.fit_transform(anomoly_df))
    return scaled_features, scaler

def train_anomoly_models(scaled_features, contamination = 0.05):
    model = IsolationForest(contamination = contamination, random_state = 42)
    model.fit(scaled_features)
    return model

def detect_anomolies(sales_df, model, scaled_features):
    result_df = sales_df.copy()
    result_df["Anomaly"] = model.predict(scaled_features)
    result_df["Anomaly"] = result_df["Anomaly"].map({1:"Normal", -1:"Anomaly"})
    anomaly_count = result_df[result_df["Anomaly"]=="Anomaly"].shape[0]  
    return result_df, anomaly_count

def anomaly_summary(result_df):
    summary = (
        result_df["Anomaly"].value_counts().reset_index()
    )
    summary.columns = ["Category","Count"]
    return summary

def plot_anomalies(result_df):
    fig = px.scatter(result_df, x = "InvoiceDate", y = "Daily_Revenue", color = "Anomaly", title = "Revenue Anomaly Detection", color_discrete_map={"Normal":"blue", "Anomaly":"red"})
    return fig

def get_top_anomalies(result_df):
    anomalies = (
        result_df[result_df["Anomaly"]=="Anomaly"].sort_values(by = "Daily_Revenue", ascending = False)
    )
    return anomalies

def save_anomaly_model(model, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,output_path)

def save_anomaly_results(result_df, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path,index=False)

def anomaly_detection_pipeline(sales_df):
    anomaly_df = (prepare_anomoly_data(sales_df))
    (scaled_features, scaler) = (scale_features(anomaly_df))
    model = (train_anomoly_models(scaled_features))
    result_df, anomaly_count = detect_anomolies(sales_df, model, scaled_features)
    summary = (anomaly_summary(result_df))
    top_anomalies = (get_top_anomalies(result_df))
    anomaly_fig = (plot_anomalies(result_df))
    return {
        "model": model,
        "scaler": scaler,
        "result_df": result_df,
        "summary": summary,
        "anomaly_count": anomaly_count,
        "top_anomalies": top_anomalies,
        "anomaly_fig": anomaly_fig
    }