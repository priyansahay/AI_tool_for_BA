import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
from src.custom_exception import log_exception
from sklearn.metrics import (mean_absolute_error, mean_squared_error,r2_score)
from xgboost import XGBRegressor

def prepare_forecasting_data(sales_df):
    df = sales_df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.sort_values(by = "InvoiceDate")
    df["Revenue_Lag1"] = (df["Daily_Revenue"].shift(1))
    df["Revenue_Lag7"] = (df["Daily_Revenue"].shift(7))
    df["Rolling_7_Day"] = (df["Daily_Revenue"].rolling(7).mean())
    df["DayOfWeek"] = (df["InvoiceDate"]).dt.dayofweek
    df["Month"] = (df["InvoiceDate"]).dt.month
    df=df.dropna()
    return df

def split_forecasting_data(df):
    features = ["Revenue_Lag1", "Revenue_Lag7","Rolling_7_Day","DayOfWeek","Month"]
    target = "Daily_Revenue"
    split_index = int(len(df)*(0.8))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    x_train = train_df[features]
    y_train = train_df[target]
    x_test = test_df[features]
    y_test = test_df[target]
    return (x_test,x_train,y_test,y_train)

def train_forecasting_model(x_train,y_train):
    model = XGBRegressor(n_estimators = 200, learning_rate = 0.05, max_depth = 5, random_state = 42)
    model.fit(x_train, y_train)
    return model

def evualate_model(model, x_test, y_test):
    prediction = (model.predict(x_test))
    metrics = {
        "MAE": round(mean_absolute_error(y_test,prediction),2),
        "RMSE" : round(mean_squared_error(y_test, prediction)**0.5,2),
        "R2": round(r2_score(y_test,prediction),4)
    }
    return prediction, metrics

def plot_forecast_result(y_test,prediction):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y = y_test, mode="lines",name="Actual Revenue"))
    fig.add_trace(go.Scatter(y = prediction, mode="lines",name="Predicted Revenue"))
    fig.update_layout(title = "Revenue Forecast")
    return fig

def forecast_next_day(prepared_df, model):
    latest_record = (prepared_df.iloc[-1])
    input_data = pd.DataFrame([{
        "Revenue_Lag1": latest_record["Daily_Revenue"],
        "Revenue_Lag7": latest_record["Revenue_Lag7"],
        "Rolling_7_Day": latest_record["Rolling_7_Day"],
        "DayOfWeek": latest_record["DayOfWeek"],
        "Month": latest_record["Month"]
    }])
    forecast = model.predict(input_data)[0]
    return round(forecast,2)

def save_forecasting_model(model,model_path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,model_path)

def forecasting_pipeline(sales_df):
    prepared_df = (prepare_forecasting_data(sales_df))
    (x_test,x_train,y_test,y_train) = split_forecasting_data(prepared_df)
    model = (train_forecasting_model(x_train,y_train))
    (prediction ,metrics) = evualate_model(model,x_test,y_test)
    forecast_value = (forecast_next_day(prepared_df,model))
    forecast_fig = (plot_forecast_result(y_test,prediction))
    return {"model":model, "metrics":metrics, "forecast":forecast_value, "forecast_fig":forecast_fig, "prepared_df": prepared_df}

