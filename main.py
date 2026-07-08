from src.data_cleaning import cleaning_summary, clean_data, save_clean_data
from src.data_ingestion import load_data, print_dataset_info
from src.custom_exception import log_exception
from src.eda import country_analysis, product_analysis, revenue_analysis,weekday_analysis,customer_analysis,monthly_sales_analysis,coorelation_analysis,categorical_summary,numerical_summary,dataset_overview
import logging
from src.visualization import plot_monthly_sales, plot_top_customers, plot_revenue_distribution, plot_country_revenue,plot_top_products,plot_coorelation_heatmap,plot_top_product_treemap,plot_country_revenue_map
from src.feature_engineering import feature_engineering_pipeline,save_features_dataset
from src.customer_segmentation import save_model, save_segmented_data, segmentation_pipeline
from src.sales_forecasting import forecasting_pipeline, save_forecasting_model
from src.churn_prediction import save_churn_model,churn_prediction_pipeline,save_churn_prediction
from src.anomoly_detection import anomaly_detection_pipeline, save_anomaly_model, save_anomaly_results
import pandas as pd
from pathlib import Path

raw_data_path = r"data/raw/Online Retail.xlsx"
clean_data_path = r"data/processed/Clean File.csv"
rfm_features_path = r"data/processed/rfm_features.csv"
rfm_features_path = r"data/processed/rfm_features.csv"
customers_features_path = r"data/processed/customers_features.csv"
churn_features_path = r"data/processed/churn_features.csv"
sales_features_path = r"data/processed/sales_features.csv"
segmented_data_path = r"data/processed/customer_segments.csv"
segmentation_model_path = r"model/segmentation_model.pkl"
forecast_model_path = r"model/forecast_model.pkl"
churn_model_path = r"model/churn_model.pkl"
churn_prediction_path = r"data/processed/churn_prediction.csv"
anomaly_model_path = r"model/anomaly_model.pkl"
anomaly_results_path = r"data/processed/anomaly_results.csv"

def main():
    try:
        # EXPLORATORY DATA ANALYSIS
        if Path(clean_data_path).exists():
            cleaned_df = pd.read_csv(clean_data_path)
            logging.info("CLEANED DATA FILE EXIST")
            overview = dataset_overview(cleaned_df)
            print(f"Dataset Overview:\n {overview}")

            print(f"NUMERICAL DATA: \n{numerical_summary(cleaned_df)}")
            print(f"CATEGORICAL DATA: \n{categorical_summary(cleaned_df)}")
            print(f"REVENUE ANALYSIS: \n{revenue_analysis(cleaned_df)}")

            #  ------  VISUALITIONS -----
            revenue_map = plot_country_revenue_map(cleaned_df)
            revenue_map.show()
            coorelation_map = plot_coorelation_heatmap(cleaned_df)
            coorelation_map.show()
            top_products_fig = plot_top_product_treemap(cleaned_df)
            top_products_fig.show()
            country_revenue = plot_country_revenue_map(cleaned_df)
            country_revenue.show()
            monthly_sales_fig = plot_monthly_sales(cleaned_df)
            monthly_sales_fig.show()


            # ---- FEATURE ENFINEERING ----- 
            feature_output = (feature_engineering_pipeline(cleaned_df))
            rfm_df = (feature_output["rfm_df"])
            customer_feature_df = (feature_output["customer_df"])
            churn_feature_df = (feature_output["churn_df"])
            sales_feature_df = (feature_output["sales_df"])
            save_features_dataset(rfm_df, rfm_features_path)
            save_features_dataset(customer_feature_df,customers_features_path)
            save_features_dataset(churn_feature_df,churn_features_path)
            save_features_dataset(sales_feature_df,sales_features_path)
            logging.info(f"RFM Shape: {rfm_df.shape}\n")
            logging.info(f"Customer Feature Shape: {customer_feature_df.shape}\n")
            logging.info(f"Churn Feature Shape: {churn_feature_df.shape}\n")
            logging.info(f"Sales Features Shape: {sales_feature_df.shape}\n")


            # --- CUSTOMER SEGMENTATION ---
            logging.info("Starting Customer Segmentation..")
            if Path(segmented_data_path).exists():
                segmented_df = pd.read_csv(segmented_data_path)

            else:
                segmentation_output = (segmentation_pipeline(rfm_df,n_clusters=4))
                segmented_df = (segmentation_output["segmented_df"])
                cluster_summary_df = (segmentation_output["summary"])
                segmentation_model = (segmentation_output["model"])
                save_clean_data(segmented_df,segmented_data_path)
                save_model(segmentation_model,segmentation_model_path)
                print(f"{cluster_summary_df}\n")


            # --- SALES FORECASTING ---
            logging.info("Starting Sales Forecasting...")
            sales_feature_df = pd.read_csv(sales_features_path)
            forecasting_output = (forecasting_pipeline(sales_feature_df))
            forecast_model = forecasting_output["model"]
            forecast_metrics = forecasting_output["metrics"]
            next_day_forecast = forecasting_output["forecast"]
            forecast_fig = forecasting_output["forecast_fig"]
            save_forecasting_model(forecast_model,forecast_model_path)
            print(f"FORECASTING METRICS\n{forecast_metrics}")
            print(f"Rs. {next_day_forecast:,.2f}")
            forecast_fig.show()
            logging.info("Forecast pipeline run successfully")


            # --- CHURN PREDICTION ---
            logging.info("Churn Prediction started...")
            churn_output = (churn_prediction_pipeline(churn_feature_df))
            churn_model = churn_output["model"]
            churn_metrics = churn_output["metrics"]
            feature_importance = churn_output["feature_importance"]
            churn_score = churn_output["churn_scores"]
            save_churn_model(churn_model,churn_model_path)
            save_churn_prediction(churn_score,churn_prediction_path)
            print(f"CHURN METRICS\n{churn_metrics}\n")
            print(f"TOP CHURN DRIVERS\n{feature_importance.head(10)}\n")
            logging.info("Churn Prediction ran successfully...")


            # --- ANOMALY DETECTION ---
            logging.info("Anomaly Detection started...")
            anomaly_output = (anomaly_detection_pipeline(sales_feature_df))
            anomaly_model = (anomaly_output["model"])
            anomaly_results = (anomaly_output["result_df"])
            anomaly_summary_df = (anomaly_output["summary"])
            top_anomalies = (anomaly_output["anomaly_fig"])
            anomaly_count = (anomaly_output["anomaly_count"])
            anomaly_fig = (anomaly_output["anomaly_fig"])
            save_anomaly_model(anomaly_model, anomaly_model_path)
            save_anomaly_results(anomaly_results, anomaly_results_path)
            print(f"Anomaly Count{anomaly_count}\n")
            logging.info("Anomaly Detection completed...")
            anomaly_fig.show()


        else:

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