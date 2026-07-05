from src.data_cleaning import cleaning_summary, clean_data, save_clean_data
from src.data_ingestion import load_data, print_dataset_info
from src.custom_exception import log_exception
from src.eda import country_analysis, product_analysis, revenue_analysis,weekday_analysis,customer_analysis,monthly_sales_analysis,coorelation_analysis,categorical_summary,numerical_summary,dataset_overview
import logging
from src.visualization import plot_monthly_sales, plot_top_customers, plot_revenue_distribution, plot_country_revenue,plot_top_products,plot_coorelation_heatmap,plot_top_product_treemap,plot_country_revenue_map
import pandas as pd
from pathlib import Path

raw_data_path = r"data/raw/Online Retail.xlsx"
clean_data_path = r"data/processed/Clean File.csv"

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

            # VISUALITIONS
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