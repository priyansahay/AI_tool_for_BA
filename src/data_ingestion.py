from pathlib import Path
import pandas as pd
import logging
from src.custom_exception import log_exception


try:
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)



    # FILE VALIDATION
    def validate_file(file_path: str) -> bool:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found : {file_path}\n"
            )
        if path.suffix not in [".csv", ".xlsx", ".xlsv"]:
            raise ValueError(
                "Supported error is .csv, .xlsx, .xlsv \n"
            )
        logger.info("File Validation Successfull\n")
        return True



    # LOADING DATA
    def load_data(file_path: str) -> pd.DataFrame:
        validate_file(file_path)
        logger.info(f"Loading Dataset: {file_path}")

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        logger.info("Dataset uploaded successfully\n")
        logger.info(f"Dataset Shape {df.shape}")
        return df



    # DATASET SUMMARY
    def dataset_summary(df: pd.DataFrame) -> dict:
        summary = {
            "rows": df.shape[0],
            "columns" : df.shape[1],
            "missing_values" : df.isnull().sum().sum(),
            "duplicate_rows" : df.duplicated().sum()
        }
        return summary



    # COLUMNS INFORMATION
    def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
        column_info = pd.DataFrame({
            "Column_Name" : df.columns,
            "Data_Type" : df.dtypes.values
        })
        return column_info



    # DATA QUALITY REPORT
    def generate_data_quality_report(df: pd.DataFrame) -> dict:
        report = {
            "Missing_Values_Per_Column" : df.isnull().sum().to_dict(),
            "Duplicate_rows" : int(df.duplicated().sum()),
            "Data_Types": df.dtypes.astype(str).to_dict()
        }
        return report

    # DISPLAY INFORMATION
    def print_dataset_info(df: pd.DataFrame) -> None:
        print("DATASET INFORMATION\n")
        summary = dataset_summary(df)
        print(f"ROWS     : {summary['rows']}")
        print(f"COLUMNS     : {summary['columns']}")
        print(f"Missing Values   :{summary['missing_values']}")
        print(f"Duplicates.    :{summary['duplicate_rows']}")
        print("\nColumns")
        for i in df.columns:
            print(f".{i}")
            

except Exception as e:
    log_exception(e)
    raise

    
