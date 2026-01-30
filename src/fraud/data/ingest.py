import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# Define the path to the source CSV file from root directory
RAW_REL_PATH = Path("data/raw/creditcard.csv")

# Find the Project Root Directory
project_root = Path(__file__).resolve().parents[3]
raw_data_path = project_root / RAW_REL_PATH


# Config columns
@dataclass(frozen=True)
class IngestConfig:
    # Canonical schema assumptions
    RAW_COLUMNS = [ "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10","V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Class"]
    TARGET_COL = "Class"
    FEATURE_COLS = ["Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]

# Function to ingest transaction data from a CSV file
def ingest_data(file_path: str) -> pd.DataFrame:
    """
    Ingest transaction data from a CSV file.

    Parameters: file_path (str): The path to the CSV file containing transaction data.

    Perform: Read-only checks on the file to ensure it exists and is accessible.
    Checks: Validate the structure of the data (e.g., required columns).
    Shape of dataset: Ensure the dataset has the expected number of rows and columns.
    Data Types: Check that the data types of each column are as expected.
    Hard Sanity Checks: Verify No missing values, Class contains only {0,1}, No duplicate rows, All feature columns are numeric (if any of these fail, fix before proceeding)
    Log Dataset Stats: Log basic statistics about the dataset (e.g., number of records, distribution of classes, fraud count, fraud percentage).
    
    Returns:
    pd.DataFrame: A DataFrame containing the ingested transaction data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"'\n Successfully ingested data from {file_path} \n")

        # Perform initial checks and validations
        if data.empty:
            print("\nWarning: The dataset is empty.\n")
            return pd.DataFrame()

        # Check for expected columns
        expected_columns = IngestConfig.RAW_COLUMNS
        missing_columns = [col for col in expected_columns if col not in data.columns]
        if missing_columns:
            print(f"\n Warning: Missing columns in the dataset: {missing_columns} \n")
        else:
            print("\n All expected columns are present.\n")    
            print(f"\n Column names: {data.columns.tolist()}\n")
        
        # Check for target column values
        if 'Class' in data.columns:
            unique_classes = data['Class'].unique()
            if not set(unique_classes).issubset({0, 1}):
                print(f"\n Warning: 'Class' column contains unexpected values: {unique_classes}\n")
            else:
                print(f"\n 'Class' column contains valid values: {unique_classes}\n")
        else:
            print("\n Warning: 'Class' column not found in the dataset.\n")

        # Data type checks
        for col in IngestConfig.FEATURE_COLS:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    print(f"\n Warning: Column '{col}' is not numeric.\n")
            else:
                print(f"\n Warning: Column '{col}' not found in the dataset.\n")

        # Returns a dictionary: {dtype: [column_names]}
        groups = data.columns.to_series().groupby(data.dtypes).groups
        # Print the result nicely
        print("\n Data types of columns: \n")
        for dtype, cols in groups.items():
            print(f"{dtype}: {list(cols)}")

        # Hard sanity checks
        if data.isnull().values.any():
            print("\n Warning: The dataset contains missing values.\n")
        if data.duplicated().any():
            print("\n Warning: The dataset contains duplicate rows.\n")

        
        # Log dataset statistics
        print(f"\n Dataset shape: {data.shape}")
        print(f"\n Number of records: {len(data)}")
        print(f"\n Number of features: {data.shape[1]}")
        
        # Log missing values only if any exist
        if data.isnull().values.any():
            print(f"\n Missing values found:\n{data.isnull().sum()}")
        else:
            print("\n No missing values detected.")

        # Log duplicate rows only if any exist
        print(f"\n Number of duplicate rows: {data.duplicated().sum()}")
        if 'Class' in data.columns:
            fraud_count = data['Class'].sum()
            total_count = len(data)
            fraud_percentage = (fraud_count / total_count) * 100
            print(f"\n Fraud count: {fraud_count}")
            print(f"\n Fraud percentage: {fraud_percentage:.2f}%")
        else:
            print("\n Warning: 'Class' column not found in the dataset.")
            print("\n Skipping fraud detection statistics.")

        return data
    except Exception as e:
        print(f"\n Error ingesting data from {file_path}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    ingest_data(raw_data_path)