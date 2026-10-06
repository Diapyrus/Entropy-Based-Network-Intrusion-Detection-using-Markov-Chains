import polars as pl
import pandas as pd
from typing import List, Optional
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif

def load_data_for_mi(parquet_path: str, feature_cols: List[str], label_col: str) -> pd.DataFrame:
    """
    Reads the dataset and extracts the necessary features and label.
    
    Args:
        parquet_path: Path to the parquet file.
        feature_cols: List of column names for the features.
        label_col: Name of the label column.
        
    Returns:
        A Pandas DataFrame containing only the selected features and label.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
        
    # Read via polars for performance, then convert to pandas
    df = pl.read_parquet(path).select(feature_cols + [label_col]).to_pandas()
    return df

def calculate_mutual_information(
    parquet_path: str,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "Label",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculates the Mutual Information I(X; Y) between the given features and the label.
    
    Args:
        parquet_path: Path to the processed parquet file.
        feature_cols: List of feature names. Defaults to the 5 tracked symmetric features if None.
        label_col: Name of the target variable column.
        random_state: Seed for reproducibility in mutual_info_classif.
        
    Returns:
        A Pandas DataFrame mapping each feature to its Mutual Information score,
        sorted in descending order.
    """
    if feature_cols is None:
        feature_cols = [
            "Flow_Duration_sym",
            "Out_Degree_sym",
            "Tot_Fwd_Pkts_sym",
            "TotLen_Fwd_Pkts_sym",
            "TCP_Flags_Int_sym"
        ]
        
    df = load_data_for_mi(parquet_path, feature_cols, label_col)
    
    X = df[feature_cols]
    y = df[label_col]
    
    # mutual_info_classif expects discrete features if we set discrete_features=True.
    # Since our _sym features are discretized symbols, we can explicitly mark them as discrete
    # to get exact Shannon mutual information rather than KNN-based estimation.
    mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=random_state)
    
    results = []
    for feature, score in zip(feature_cols, mi_scores):
        results.append({
            "Feature": feature,
            "Mutual Information (bits)": score
        })
        
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Mutual Information (bits)", ascending=False).reset_index(drop=True)
    
    return results_df

if __name__ == "__main__":
    # Example usage
    parquet_file = Path(__file__).resolve().parent.parent / "data" / "processed.parquet"
    if Path(parquet_file).exists():
        print("Calculating Mutual Information...")
        df_mi = calculate_mutual_information(parquet_file)
        print("\nFeature Importance (Mutual Information):")
        print(df_mi)
    else:
        print(f"File not found: {parquet_file}")
