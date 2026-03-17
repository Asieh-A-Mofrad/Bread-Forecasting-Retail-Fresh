# src/data/io.py
import pandas as pd
import os
from pathlib import Path

bread_path = "../data/Bread/"

bread_path = Path(bread_path)
sales_hourly_path = bread_path / "sales_hourly"


def _to_numeric_if_possible(series: pd.Series) -> pd.Series:
    """
    Convert to numeric only when the full column can be safely parsed.
    This preserves the old pandas `errors='ignore'` behavior.
    """
    try:
        return pd.to_numeric(series, errors="raise", downcast="integer")
    except (TypeError, ValueError):
        return series


def read_data(folder_path):
    """
    Reads and combines CSV files from nested folders into a single DataFrame.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        pd.DataFrame: A combined DataFrame.
    """
    # List all CSV file paths in the given folder and its subfolders
    files = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser(folder_path))
        for f in fn
        if f.endswith('.csv')
    ]

    # Read all CSV files into a list (skip empty files)
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        if not df.empty and not df.isna().all().all():  # skip all-empty files
            dfs.append(df)

    # Combine all data once
    if dfs:  # only concat if list not empty
        combined_data = pd.concat(dfs, ignore_index=True)
    else:
        combined_data = pd.DataFrame()

    # Ensure proper data types
    if not combined_data.empty:
        if "date" in combined_data.columns:
            combined_data["date"] = pd.to_datetime(combined_data["date"], errors="coerce")
        if "gln" in combined_data.columns:
            combined_data["gln"] = _to_numeric_if_possible(combined_data["gln"])
        if "eanCode" in combined_data.columns:
            combined_data["eanCode"] = _to_numeric_if_possible(combined_data["eanCode"])

    return combined_data
