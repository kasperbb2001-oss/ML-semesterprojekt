import pandas as pd
from typing import List, Any

def process_and_merge_data(uploaded_files: List[Any]) -> pd.DataFrame:
    """
    Reads multiple CSV files (separated by ';'), reconstructs proper datetimes from 
    'Dato' and 'Time' columns, aligns them via DatetimeIndex, and merges via outer join.
    Handles NaNs via forward fill.
    """
    if not uploaded_files:
        return pd.DataFrame()
        
    dfs = []
    
    for file in uploaded_files:
        # Read the semicolon-separated file
        df = pd.read_csv(file, sep=';', encoding='utf-8', decimal=',', thousands='.')
        
        # Datetime processing
        if 'Dato' in df.columns and 'Time' in df.columns:
            # Extract start time from Time col, e.g., "00:00-01:00" -> "00:00", or "  00:00 " --> "00:00"
            # Some files might have "00:00 - 01:00"
            start_hours = df['Time'].astype(str).str.split('-').str[0].str.strip()
            
            # Create a string for datetime
            datetime_str = df['Dato'].astype(str) + ' ' + start_hours
            
            # Convert to pandas datetime more flexibly
            df['Datetime'] = pd.to_datetime(datetime_str, dayfirst=True, errors='coerce')
            
            # Kun drop rows hvor datetime fejlede
            df.dropna(subset=['Datetime'], inplace=True)
            df.set_index('Datetime', inplace=True)
            df.drop(columns=['Dato', 'Time'], inplace=True, errors='ignore')
            
        dfs.append(df)
        
    if not dfs:
        return pd.DataFrame()
        
    # Outer join all dataframes based on the Datetime index
    merged_df = pd.concat(dfs, axis=1, join='outer')
    
    # Remove potentially duplicated columns with identical names entirely
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    
    # Try to clean up any str columns to numeric (if decimal/thousands didn't catch it)
    for col in merged_df.columns:
        if merged_df[col].dtype == 'object':
            temp = merged_df[col].astype(str)
            # Remove string artifacts, spaces, and specifically letters (like " kWh", " MWh")
            temp = temp.str.replace(r'[^\d,\.-]', '', regex=True)
            # Convert Danish logic to machine logic
            temp = temp.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            merged_df[col] = pd.to_numeric(temp, errors='coerce')
            
    # Fill small gaps/NaNs (forward fill then backward fill for start)
    merged_df = merged_df.ffill().bfill()
    
    # Drop rows that are completely empty across all non-index columns
    merged_df.dropna(how='all', inplace=True)
    
    # KUN drop tomme kolonner, HVIS resultatet ikke bliver helt tomt
    if len(merged_df) > 0:
        cleaned_df = merged_df.dropna(axis=1, how='all')
        if not cleaned_df.empty and len(cleaned_df.columns) > 0:
            merged_df = cleaned_df
    
    # Sort index chronologically
    merged_df.sort_index(inplace=True)
    
    return merged_df
