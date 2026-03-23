import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list, contamination_pct: float = 1.0) -> pd.DataFrame:
    """
    Trains an Isolation Forest model on historical train_df data using the specified feature_columns.
    Predicts outliers on test_df based on the learned seasonal baseload.
    Returns test_df with an added 'Anomaly' flag (-1 for outlier, 1 for normal).
    """
    # Defensive cleanup based on features, ensuring no unhandled NaNs slip into sklearn
    train_clean = train_df.dropna(subset=feature_columns).copy()
    test_clean = test_df.dropna(subset=feature_columns).copy()

    if train_clean.empty:
        raise ValueError("Trænings-datasættet er tomt for de valgte kolonner.")
    if test_clean.empty:
        raise ValueError("Test-datasættet er tomt for de valgte kolonner.")

    # Ekstrahér tidsbaserede dimensioner automatisk, så algoritmen lærer "hvornår" noget er normalt.
    # Uden dette ved modellen f.eks. ikke at et forbrug på 0 kl. 12:00 i en butik er unormalt.
    if isinstance(train_clean.index, pd.DatetimeIndex):
        train_clean['Hour'] = train_clean.index.hour
        train_clean['DayOfWeek'] = train_clean.index.dayofweek
        test_clean['Hour'] = test_clean.index.hour
        test_clean['DayOfWeek'] = test_clean.index.dayofweek
        
        # Også tilknyt en beregning af HASTIGHEDEN hvormed forbruget ændrer sig.
        # Hvis den pludselig falder fra 9 til 0, vil 'diff' funktionen råbe vagt i gevær.
        diff_cols = []
        for col in feature_columns:
            diff_col_name = f"{col}_diff"
            train_clean[diff_col_name] = train_clean[col].diff().fillna(0)
            test_clean[diff_col_name] = test_clean[col].diff().fillna(0)
            diff_cols.append(diff_col_name)
            
        model_features = feature_columns + ['Hour', 'DayOfWeek'] + diff_cols
    else:
        model_features = feature_columns

    # Isolation Forest setup
    # Vi bruger brugerens valgte procent som "contamination" for træningsdataene.
    # Hvis brugeren sætter den til 10%, lærer algoritmen en MEGET stram grænse for normalitet,
    # og derved vil alle de høje "toppe" blive flagget i test-settet, uanset hvor mange de er.
    model = IsolationForest(contamination=contamination_pct / 100.0, random_state=42)
    
    # Train purely on Year 1 and Year 2 data
    model.fit(train_clean[model_features])
    
    # Predict exclusively on Year 3 data ved hjælp af den stramme grænse vi lige har lært
    predictions = model.predict(test_clean[model_features])
    test_clean['Anomaly'] = predictions
    
    return test_clean
