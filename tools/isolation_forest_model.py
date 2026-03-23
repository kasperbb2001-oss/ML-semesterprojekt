import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
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
        model_features = feature_columns + ['Hour', 'DayOfWeek']
    else:
        model_features = feature_columns

    # Isolation Forest setup
    # Vi sætter auto contamination under træningen for ren læring af strukturen.
    model = IsolationForest(contamination='auto', random_state=42)
    
    # Train purely on Year 1 and Year 2 data
    model.fit(train_clean[model_features])
    
    # Predict exclusively on Year 3 data
    # Frem for at bruge modellens interne "threshold" fra træningen (hvilket kan give 0 anomalier i test),
    # regner vi "anomali-scoren" ud for hver time. Jo lavere score, jo mere ekstrem afvigelse.
    scores = model.decision_function(test_clean[model_features])
    
    # Vi tvinger den her til at finde de 1% mærkeligste timer i det NUVÆRENDE test-år
    import numpy as np
    threshold = np.percentile(scores, 1) # Find the 1% lowest scores
    
    predictions = [-1 if s <= threshold else 1 for s in scores]
    test_clean['Anomaly'] = predictions
    
    return test_clean
