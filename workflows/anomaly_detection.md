# Anomaly Detection Workflow

**Objective:** Detect anomalies in shopping center electricity and water usage based on historic patterns.

## Required Inputs
- **Training data:** CSV files representing Year 1 and Year 2 consumption.
- **Testing data:** CSV files representing Year 3 consumption.
*All CSV files must be semicolon-separated and include `Dato` and `Time` columns.*

## Tools Used
1. `tools/data_wrangling.py`: Preprocesses the CSVs by extracting datetimes, setting the index, and joining 23 datasets together via an outer join. Applies forward-fill to missing values.
2. `tools/isolation_forest_model.py`: Trains an Isolation Forest to establish a baseline, then detects outliers in Year 3.

## Expected Outputs
- A Streamlit interface displaying clean data structures and interactive Plotly scatterplots where identified anomalies are colored red.

## Handling Edge Cases
- **Missing or non-existent columns:** Handled defensively via Python properties.
- **Isolation Forest training errors:** If columns contain entirely NaNs, they skip training smoothly via subset dropna.
