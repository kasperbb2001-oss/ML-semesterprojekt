import streamlit as st
import pandas as pd
import plotly.express as px
from tools.data_wrangling import process_and_merge_data
from tools.isolation_forest_model import detect_anomalies

# Page configurations
st.set_page_config(page_title="Shopping Center Anomaly App", layout="wide", page_icon="📈")

st.title("📈 Shopping Center Anomaly Detection")
st.markdown("Opdag unormale forbrugsmønstre via Isolation Forest algorithmen.")

# 1. Upload Logic
st.header("1. Upload Datasets")
col1, col2 = st.columns(2)

with col1:
    st.info("Upload de første 2  års data til træning af baseloaden.")
    train_files = st.file_uploader("Træningsdata CSV-filer", accept_multiple_files=True, type=['csv'], key="train")

with col2:
    st.warning("Upload Data for det 3. år til validering og test.")
    test_files = st.file_uploader("Testdata CSV-filer", accept_multiple_files=True, type=['csv'], key="test")

st.divider()

if train_files and test_files:
    if st.button("Klargør & Merge Data", use_container_width=True):
        with st.spinner("Processor CSV-filer, retter tidstempler og outer-joiner data..."):
            train_df = process_and_merge_data(train_files)
            test_df = process_and_merge_data(test_files)
            
            st.session_state['train_df'] = train_df
            st.session_state['test_df'] = test_df
            st.success("Tidsstempler er formateret korrekt og alle 23 datasæt er sucessfuldt merged til ét struktureret datasæt!")

# 2. UI and ML execution
if 'train_df' in st.session_state and 'test_df' in st.session_state:
    train_df = st.session_state['train_df']
    test_df = st.session_state['test_df']
    
    st.header("2. Kør Anomaly Detection (Isolation Forest)")
    
    # Identify common numeric target columns between train and test
    train_cols = train_df.select_dtypes('number').columns
    test_cols = test_df.select_dtypes('number').columns
    common_cols = list(set(train_cols).intersection(test_cols))
    
    if len(common_cols) > 0:
        c1, c2 = st.columns(2)
        target_col = c1.selectbox("Vælg Forbrugsmåler til Analyse", common_cols)
        
        # Default Features
        features = [target_col]
        # Include Out-Temp automatically if found logically to handle seasonality
        temp_cols = [c for c in common_cols if 'temp' in c.lower() or 'ude' in c.lower() or 'outside' in c.lower()]
        
        if temp_cols and temp_cols[0] != target_col:
            features.append(temp_cols[0])
            c2.info(f"Modellen benytter desuden automatisk **`{temp_cols[0]}`** som støtte-variabel for at minimere false-positives under høj-/lavsæsoner.")
        else:
            c2.warning("Ingen udendørs temperatur fundet - modellen kører rent univariat.")

        if st.button("🚀 Start Detection", type="primary", use_container_width=True):
            with st.spinner("Træner Isolation Forest model og søger efter afvigelser..."):
                try:
                    result_df = detect_anomalies(train_df, test_df, features)
                    st.toast("Detektion 100% Gennemført!", icon="✅")
                    
                    st.subheader(f"Test Resultater: {target_col}")
                    
                    # Map anomalies for plotting
                    result_df['Type'] = result_df['Anomaly'].apply(lambda x: 'Anomali' if x == -1 else 'Normal')
                    
                    # Interactive Graphic using Plotly
                    fig = px.scatter(
                        result_df.reset_index(),
                        x='Datetime',
                        y=target_col,
                        color='Type',
                        color_discrete_map={'Normal': '#1f77b4', 'Anomali': '#d62728'},
                        title=f"{target_col}: Observerede Anomalier over Tidslinjen"
                    )
                    
                    # Refine UI to be crisp and clear
                    fig.update_traces(marker=dict(size=4.5, opacity=0.8)) 
                    fig.update_layout(xaxis_title="Dato & Tid", yaxis_title="Forbrugværdi")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Identify anomalies explicitly
                    anomalies_only = result_df[result_df['Anomaly'] == -1]
                    st.error(f"⚠️ **{len(anomalies_only)} specifikke datapunkter** er markeret som afvigelser (potentielle små lækager/fejl).")
                    st.dataframe(anomalies_only.reset_index(), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Fejl under udførelse af algoritmen: {str(e)}")
    else:
        st.error("Der blev ikke fundet nogen delte numeriske datakolonner imellem Train og Test.")
