import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Setup paths to import from 6_Bonus_Theory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BONUS_DIR = PROJECT_ROOT / "6_Bonus_Theory"
sys.path.append(str(BONUS_DIR))

# Safe imports for theoretical modules
try:
    from theory_metrics import calculate_mutual_information
    from huffman_eval import evaluate_huffman_compression
except ImportError as e:
    st.error(f"Import Error: {e}. Make sure 6_Bonus_Theory modules exist.")

# App Configuration
st.set_page_config(page_title="IDS Dashboard", layout="wide", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Static Metrics (Evaluation)", "Dynamic Theoretical Proofs"])

st.sidebar.markdown("---")
st.sidebar.info("Information Theory-based IDS\n\nMarkov Chains & Cross-Entropy")

DATA_PATH = PROJECT_ROOT.parent.parent / "data" / "processed.parquet"

if section == "Static Metrics (Evaluation)":
    st.title("Final Evaluation Metrics")
    st.markdown("These metrics were pre-computed during the Evaluation Phase (Phase 5).")
    
    # Display Deterministic Metrics as defined by the user
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", "99.79%")
    col2.metric("Recall", "99.98%")
    col3.metric("F1-Score", "99.89%")
    col4.metric("ROC-AUC", "1.0000")
    
    st.markdown("---")
    
    # Optionally load the detection results if available
    detection_csv = PROJECT_ROOT / "5_Evaluation" / "results" / "detection_results.csv"
    if detection_csv.exists():
        st.subheader("Detection Results Preview")
        try:
            df_det = pd.read_csv(detection_csv)
            st.dataframe(df_det.head(10))
            
            # Simple plot of Deviation over Window Index
            if "Deviation" in df_det.columns and "Window_Index" in df_det.columns:
                st.subheader("Entropy Deviation over Time")
                # Plot the first 1000 windows for performance
                fig = px.line(df_det.head(1000), x="Window_Index", y="Deviation", title="Entropy Deviation (First 1000 Windows)")
                fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Threshold")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Could not load detection results: {e}")
    else:
        st.info("No pre-computed detection results found at `5_Evaluation/results/detection_results.csv`.")
        
    # Load training summary if available
    training_csv = PROJECT_ROOT / "3_Training" / "models" / "training_summary.csv"
    if training_csv.exists():
        st.subheader("Training Summary")
        try:
            df_train = pd.read_csv(training_csv)
            st.dataframe(df_train)
        except Exception as e:
            st.warning(f"Could not load training summary: {e}")

elif section == "Dynamic Theoretical Proofs":
    st.title("Information Theory Live Proofs")
    st.markdown("Execute the theoretical algorithms live against the `processed.parquet` dataset.")
    
    st.subheader("1. Mutual Information (Feature Selection)")
    st.markdown("Calculates $I(X; Y)$ to mathematically prove which features are optimal for detection.")
    
    if st.button("Calculate Mutual Information"):
        if not DATA_PATH.exists():
            st.error(f"Dataset not found at {DATA_PATH}")
        else:
            with st.spinner("Calculating Mutual Information (this may take a few seconds)..."):
                try:
                    df_mi = calculate_mutual_information(str(DATA_PATH))
                    st.success("Calculation complete!")
                    st.dataframe(df_mi)
                    
                    fig = px.bar(df_mi, x='Feature', y='Mutual Information (bits)', 
                                 title="Mutual Information of Features w.r.t Label",
                                 color='Mutual Information (bits)', color_continuous_scale="Teal")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error during calculation: {e}")

    st.markdown("---")
    
    st.subheader("2. Huffman Compression Ratio")
    st.markdown("Compares the compression efficiency of Benign vs. Attack traffic. A DDoS attack reduces entropy, meaning it should compress more efficiently according to Shannon's Source Coding Theorem.")
    
    if st.button("Run Huffman Compression on Tot_Fwd_Pkts_sym"):
        if not DATA_PATH.exists():
            st.error(f"Dataset not found at {DATA_PATH}")
        else:
            with st.spinner("Isolating 5-minute windows and encoding..."):
                try:
                    df_huff = evaluate_huffman_compression(str(DATA_PATH))
                    st.success("Compression complete!")
                    st.dataframe(df_huff)
                    
                    if not df_huff.empty:
                        fig = px.bar(df_huff, x='Traffic Type', y='Compression Ratio', 
                                     title="Huffman Compression Ratio (Higher is better)",
                                     color='Traffic Type', color_discrete_map={"Benign Traffic (5 min)": "blue", "Attack Traffic (5 min)": "red"})
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error during calculation: {e}")
