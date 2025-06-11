import streamlit as st
import pandas as pd
import numpy as np
import io
from .demo_local_patched import (
    preprocess_text,
    clean_dataframe,
    get_vader_sentiment_english,
    get_vader_sentiment_augmented,
    get_filipino_keyword_sentiment,
    get_textblob_polarity_subjectivity,
    CONFIG
)

st.set_page_config(page_title="Sentiment & Topic Modeling Dashboard", layout="wide")

st.title("📊 Student Feedback Analyzer")
st.markdown("This app analyzes sentiment and topics from student feedback using NLP and Generative AI.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully.")
        
        # Clean and preprocess
        df_cleaned = clean_dataframe(df_raw.copy(), CONFIG["default_feedback_column_names"])
        df_cleaned['Cleaned_Text_Main'] = df_cleaned['Feedback_Text'].apply(lambda x: preprocess_text(x, for_lda=True))
        df_cleaned.dropna(subset=['Cleaned_Text_Main'], inplace=True)

        # Run sentiment analysis
        with st.spinner("Running sentiment analysis..."):
            df_cleaned[['VADER_Score_Eng', 'VADER_Sentiment_Eng']] = df_cleaned['Feedback_Text'].apply(lambda x: pd.Series(get_vader_sentiment_english(x)))
            df_cleaned[['VADER_Score_Aug', 'VADER_Sentiment_Aug']] = df_cleaned['Cleaned_Text_Main'].apply(lambda x: pd.Series(get_vader_sentiment_augmented(x)))
            df_cleaned[['Filipino_Keyword_Score', 'Filipino_Keyword_Sentiment', _, _]] = df_cleaned['Cleaned_Text_Main'].apply(lambda x: pd.Series(get_filipino_keyword_sentiment(x)))
            df_cleaned[['TextBlob_Polarity', 'TextBlob_Subjectivity']] = df_cleaned['Feedback_Text'].apply(lambda x: pd.Series(get_textblob_polarity_subjectivity(x)))

        # Display data
        st.subheader("📋 Processed Data")
        st.dataframe(df_cleaned.head(20), use_container_width=True)

        # Visualizations
        st.subheader("📈 Sentiment Distribution")
        st.bar_chart(df_cleaned['VADER_Sentiment_Aug'].value_counts())

        st.subheader("🔍 TextBlob Polarity vs. Subjectivity")
        st.scatter_chart(df_cleaned[['TextBlob_Subjectivity', 'TextBlob_Polarity']])

        st.download_button("Download Processed CSV", df_cleaned.to_csv(index=False), file_name="sentiment_results.csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
