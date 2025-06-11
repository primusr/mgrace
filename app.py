import streamlit as st
import pandas as pd
from core import run_sentiment_analysis_pipeline, read_csv_with_fallback_encoding, generate_wordcloud


st.title("Feedback Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df_raw = read_csv_with_fallback_encoding(uploaded_file)
    if df_raw is not None:
        results = run_sentiment_analysis_pipeline(df_raw)

        if results:
            cleaned_df, topic_sentiments, recommendations, pdf_buffer = results

            st.header("Standard VADER Sentiment Analysis")
            st.markdown("Shows sentiment using the VADER model (ideal for English).")
            st.dataframe(cleaned_df[['Feedback_Text', 'VADER_Score', 'VADER_Label']].head())

            st.header("Augmented VADER Sentiment Analysis")
            st.markdown("Includes preprocessing and translation if text is not in English.")

            st.header("Filipino Keyword Sentiment Analysis")
            st.dataframe(cleaned_df[['Cleaned_Text', 'Fil_Score', 'Fil_Label']].head())

            st.header("TextBlob Polarity/Subjectivity Analysis")
            st.dataframe(cleaned_df[['TB_Polarity', 'TB_Subjectivity']].head())

            st.header("Processed DataFrame with Sentiment Scores (Sample)")
            st.dataframe(cleaned_df.head())

            st.header("Overall System Sentiment Scores & Distributions")
            st.bar_chart(cleaned_df['VADER_Label'].value_counts())

            st.header("Main Topic Modeling (LDA), AI Labeling & Overall Coherence")
            st.dataframe(topic_sentiments)

            st.header("Overall Sentiment Per Topic")
            st.markdown("**(Placeholder for logic to associate sentiments with topics if needed)**")

            st.header("Gemini AI Recommendations for Each Labeled Topic")
            for rec in recommendations:
                st.subheader(rec['Type'])
                st.markdown(f"**Prompt**: {rec['Input_Summary_To_Gemini']}")
                st.markdown(f"**Recommendation**: {rec['Gemini_Recommendations']}")

            st.header("Visualization and Sentiments")
            st.subheader("Sentiment Visualizations")
            generate_wordcloud(cleaned_df['Cleaned_Text'])

            st.subheader("Gemini-Powered Recommendations")
            for rec in recommendations:
                st.markdown(f"**{rec['Type']}**: {rec['Gemini_Recommendations']}")

            st.download_button(
                label="📄 Download Full PDF Report",
                data=pdf_buffer,
                file_name="sentiment_report.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Analysis failed. Please check your data format and content.")
    else:
        st.error("Could not read the uploaded CSV file. Try reformatting or re-saving it.")