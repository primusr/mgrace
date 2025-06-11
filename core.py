import os
import pandas as pd
import numpy as np
import nltk
import time
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from wordcloud import WordCloud
from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from googletrans import Translator
import stopwordsiso as custom_iso_stopwords
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import google.generativeai as genai
import base64
import streamlit as st
from fpdf import FPDF

DetectorFactory.seed = 0

nltk.download('vader_lexicon', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

CONFIG = {
   "filipino_lexicon_path": "/content/Fil_words_converted.csv", # Ensure this file is uploaded
    "output_sentiment_comparison_csv": "sentiment_analysis_results_comparison.csv",
    "output_all_gemini_recs_csv": "all_gemini_recommendations.csv",
    "default_feedback_column_names": ['feedback', 'comments', 'Comment', 'Strengths'],
    "lda_stopwords": ['mam', 'maam', 'sir', 'po', 'lang', 'naman', 'wala', 'nya', 'sana', 'mag', 'kasi', 'wag', 'tsaka', 'di', 'pang', 'pag', 'thank', 'thankyou', 'none', 'nothing', 'po', 'ako', 'naman', 'naman po','kayo'],
    "find_optimal_lda_topics": True,  # Set to False to skip optimal topic search for speed
    "lda_topic_range_start": 3,
    "lda_topic_range_end": 10, # Keep this moderate for reasonable runtime
    "lda_topic_range_step": 1,
    "lda_default_num_topics": 6,
    "lda_passes": 25,
    "lda_random_state": 42,
    "gemini_model_name": 'models/gemini-2.0-flash-lite', #Option to change Gemini Version
    "top_n_words_for_topic_labeling": 10,
    "top_n_words_for_topic_gemini_summary": 7,
    "num_topics_for_overall_gemini_summary": 3,
    "num_ai_recs_per_topic_subset": 3, # How many topics to get detailed AI recs for
    "api_call_delay": 1.5 # Increased default delay for Gemini
}

GOOGLE_API_KEY = "AIzaSyCTa1UnujR0fTUQ2tjotO33k71M15-Ja_I"
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(CONFIG["gemini_model_name"])
else:
    gemini_model = None

iso_stopwords = custom_iso_stopwords.stopwords(["en", "tl"])
nltk_en_stopwords = set(nltk_stopwords.words('english'))
combined_stopwords = set(CONFIG["lda_stopwords"]).union(iso_stopwords).union(nltk_en_stopwords)
preprocessing_stopwords = set(iso_stopwords).union(nltk_en_stopwords)
lemmatizer = WordNetLemmatizer()
translator = Translator()
vader_analyzer_english = SentimentIntensityAnalyzer()

filipino_positive_keywords = ['magaling', 'mahusay', 'matalino', 'mabait']
filipino_negative_keywords = ['hindi', 'pangit', 'masama', 'mahirap']

def read_csv_with_fallback_encoding(uploaded_file):
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            uploaded_file.seek(0)
            continue
    return None

def preprocess_text(text):
    import re
    import string
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in preprocessing_stopwords and len(t) > 1]
    return ' '.join(tokens)

def clean_dataframe(df, preferred_columns):
    for col in preferred_columns:
        if col in df.columns:
            df = df[df[col].notnull() & (df[col].astype(str) != '0')]
            df = df.rename(columns={col: 'Feedback_Text'})
            return df[['Feedback_Text']].copy()
    return pd.DataFrame(columns=['Feedback_Text'])

def get_vader_sentiment(text):
    try:
        if detect(text) != 'en':
            text = translator.translate(text, dest='en').text
    except: pass
    score = vader_analyzer_english.polarity_scores(text)['compound']
    if score >= 0.05: return score, "Positive"
    elif score <= -0.05: return score, "Negative"
    return score, "Neutral"

def get_filipino_keyword_sentiment(text):
    score = 0
    tokens = text.lower().split()
    for word in filipino_positive_keywords:
        if word in tokens: score += 1
    for word in filipino_negative_keywords:
        if word in tokens: score -= 1
    if score > 0: return score, "Positive"
    elif score < 0: return score, "Negative"
    return score, "Neutral"

def get_textblob_sentiment(text):
    try:
        if detect(text) != 'en':
            text = translator.translate(text, dest='en').text
    except: pass
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def generate_wordcloud(texts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texts))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def make_gemini_request(prompt):
    if not gemini_model:
        return "Gemini API not configured."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

import unicodedata

def export_results_to_pdf(cleaned_df, topic_sentiments, recommendations):
    def clean_text(text):
        # Remove unsupported characters, replace smart quotes, etc.
        return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Sentiment Analysis Report", ln=True)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Summary Statistics", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 10, f"Total Comments: {len(cleaned_df)}")

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Topic Sentiments", ln=True)
    pdf.set_font("Arial", '', 10)
    for _, row in topic_sentiments.iterrows():
        pdf.multi_cell(0, 10, clean_text(f"Topic {row['Topic ID']}: {row['AI Label']}\nTop Words: {row['Top Words']}"))

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Gemini Recommendations", ln=True)
    pdf.set_font("Arial", '', 10)
    for rec in recommendations:
        pdf.multi_cell(0, 10, clean_text(
            f"{rec['Type']}\nPrompt: {rec['Input_Summary_To_Gemini']}\nRecommendation: {rec['Gemini_Recommendations']}\n"
        ))

    buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin1', 'ignore')
    buffer.write(pdf_output)
    buffer.seek(0)
    return buffer



def run_sentiment_analysis_pipeline(df_raw):
    df_cleaned = clean_dataframe(df_raw, CONFIG['default_feedback_column_names'])
    if df_cleaned.empty:
        return None

    df_cleaned['Cleaned_Text'] = df_cleaned['Feedback_Text'].apply(preprocess_text)

    vader_results = df_cleaned['Feedback_Text'].apply(get_vader_sentiment)
    df_cleaned['VADER_Score'], df_cleaned['VADER_Label'] = zip(*vader_results)

    filipino_results = df_cleaned['Cleaned_Text'].apply(get_filipino_keyword_sentiment)
    df_cleaned['Fil_Score'], df_cleaned['Fil_Label'] = zip(*filipino_results)

    tb_results = df_cleaned['Feedback_Text'].apply(get_textblob_sentiment)
    df_cleaned['TB_Polarity'], df_cleaned['TB_Subjectivity'] = zip(*tb_results)

    tokenized_texts = [t.split() for t in df_cleaned['Cleaned_Text'].tolist() if t]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    lda_model = LdaModel(corpus, num_topics=CONFIG['lda_default_num_topics'], id2word=dictionary, 
                         passes=CONFIG['lda_passes'], random_state=CONFIG['lda_random_state'])

    topic_labels = []
    for i in range(lda_model.num_topics):
        top_words = [word for word, _ in lda_model.show_topic(i, topn=5)]
        label = make_gemini_request(f"Suggest a concise label for the topic defined by: {', '.join(top_words)}")
        topic_labels.append({"Topic ID": i, "Top Words": ', '.join(top_words), "AI Label": label})

    topic_sentiments = pd.DataFrame(topic_labels)

    recommendations = []
    for topic in topic_labels:
        prompt = f"For the topic '{topic['AI Label']}', provide 2 actionable teaching recommendations."
        rec = make_gemini_request(prompt)
        recommendations.append({
            'Type': f"Topic {topic['Topic ID']} - {topic['AI Label']}",
            'Input_Summary_To_Gemini': topic['Top Words'],
            'Gemini_Recommendations': rec
        })

    pdf_buffer = export_results_to_pdf(df_cleaned, topic_sentiments, recommendations)

    return df_cleaned, topic_sentiments, recommendations, pdf_buffer
