import os
import time
import re
import pandas as pd
import nltk
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from gensim import corpora, models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from googletrans import Translator
from langdetect import detect
import google.generativeai as genai

# Setup
nltk.download('vader_lexicon')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
vader_analyzer = SentimentIntensityAnalyzer()
translator = Translator()
genai.configure(api_key="your-api-key-here")
gemini_model = genai.GenerativeModel('gemini-pro')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_vader_sentiment(text):
    try:
        lang = detect(text)
        if lang != 'en' and len(text.split()) > 2:
            text_en = translator.translate(text, dest='en').text
        else:
            text_en = text
    except Exception:
        text_en = text

    compound = vader_analyzer.polarity_scores(text_en)['compound']
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return compound, sentiment


def clean_and_analyze(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    first_col = df.columns[0]
    df = df[[first_col]].dropna()
    df.columns = ['text']
    df['clean_text'] = df['text'].astype(str).apply(lambda x: re.sub(r'\W+', ' ', x).lower())
    df['tokens'] = df['clean_text'].apply(lambda x: [word for word in x.split() if word not in stop_words])
    df['sentiment_data'] = df['clean_text'].apply(get_vader_sentiment)
    df['compound'] = df['sentiment_data'].apply(lambda x: x[0])
    df['sentiment'] = df['sentiment_data'].apply(lambda x: x[1])
    return df


def generate_wordcloud(df):
    text = ' '.join([' '.join(tokens) for tokens in df['tokens']])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("static/wordcloud.png")
    plt.close()


def generate_lda(df, num_topics=5):
    dictionary = corpora.Dictionary(df['tokens'])
    corpus = [dictionary.doc2bow(text) for text in df['tokens']]
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        if f:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
            f.save(filepath)

            df = clean_and_analyze(filepath)
            generate_wordcloud(df)
            lda_model = generate_lda(df)

            summary = "Sentiment Summary:\n"
            summary += df['sentiment'].value_counts().to_string() + "\n\n"
            summary += "Topics:\n"
            for i in range(lda_model.num_topics):
                topic_words = [w for w, p in lda_model.show_topic(i)]
                summary += f"  - Topic {i}: {', '.join(topic_words)}\n"

            try:
                gemini_response = gemini_model.generate_content(summary)
                recommendation = gemini_response.text
            except:
                recommendation = "Gemini API failed."

            return render_template("result.html", 
                                   sentiment=df['sentiment'].value_counts().to_dict(),
                                   recommendation=recommendation,
                                   topics=[lda_model.show_topic(i, topn=5) for i in range(lda_model.num_topics)])
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
