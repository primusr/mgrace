
from flask import Flask, request, render_template, send_file
import pandas as pd
import nltk
import re
import gensim
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from gensim import corpora
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googletrans import Translator
from langdetect import detect
import time
import os

app = Flask(__name__)

nltk.download('vader_lexicon')
nltk.download('stopwords')

vader_analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
translator = Translator()

def get_vader_sentiment(text):
    try:
        lang = detect(text)
        if lang != 'en':
            if len(text.split()) > 2:
                time.sleep(0.2)
                text_en = translator.translate(text, dest='en').text
            else:
                text_en = text
        else:
            text_en = text
    except Exception:
        text_en = text

    compound_score = vader_analyzer.polarity_scores(text_en)['compound']
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return compound_score, sentiment

def clean_dataframe(file_path):
    df = pd.read_csv(file_path)
    for col in df.columns:
        if col.lower() in ['text', 'comment', 'feedback', 'message']:
            df.rename(columns={col: 'text'}, inplace=True)
            break
    else:
        raise Exception("The CSV file must contain a 'text', 'comment', 'feedback', or 'message' column.")

    df['clean_text'] = df['text'].astype(str).apply(lambda x: re.sub(r'\W+', ' ', x).lower())
    df['tokens'] = df['clean_text'].apply(lambda x: [word for word in x.split() if word not in stop_words])
    df['sentiment_data'] = df['clean_text'].apply(get_vader_sentiment)
    df['compound'] = df['sentiment_data'].apply(lambda x: x[0])
    df['sentiment'] = df['sentiment_data'].apply(lambda x: x[1])
    return df

def generate_wordcloud(df):
    text = ' '.join([' '.join(tokens) for tokens in df['tokens']])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    if not os.path.exists('static'):
        os.makedirs('static')
    wordcloud_path = os.path.join('static', 'wordcloud.png')
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(wordcloud_path)
    plt.close()
    return wordcloud_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploads', file.filename)
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            file.save(filepath)
            try:
                df = clean_dataframe(filepath)
                wordcloud_path = generate_wordcloud(df)
                sentiment_counts = df['sentiment'].value_counts().to_dict()
                return render_template('result.html', sentiment=sentiment_counts, wordcloud_url=wordcloud_path)
            except Exception as e:
                return str(e)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
