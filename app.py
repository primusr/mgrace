import os
import pandas as pd
import nltk
from langdetect import detect
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords
from flask import Flask, request, render_template, redirect, url_for, flash

# Download required NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Flask setup
app = Flask(__name__)
app.secret_key = 'secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize tools and resources
stop_words = set(stopwords.words('english'))
filipino_keywords = [
    'ganda', 'gwapo', 'masaya', 'malungkot', 'mahal', 'galit', 'lungkot', 'takot',
    'kinakabahan', 'nainis', 'tuwa', 'kilig', 'saya', 'inip', 'pagod'
]
sia = SentimentIntensityAnalyzer()

# Sentiment detection functions
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def get_vader_sentiment(text):
    if detect_language(text) != 'en':
        return {}
    return sia.polarity_scores(text)

def get_textblob_sentiment(text):
    if detect_language(text) != 'en':
        return {}
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

def get_filipino_keywords(text):
    return [word for word in filipino_keywords if word in text.lower()]

# Feedback analysis logic
def analyze_feedback(df):
    results = []
    for _, row in df.iterrows():
        feedback = str(row['feedback'])
        lang = detect_language(feedback)
        result = {
            'feedback': feedback,
            'language': lang,
            'vader': get_vader_sentiment(feedback),
            'textblob': get_textblob_sentiment(feedback),
            'filipino_keywords': get_filipino_keywords(feedback)
        }
        results.append(result)
    return results

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected. Please upload a CSV file.')
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except pd.errors.ParserError:
            flash('CSV formatting error. Please check for missing quotes or delimiters.')
            return redirect(request.url)
        except Exception as e:
            flash(f'Unexpected error: {str(e)}')
            return redirect(request.url)

        if 'feedback' not in df.columns:
            flash("CSV must contain a 'feedback' column.")
            return redirect(request.url)

        results = analyze_feedback(df)
        return render_template('results.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
