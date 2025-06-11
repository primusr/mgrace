from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from googletrans import Translator
from textblob import TextBlob
import time
from sentiment_analysis.config import CONFIG

nltk.download('vader_lexicon', quiet=True)
vader = SentimentIntensityAnalyzer()
translator = Translator()

def get_vader_sentiment(text):
    try:
        lang = detect(text)
        if lang != 'en':
            time.sleep(CONFIG["api_call_delay"])
            text = translator.translate(text, dest='en').text
    except:
        pass
    score = vader.polarity_scores(text)['compound']
    label = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
    return score, label

def get_textblob_sentiment(text):
    try:
        lang = detect(text)
        if lang != 'en':
            time.sleep(CONFIG["api_call_delay"])
            text = translator.translate(text, dest='en').text
    except:
        pass
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity