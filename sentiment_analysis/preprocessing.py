import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 1]
    return ' '.join(tokens)

def clean_dataframe(df, preferred_cols):
    for col in preferred_cols:
        if col in df.columns:
            df['Feedback_Text'] = df[col]
            break
    else:
        df['Feedback_Text'] = df.iloc[:, 0]
    df['Original_Text'] = df['Feedback_Text']
    df = df[df['Feedback_Text'].notna()]
    return df[['Original_Text', 'Feedback_Text']]