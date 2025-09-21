from flask import Flask, request, render_template
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

nltk.download('stopwords')

set(stopwords.words('english'))
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    text1 = request.form['text1'].lower()

    processed_doc1 = ' '.join([word for word in text1.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)

    return render_template('form.html', final=compound, text1=text1)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)

'''
This Flask app performs sentiment analysis on user input text using the VADER sentiment analyzer,
with some preprocessing to remove stopwords.

1. Imports and Setup
Flask, request, render_template: For creating the web application, handling requests, rendering HTML templates.

nltk: Natural Language Toolkit, used here to load English stopwords.

vaderSentiment: A rule-based sentiment analysis tool optimized for social media texts.

The app downloads NLTK stopwords once with nltk.download('stopwords').

The Flask app is initialized with app = Flask(__name__).
'''