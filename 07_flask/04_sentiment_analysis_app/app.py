from flask import Flask, render_template, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import render_template
import matplotlib.pyplot as plt
import io
import base64
plt.switch_backend('Agg')

app = Flask(__name__)
def plot_sentiment_pie_chart(sentiments):
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [sentiments.count('positive'), sentiments.count('negative'), sentiments.count('neutral')]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

@app.route("/")
def index():
    return render_template("index.html")  # Assuming an 'index.html' template exists


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment = analyze_sentiment(text)
    pie_chart = plot_sentiment_pie_chart([sentiment])
    return render_template('index.html', text=text, sentiment=sentiment, pie_chart=pie_chart)

