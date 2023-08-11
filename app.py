import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('\', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(text)['compound']

        if score > 0:
            label = 'Positive'
        elif score == 0:
            label = 'Neutral'
        else:
            label = 'Negative'

        return render_template("index.html", text=text, label=label)

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
