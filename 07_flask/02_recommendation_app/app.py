from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Essential for session management

@app.route('/')
def home():
    return render_template('home.html')  # Render input form

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['preferences']
    # Store user input in the session for state management
    session['preferences'] = user_input
    return redirect(url_for('results'))

@app.route('/results')
def results():
    user_input = session.get('preferences', None)
    if not user_input:
        return redirect(url_for('home'))
    # Run recommendation algorithm here
    recommendations = get_recommendations(user_input)
    return render_template('results.html', recommendations=recommendations)

# Mock dataset for content-based recommendation
data = pd.DataFrame({
    'item': ['Book1', 'Book2', 'Book3'],
    'features': ['fiction mystery', 'sci-fi action', 'romance comedy']
})

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the feature data
tfidf_matrix = tfidf_vectorizer.fit_transform(data['features'])

def get_recommendations(user_input):
    # Transform the user input to TF-IDF format
    user_input_tfidf = tfidf_vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and dataset items
    similarity_scores = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()

    # Add similarity scores to the data and sort for top recommendations
    data['similarity'] = similarity_scores
    recommended_items = data.sort_values(by='similarity', ascending=False)

    # Return top 5 recommendations as a dictionary
    return recommended_items[['item', 'similarity']].head(5).to_dict('records')

if __name__ == '__main__':
    app.run(debug=True)

