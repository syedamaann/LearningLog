from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('iris_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)
            # Make predictions using the model
            predictions = model.predict(df)
            # Add predictions to the DataFrame
            df['Predictions'] = predictions
            return render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

