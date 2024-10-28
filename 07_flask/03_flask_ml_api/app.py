from flask import Flask, request, jsonify
from models.classification_model import predict_classification
from models.regression_model import predict_regression

app = Flask(__name__)

@app.route('/predict_classification', methods=['POST'])
def classify():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    features = data['features']
    prediction = predict_classification(features)
    return jsonify({'prediction': int(prediction)})

@app.route('/predict_regression', methods=['POST'])
def regress():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    features = data['features']
    prediction = predict_regression(features)
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
