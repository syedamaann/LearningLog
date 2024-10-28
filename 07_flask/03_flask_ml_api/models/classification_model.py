import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
iris = load_iris()
X, y = pd.DataFrame(iris.data, columns=iris.feature_names), iris.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open('classification_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Function to make predictions
def predict_classification(input_data):
    model = pickle.load(open('classification_model.pkl', 'rb'))
    prediction = model.predict([input_data])
    return prediction[0]
