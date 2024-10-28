import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
housing = fetch_california_housing()
X, y = pd.DataFrame(housing.data, columns=housing.feature_names), housing.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Function to make predictions
def predict_regression(input_data):
    model = pickle.load(open('regression_model.pkl', 'rb'))
    prediction = model.predict([input_data])
    return prediction[0]
