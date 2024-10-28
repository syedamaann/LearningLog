*Description* 
Create a RESTful API that serves predictions from multiple machine learning models (e.g., classification, regression).

*Key Features*
Implement endpoints for different ML models (e.g., /predict_classification, /predict_regression).
Accept input data in JSON format and return predictions in JSON.
Include error handling for invalid input.

*Learning Outcomes*
Understand how to build and document RESTful APIs.
Learn about JSON data handling and HTTP methods in Flask.

---

- User accesses the Flask API by sending an HTTP POST request to a specific endpoint, such as `/predict_classification` or `/predict_regression`.
- Flask receives the request at the defined endpoint and reads the HTTP method used (POST).
- The `@app.route()` decorator maps each endpoint URL to a specific function in the application that processes the request.
- The request data, which contains input features for the model, is included in the body of the POST request as a JSON object.
- Flask’s `request.get_json()` function is used to parse the incoming JSON data from the POST request, making it accessible as a Python dictionary.
- The application then retrieves the data from the dictionary and formats it appropriately for the model’s input requirements.
- Based on the endpoint, Flask triggers the relevant function that loads or references the pre-trained machine learning model (classification or regression).
- The function preprocesses the data, if necessary, to match the model’s training format.
- The input data is passed to the model’s `predict()` method to generate a prediction.
- The prediction result, typically a numerical or categorical output, is converted into a JSON-compatible format.
- Flask’s `jsonify()` function is used to package the prediction result in a JSON response object, adhering to RESTful API standards.
- The API sends the JSON response back to the user, which includes the prediction result.
- Users can access the API using tools like `curl` or Postman to interact with endpoints directly.
- `curl` allows for testing by sending requests directly from the command line, which is useful for quickly verifying responses without a full UI.
- In `curl`, users specify the HTTP method, URL, and JSON data to be sent (e.g., `curl -X POST -H "Content-Type: application/json" -d '{"data": ...}' http://localhost:5000/predict_regression`).
- Postman provides a graphical interface for API testing, enabling detailed inspection of request and response data, and is often preferred by developers for ease of use and in-depth testing options.
- Flask handles any errors in input validation, ensuring that invalid or missing data receives an appropriate error message in the JSON response.
- The application handles a RESTful structure, meaning that each endpoint and method has a specific function, which makes the service modular and organized.
- The API is structured for scalability, allowing additional machine learning models or endpoints to be added as needed.
