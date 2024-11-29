import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from model_predict import predict_genetic_disorder  # Ensure this imports the function with on-demand model loading

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/genetic_prediction", methods=["POST"])
def genetic_predict():
    try:
        # Get input data from the request
        input_data = request.get_json()

        # Check if input_data is present
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Call the prediction function with the input data
        predicted_output = predict_genetic_disorder(input_data)

        # Return the prediction as a JSON response
        return jsonify(predicted_output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Get the port from environment variable (Render will set this)
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=False)  # Bind to 0.0.0.0 for external access
