from flask import Flask, request, jsonify
from xss_model import predict  # Import the predict function

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        text_to_check = data.get('text')

        if not text_to_check:
            return jsonify({'error': 'No text provided'}), 400

        prediction = predict(text_to_check) # Call the predict function

        return jsonify({'prediction': prediction}), 200  # Return the integer prediction

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # debug=True for development
