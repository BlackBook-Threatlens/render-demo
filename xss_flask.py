from flask import Flask, request, jsonify
from flask_cors import CORS
from xss_model import predict  # Import the predict function

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        text = data.get('text', '  ')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        prediction = predict(text) # Call the predict function
        result = int(prediction)

        return jsonify({'result': result}), 200  # Return the integer prediction

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)  # debug=True for development
