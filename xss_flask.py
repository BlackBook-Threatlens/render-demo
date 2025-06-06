from flask import Flask, request, jsonify
from xss_model import predict  # Import the predict function

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        text = data.get('text', '  ')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        Prediction = predict(text) # Call the predict function
        result = int(Prediciton[0])

        return jsonify({'result': result}), 200  # Return the integer prediction

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # debug=True for development
