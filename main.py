from Modelo_IA.src.predict import predict_message
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "El mensaje está vacío"}), 400
    
    result = predict_message(message)
    return jsonify({"message": message, "classification": result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
