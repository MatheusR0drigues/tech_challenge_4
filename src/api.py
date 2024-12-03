from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Carregar o modelo treinado e o escalador
MODEL_PATH = "models/lstm_model.h5"
model = load_model(MODEL_PATH)

# Função auxiliar para processar os dados
def preprocess_input(data, sequence_length=60):
    """
    Prepara os dados para entrada no modelo LSTM.
    
    Args:
        data (list): Lista de preços históricos.
        sequence_length (int): Comprimento da sequência usada pelo modelo.

    Returns:
        np.array: Sequência formatada para o modelo.
    """
    # Normalizar os dados
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))

    # Criar sequência
    X = []
    for i in range(len(data_scaled) - sequence_length + 1):
        X.append(data_scaled[i:i + sequence_length])
    return np.array(X), scaler

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para realizar previsões.
    """
    try:
        # Log dos cabeçalhos para debugging
        print(request.headers)

        # Tentar carregar o JSON da requisição
        request_data = request.get_json()
        if not request_data or "prices" not in request_data:
            return jsonify({"error": "Nenhum JSON válido foi enviado ou o campo 'prices' está ausente."}), 400

        prices = request_data["prices"]

        # Garantir que temos pelo menos 60 valores
        if len(prices) < 60:
            return jsonify({"error": "É necessário pelo menos 60 valores para previsão."}), 400

        # Pré-processar os dados
        X, scaler = preprocess_input(prices)

        # Fazer previsão
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)  # Desnormalizar os resultados

        # Retornar a previsão
        return jsonify({"predictions": predictions.flatten().tolist()})
    except Exception as e:
        # Garantir que um erro sempre retorne uma resposta
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)