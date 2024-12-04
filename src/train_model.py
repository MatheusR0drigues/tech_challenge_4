import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_sequences(data, seq_length):
    """
    Cria sequências de dados para o modelo LSTM.
    
    Args:
        data (np.array): Dados de entrada.
        seq_length (int): Comprimento da sequência.
    
    Returns:
        np.array, np.array: Sequências de entrada (X) e saídas (y).
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Carregar os dados processados
df = pd.read_csv("data/processed/scaled_prices.csv", index_col="Date", parse_dates=True)

# Configurações
sequence_length = 60
data = df['Close'].values

# Criar sequências
X, y = create_sequences(data, sequence_length)
X = np.expand_dims(X, axis=-1)

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Construir o modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compilar e treinar
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# Salvar o modelo treinado
model.save("models/lstm_model.h5")
print("Modelo salvo em models/lstm_model.h5")

# Avaliar o modelo no conjunto de teste
y_pred = model.predict(X_test)  # Previsões feitas pelo modelo

# Calcular as métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

# Exibir os resultados
print(f"MAE (Erro Médio Absoluto): {mae}")
print(f"RMSE (Raiz do Erro Médio Quadrático): {rmse}")
print(f"MAPE (Erro Percentual Médio Absoluto): {mape}%")