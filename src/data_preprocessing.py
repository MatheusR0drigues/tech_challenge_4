import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib  # Para salvar o scaler
import os

def preprocess_data(input_path, output_path, scaler_path):
    """
    Normaliza os dados de preços de fechamento e salva o resultado.
    
    Args:
        input_path (str): Caminho para o CSV de entrada.
        output_path (str): Caminho para salvar o CSV processado.
        scaler_path (str): Caminho para salvar o objeto MinMaxScaler.
    
    Returns:
        str: Caminho do arquivo processado.
    """
    # Carregar o CSV ignorando as duas primeiras linhas (que contêm cabeçalhos extras)
    df = pd.read_csv(input_path, skiprows=2, index_col=0, parse_dates=True)
    
    # Renomear as colunas para garantir consistência
    df.columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    
    # Filtrar apenas as colunas necessárias (no caso, "Close")
    df = df[["Close"]]
    
    # Converter os valores para float, caso necessário
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    
    # Remover linhas com valores ausentes ou inválidos
    df = df.dropna()
    
    # Normalizar os dados usando MinMaxScaler
    scaler = MinMaxScaler()
    df["Close"] = scaler.fit_transform(df[["Close"]])
    
    # Criar os diretórios de saída, se não existirem
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    # Salvar o DataFrame processado
    df.to_csv(output_path)
    
    # Salvar o scaler para uso futuro
    joblib.dump(scaler, scaler_path)
    print(f"Scaler salvo em {scaler_path}")
    
    return output_path

if __name__ == "__main__":
    input_file = "data/raw/AAPL_historical.csv"
    output_file = "data/processed/scaled_prices.csv"
    scaler_file = "models/scaler.pkl"  # Caminho para salvar o scaler
    
    print("Processando dados...")
    processed_path = preprocess_data(input_file, output_file, scaler_file)
    print(f"Dados processados salvos em {processed_path}")
