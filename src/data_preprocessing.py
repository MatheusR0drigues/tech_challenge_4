import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(input_path, output_path):
    """
    Normaliza os dados de preços de fechamento e salva o resultado.
    
    Args:
        input_path (str): Caminho para o CSV de entrada.
        output_path (str): Caminho para salvar o CSV processado.
    
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
    
    # Criar o diretório de saída, se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salvar o DataFrame processado
    df.to_csv(output_path)
    return output_path

if __name__ == "__main__":
    input_file = "data/raw/AAPL_historical.csv"
    output_file = "data/processed/scaled_prices.csv"
    
    print("Processando dados...")
    processed_path = preprocess_data(input_file, output_file)
    print(f"Dados processados salvos em {processed_path}")
