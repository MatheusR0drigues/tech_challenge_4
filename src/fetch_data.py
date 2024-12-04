import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(symbol, start_date, end_date, output_dir="data/raw"):
    """
    Coleta dados históricos de preços de ações usando yfinance.
    
    Args:
        symbol (str): Símbolo da empresa (ex.: 'AAPL' para Apple).
        start_date (str): Data inicial no formato 'YYYY-MM-DD'.
        end_date (str): Data final no formato 'YYYY-MM-DD'.
        output_dir (str): Diretório para salvar o arquivo CSV.

    Returns:
        str: Caminho do arquivo salvo.
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{symbol}_historical.csv")
    df.to_csv(output_path)
    return output_path

if __name__ == "__main__":
    # Configurações iniciais
    symbol = "AAPL"  
    start_date = "2018-12-01"
    end_date = "2024-08-01"
    
    print("Baixando dados...")
    file_path = fetch_stock_data(symbol, start_date, end_date)
    print(f"Dados salvos em {file_path}")
