# Projeto de Previsão de Preços de Ações com LSTM

## Visão Geral

Este projeto utiliza um modelo de redes neurais recorrentes (RNN), especificamente um modelo LSTM, para prever os preços de fechamento de ações da Apple (AAPL). O sistema foi projetado para ser usado via API, com funcionalidades de pré-processamento de dados, normalização e desnormalização automatizadas.

---

## Organização do Projeto

```
TECH_CHALLENGE_4/
│
├── data/
│   ├── processed/
│   │   └── scaled_prices.csv
│   ├── raw/
│   │   └── AAPL_historical.csv
│
├── models/
│   ├── lstm_model.h5
│   ├── scaler.pkl
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── uso_api.ipynb
│
├── src/
│   ├── api.py
│   ├── data_preprocessing.py
│   ├── fetch_data.py
│   ├── train_model.py
│   ├── requirements.txt
│
├── predict/
```

---

## Requisitos

Certifique-se de ter o seguinte instalado:
- Python 3.8+
- Bibliotecas listadas em `requirements.txt`

Instale as dependências com:
```bash
pip install -r src/requirements.txt
```

---

## Etapas do Projeto

### 1. Coleta de Dados
Os dados históricos de preços da AAPL foram baixados utilizando a biblioteca `yfinance`.

- **Período Treinado:** `2018-12-01` a `2024-08-01`

Script para baixar dados:
```python
symbol = "AAPL"
start_date = "2018-12-01"
end_date = "2024-08-01"
df = yf.download(symbol, start=start_date, end=end_date)
```

---

### 2. Pré-processamento de Dados
O pré-processamento remove cabeçalhos adicionais, renomeia colunas e normaliza os preços de fechamento.

Exemplo do pré-processamento:
```python
df.columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
df = df[["Close"]].copy()
```

---

### 3. Treinamento do Modelo
O modelo LSTM foi treinado com dados normalizados. Os hiperparâmetros incluem:
- **Sequence Length:** 60
- **Batch Size:** 32
- **Epochs:** 20

---

### 4. API Flask
A API Flask permite realizar previsões enviando dados brutos. Os valores são normalizados internamente antes de serem processados pelo modelo.

Rota principal:
- **Endpoint:** `/predict`
- **Método:** POST
- **Entrada:** JSON contendo uma lista de preços históricos
- **Saída:** Previsões desnormalizadas em dólares

---

### 5. Uso da API
Exemplo de uso com `requests`:
```python
payload = {"prices": prices}
response = requests.post(api_url, json=payload)
predictions = response.json()["predictions"]
```

---

### 6. Visualização
O script inclui uma função para visualizar a evolução dos preços reais e previstos:

```python
plt.figure(figsize=(12, 6))
sns.lineplot(data=df.iloc[:split_index], x="Date", y="Close", label="Dados Reais", color="blue")
sns.lineplot(data=df.iloc[split_index:], x="Date", y="Close", label="Previsões", color="orange")
plt.ylim(0, 270)  # Define os limites do eixo Y
plt.title("Evolução dos Preços com Previsões", fontsize=16)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Preço de Fechamento (Close)", fontsize=12)
plt.show()
```

---

## Avaliação do Modelo

Métricas de avaliação no conjunto de testes:
- **MAE:** 0.0220
- **RMSE:** 0.0285
- **MAPE:** 2.74%

---

## Execução

1. Baixe os dados e pre-processamento:
   ```bash
   python src/fetch_data.py
   python src/data_preprocessing.py
   ```

2. Treine o modelo:
   ```bash
   python src/train_model.py
   ```

3. Inicie a API:
   ```bash
   python src/api.py
   ```

4. Use a API via script ou ferramenta como Postman.