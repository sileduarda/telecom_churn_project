# 🚀 Previsão de Churn - Telecom

Este projeto tem como objetivo desenvolver um **modelo preditivo de churn** para uma empresa de telecomunicações, identificando clientes com maior risco de cancelar o serviço.  

Além do modelo, o projeto inclui um **dashboard interativo em Streamlit**, permitindo que qualquer usuário envie dados de clientes em CSV e obtenha predições em tempo real, métricas de desempenho do modelo e visualizações intuitivas da distribuição de risco.

Este trabalho demonstra habilidades avançadas em **ciência de dados aplicada a negócios**, incluindo **exploração de dados, engenharia de features, modelagem preditiva, avaliação de desempenho e deployment interativo**.

---

## 🎯 Objetivos do Projeto

- Explorar padrões de comportamento de clientes de telecom e identificar fatores que influenciam o churn.
- Realizar **pré-processamento robusto** e engenharia de features para alimentar modelos de machine learning.
- Treinar modelos supervisionados para **classificação binária de churn** e otimizar thresholds para diferentes cenários.
- Desenvolver um **dashboard interativo** que possibilita:
  - Upload de CSV para predições em tempo real.
  - Ajuste de thresholds para classificação.
  - Visualização de métricas de desempenho (precision, recall, F1-score, matriz de confusão).
  - Análise da distribuição de probabilidade de churn para decisões estratégicas.
- Garantir **reprodutibilidade** e **acessibilidade**, permitindo que qualquer stakeholder use o modelo sem necessidade de conhecimento técnico avançado.

---

## 📂 Estrutura do Repositório

```text
telecom_churn_project/
├── app.py                # Aplicativo Streamlit principal
├── model.pkl             # Modelo treinado (Joblib)
├── x_columns.pkl         # Lista de colunas utilizadas no treino (para validação de CSV)
├── requirements.txt      # Dependências do projeto
├── data/                 # Dados brutos e processados
│   ├── raw/              # Dados originais
│   └── processed/        # Dados tratados para modelagem
├── reports/              # Scripts auxiliares e análises exploratórias
│   └── utils.py          # Funções utilitárias
└── README.md             # Documentação do projeto
```
## 🛠 Tecnologias e Bibliotecas

* Python 3.13
* Streamlit – Dashboard interativo e deployment
* Pandas / NumPy – Manipulação de dados
* Scikit-learn – Modelagem preditiva e métricas
* Joblib – Serialização do modelo e listas de features
* Matplotlib / Seaborn (opcional) – Visualização complementa

## ⚙️ Como usar
1. Clonar o repositório
```
git clone https://github.com/seu-usuario/telecom_churn_project.git
cd telecom_churn_project
```
2. Instalar dependências
```
pip install -r requirements.txt
```
3. Rodar o aplicativo Streamlit
```
streamlit run app.py
```
4. Uso do dashboard:
- Faça upload de um arquivo CSV com os dados dos clientes (certifique-se de que as colunas correspondam às utilizadas no treinamento).
- Ajuste o threshold de classificação conforme necessário.
- Visualize as predições, métricas de desempenho e a distribuição de risco.
- Analise a distribuição de probabilidades de churn para identificar clientes de alto risco e tomar decisões estratégicas.

## 📈 Modelo
O modelo foi treinado com técnicas de machine learning supervisionado para classificação binária (Churn ou Não Churn).
Ele gera probabilidades de churn por cliente, permitindo:
* Definir thresholds personalizados.
* Avaliar impacto operacional (quantos clientes classificados como risco vs. churns reais capturados).
* Tomada de decisão baseada em métricas robustas e visualizações interativas.

## 🌐 Demonstração Online

Você pode acessar o dashboard interativo e experimentar o envio de dados em tempo real:
https://projetochurn.streamlit.app

