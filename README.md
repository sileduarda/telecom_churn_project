# Previsão de Churn - Telecom

Este projeto foi desenvolvido com o objetivo de criar um modelo preditivo capaz de identificar clientes que têm maior risco de **churn** (cancelamento do serviço) em uma empresa de telecomunicações.

O projeto também conta com um **app interativo em Streamlit**, permitindo o upload de dados e a geração de predições de forma intuitiva.

---

## 🎯 Objetivos

- Explorar e entender o comportamento dos clientes de telecom.
- Pré-processar dados para análise e modelagem.
- Treinar modelos de machine learning para prever churn.
- Criar um aplicativo web interativo para gerar previsões em arquivos CSV.
- Tornar os resultados replicáveis e facilmente acessíveis por qualquer usuário.

---

## 📂 Estrutura do Repositório

```
telecom_churn_project/
├── app.py # Aplicativo Streamlit principal
├── model.pkl # Modelo treinado (Joblib)
├── requirements.txt # Dependências do projeto
├── data/ # Dados brutos e processados
│ ├── raw/ # Dados originais
│ └── processed/ # Dados tratados
├── reports/ # Scripts auxiliares e análises exploratórias
│ └── utils.py # Funções auxiliares
└── README.md # Documentação do projeto
```

---

## 🛠 Tecnologias e Bibliotecas Utilizadas

- Python 3.13
- Streamlit
- Pandas
- Scikit-learn
- Joblib

---

## ⚙️ Como usar

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/telecom_churn_project.git
cd telecom_churn_project
```
2. Instale as dependências:

```
pip install -r requirements.txt
```

3. Execute o aplicativo Streamlit:

```
streamlit run app.py
```
## Modelo
O modelo preditivo foi treinado com base nos dados disponíveis da empresa de telecom, utilizando técnicas de machine learning supervisionado para classificação binária (Churn ou Não Churn). Ele está salvo no arquivo model.pkl e é carregado dinamicamente pelo aplicativo. O modelo é capaz de gerar probabilidades de churn para cada cliente, permitindo uma análise mais detalhada e a definição de thresholds personalizados para classificação.

O dashboard interativo permite que os usuários explorem as métricas de desempenho do modelo, como matriz de confusão, precisão, recall e F1-score, além de visualizar a distribuição das probabilidades de churn. Isso facilita a tomada de decisões estratégicas para retenção de clientes e otimização de campanhas de marketing. 

Você também pode acessar o app pelo link: [https://projetochurn.streamlit.app](https://projetochurn.streamlit.app/) para experenciar a geração de predições em tempo real.