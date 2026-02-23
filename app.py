import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import joblib
import os

# -------------------------------
# Configurações iniciais
# -------------------------------
st.set_page_config(page_title="Dashboard de Churn", layout="wide")
st.title("📊 Dashboard de Churn - Telecom")
st.write("Faça upload de CSV para predições em tempo real ou ajuste o threshold do modelo com dados de teste.")

# Slider de threshold
threshold = st.slider("Escolha o Threshold", 0.0, 1.0, 0.5, 0.05)

# -------------------------------
# Carregar modelo e colunas
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

# Modelo treinado
modelo_path = os.path.join(BASE_DIR, "model.pkl")
try:
    modelo = joblib.load(modelo_path)
except FileNotFoundError:
    st.error(f"Modelo não encontrado em {modelo_path}")
    st.stop()

# Colunas usadas no treino
x_columns_path = os.path.join(BASE_DIR, "x_columns.pkl")
try:
    x_columns = joblib.load(x_columns_path)
except FileNotFoundError:
    st.error(f"Lista de colunas do treino não encontrada em {x_columns_path}")
    st.stop()

# -------------------------------
# Upload de CSV
# -------------------------------
uploaded_file = st.file_uploader("📁 Envie um arquivo CSV com dados dos clientes para predição", type="csv")

if uploaded_file:
    # Leitura do CSV
    df_new = pd.read_csv(uploaded_file)
    st.write("✅ Dados carregados:")
    st.dataframe(df_new.head())

    # Mantém apenas as colunas usadas no treino
    missing_cols = [c for c in x_columns if c not in df_new.columns]
    if missing_cols:
        st.error(f"As seguintes colunas estão faltando no CSV enviado: {missing_cols}")
    else:
        df_new_clean = df_new[x_columns]  # mantém somente as colunas corretas

        # Predição em tempo real
        try:
            y_prob_new = modelo.predict_proba(df_new_clean)[:,1]
            y_pred_new = (y_prob_new >= threshold).astype(int)
            df_new["Predição_Churn"] = y_pred_new

            st.subheader("📈 Resultados das Predições")
            st.dataframe(df_new.head())

            st.write(f"Clientes classificados como risco: {y_pred_new.sum()}")
            st.write(f"Clientes classificados como não risco: {len(y_pred_new) - y_pred_new.sum()}")

            # Gráfico de distribuição de risco
            st.subheader("📊 Distribuição de Probabilidades")
            st.bar_chart(pd.Series(y_prob_new, name="Probabilidade de Churn"))

        except Exception as e:
            st.error(f"Erro ao gerar predições: {e}")

# -------------------------------
# Avaliação com dados de teste (fallback)
# -------------------------------
else:
    try:
        x_test = joblib.load(os.path.join(BASE_DIR, "x_test.pkl"))
        y_test = joblib.load(os.path.join(BASE_DIR, "y_test.pkl"))

        y_prob = modelo.predict_proba(x_test)[:,1]
        y_pred_custom = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred_custom)
        tn, fp, fn, tp = cm.ravel()

        st.subheader("📊 Avaliação com Dados de Teste")
        st.write(f"VN: {tn}, FP: {fp}, FN: {fn}, VP: {tp}")

        precision = precision_score(y_test, y_pred_custom)
        recall = recall_score(y_test, y_pred_custom)
        f1 = f1_score(y_test, y_pred_custom)

        st.subheader("📈 Métricas")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        st.subheader("📌 Impacto Operacional")
        st.write(f"Clientes classificados como risco: {tp + fp}")
        st.write(f"Churns capturados: {tp}")

    except FileNotFoundError:
        st.warning("Dados de teste não encontrados. Envie um CSV para realizar predições em tempo real.")