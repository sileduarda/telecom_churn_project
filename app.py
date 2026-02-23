import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import joblib

# Carregar modelo treinado
modelo = joblib.load("model.pkl")

# Carregar dados de teste
x_test = joblib.load("x_test.pkl")
y_test = joblib.load("y_test.pkl")

# Gerar probabilidades
y_prob = modelo.predict_proba(x_test)[:,1]

st.title("Dashboard de Churn - Avaliação do Modelo")

st.write("Escolha o threshold para classificar clientes em risco.")

# Slider de threshold
threshold = st.slider("Escolha o Threshold", 0.0, 1.0, 0.5, 0.05)

# Aplicando threshold
y_pred_custom = (y_prob >= threshold).astype(int)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_custom)

tn, fp, fn, tp = cm.ravel()

# Métricas
precision = precision_score(y_test, y_pred_custom)
recall = recall_score(y_test, y_pred_custom)
f1 = f1_score(y_test, y_pred_custom)

st.subheader("📊 Matriz de Confusão")

st.write(f"VN (Verdadeiro Negativo): {tn}")
st.write(f"FP (Falso Positivo): {fp}")
st.write(f"FN (Falso Negativo): {fn}")
st.write(f"VP (Verdadeiro Positivo): {tp}")

st.subheader("📈 Métricas")

st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

st.subheader("📌 Impacto Operacional")

st.write(f"Clientes classificados como risco: {tp + fp}")
st.write(f"Churns capturados: {tp}")